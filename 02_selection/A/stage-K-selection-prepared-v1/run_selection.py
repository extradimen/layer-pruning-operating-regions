from pathlib import Path
import argparse,json,hashlib,time,gc,importlib,tarfile
import torch
from controller_identity import controller_identity
from journal import Journal
from cost_recorder import Costs,measured
from resource_probe import ResourceProbe
from selection import block_influence,greedy,random_mask,lowest

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);here=Path(__file__).resolve().parent;s=json.loads((here/'spec.json').read_text());start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for v in json.loads((here/'snapshot.json').read_text())['files']:assert sha(here/v['path'])==v['sha256']
 for v in s['inputs']:assert sha(root/v['path'])==v['sha256']
 rows=[json.loads(l) for l in (root/s['calibration']).read_text().splitlines()];assert len(rows)==96
 original={v['id']:v for v in map(json.loads,(root/s['original_texts']).read_text().splitlines())}
 ids=json.loads((root/s['calibration_ids']).read_text())['selections']
 for seed in [11,23,37]:
  selected=[v for v in rows if v['calibration_seed']==seed];assert [v['id'] for v in selected]==ids[str(seed)]['uniform']
  for v in selected:assert v['text']==original[v['id']]['text']
 c=controller_identity(s['project_id']);assert len(c['gpu_uuids'])==1;out=root/s['output'];assert not out.exists();report={'status':'RUNNING','controller':c,'Final_read':False,'endpoints':[],'model_reports':[]}
 def check():
  if time.monotonic()-start>s['soft_timeout_seconds']:raise TimeoutError('selection soft budget')
 try:
  for ms in s['models']:
   check();session=importlib.import_module(ms['name']+'_model_session');precision=importlib.import_module(ms['name']+'_precision_policy');precision.configure('BF16-strict-fp32-attention')
   j=Journal(root/(s['journal_prefix']+'-'+ms['name']),{'controller':c,'Final_read':False,'source_sha':s['protocol_sha']});cost=Costs(j,c['run_id'],gpu_uuids=c['gpu_uuids'],instrument_factory=ResourceProbe)
   model,tok,identity=session.load_ready(root/ms['model_directory'],ms['model_files'],j,cost);n=len(model.model.layers)
   for seed in [11,23,37]:
    documents=[]
    for row in [v for v in rows if v['calibration_seed']==seed]:
     tokens=tok.encode(row['text'],add_special_tokens=False)[:512];assert 2<=len(tokens)<=512
     if ms['name']=='mistral':assert tokens==row['tokens']
     key=hashlib.sha256(row['id'].encode()).hexdigest()[:16];documents.append((row['id'],tokens,key))
     j.complete_work(f'tokens-{seed}-{key}',{'id':row['id'],'seed':seed,'tokens':tokens,'token_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()})
    vectors=[]
    with cost.phase(f'BI-{seed}','prior','method_production') as metric:
     for id,tokens,key in documents:
      check();v=block_influence(model,tokens);vectors.append(v);j.complete_work(f'BI-{seed}-{key}',{'id':id,'seed':seed,'BI':v})
     metric['forward_calls']=measured(32,'one full forward per calibration document','calls')
    means={i:sum(v[i] for v in vectors)/32 for i in range(1,n-1)}
    chosen=[]
    for _ in range(3):i=lowest({k:v for k,v in means.items() if k not in chosen});chosen.append(i)
    report['endpoints'].append({'model':ms['name'],'seed':seed,'method':'shortgpt_bi_adapted','k':3,'mask':sorted(chosen),'BI_mean':means})
    evaluations=[]
    def evaluate(mask):
     vals=[];maskkey='-'.join(map(str,mask))
     with cost.phase(f'SLEB-{seed}-{maskkey}','prior','method_production') as metric:
      with session.physical_mask(model,mask):
       for id,tokens,key in documents:
        check();v=session.document_nll(model,tokens);vals.append(v['nll']);j.complete_work(f'SLEB-{seed}-{maskkey}-{key}',{'id':id,'seed':seed,'deleted':mask,**v})
      metric['forward_calls']=measured(32,'each candidate evaluated on all32 documents','calls')
     return sum(vals)/32
    def record(v):
     evaluations.append(v)
     if 'chosen' in v:print(json.dumps({'model':ms['name'],'seed':seed,'SLEB_step':v['step'],'chosen':v['chosen']}),flush=True)
    mask=greedy(n,3,evaluate,record);assert len([v for v in evaluations if 'candidate'in v])==sum(n-2-i for i in range(3))
    j.complete_work(f'SLEB-trajectory-{seed}',{'seed':seed,'evaluations':evaluations})
    report['endpoints'].append({'model':ms['name'],'seed':seed,'method':'sleb_iterative_adapted','k':3,'mask':mask,'candidate_evaluations':len([v for v in evaluations if 'candidate'in v])})
    with cost.phase(f'random-{seed}','prior','method_production'):mask=random_mask(session.MODEL_REVISION,seed,n,3)
    report['endpoints'].append({'model':ms['name'],'seed':seed,'method':'random','k':3,'mask':mask})
    print(json.dumps({'model':ms['name'],'seed_done':seed,'logical_endpoints':len(report['endpoints'])}),flush=True)
   report['model_reports'].append({'model':ms['name'],'base_identity':identity,'layers':n});j._commit('summary',{'endpoints':[v for v in report['endpoints'] if v['model']==ms['name']]});del model,tok;gc.collect();torch.cuda.empty_cache()
  assert len(report['endpoints'])==18;report['status']='PASS_18_BASELINE_K3_SELECTION_ENDPOINTS'
 except BaseException as e:report.update(status='FAIL',error=repr(e));raise
 finally:
  report['wall_seconds']=time.monotonic()-start;out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  archive=root/s['archive'];assert not archive.exists()
  with tarfile.open(archive,'w:gz') as tar:
   tar.add(out,arcname=str(out.relative_to(root)))
   for ms in s['models']:
    d=root/(s['journal_prefix']+'-'+ms['name'])
    if d.exists():tar.add(d,arcname=str(d.relative_to(root)))
  print(json.dumps({'status':report['status'],'archive_sha256':sha(archive),'wall_seconds':report['wall_seconds']}),flush=True)
if __name__=='__main__':main()

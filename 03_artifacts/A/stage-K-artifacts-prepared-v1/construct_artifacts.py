from pathlib import Path
import json,hashlib,argparse,time,shutil,gc,importlib,tarfile
import torch
from native_loader import load_native
from controller_identity import controller_identity
from journal import Journal
from cost_recorder import Costs,measured
from resource_probe import ResourceProbe

def probes(model):
 with torch.no_grad():
  ids=torch.tensor([[1,2,3,4,5,6,7,8]*2],device='cuda:0');p=model(input_ids=ids,use_cache=True);logits=[p.logits.detach().cpu()];cache=p.past_key_values
  for i in [9,10,11]:
   p=model(input_ids=torch.tensor([[i]],device='cuda:0'),past_key_values=cache,use_cache=True);cache=p.past_key_values;logits.append(p.logits.detach().cpu())
 return logits

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);code=Path(__file__).resolve().parent;s=json.loads((code/'spec.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
 for v in json.loads((code/'snapshot.json').read_text())['files']:assert sha(code/v['path'])==v['sha256']
 for v in s['inputs']:assert sha(root/v['path'])==v['sha256']
 plan=json.loads((root/s['point_plan']).read_text());assert len(plan['logical_points'])==24
 c=controller_identity(s['project_id']);assert len(c['gpu_uuids'])==1;out=root/s['output'];assert not out.exists();report={'status':'RUNNING','artifacts':[],'logical_points':[],'controller':c,'Final_read':False}
 def check():
  if time.monotonic()-start>s['soft_timeout_seconds']:raise TimeoutError('artifact soft cap')
 try:
  assert shutil.disk_usage(root).free>s['initial_disk_required_bytes']
  for ms in s['models']:
   check();session=importlib.import_module(ms['name']+'_model_session');importlib.import_module(ms['name']+'_precision_policy').configure('BF16-strict-fp32-attention')
   j=Journal(root/(s['journal_prefix']+'-'+ms['name']),{'controller':c,'Final_read':False});cost=Costs(j,c['run_id'],gpu_uuids=c['gpu_uuids'],instrument_factory=ResourceProbe)
   base,tok,identity=session.load_ready(root/ms['model_directory'],ms['model_files'],j,cost);total=sum(p.numel() for p in base.parameters());groups={}
   for row in plan['logical_points']:
    if row['model']!=ms['name']:continue
    assert len(row['mask'])==len(set(row['mask']))==3 and all(0<i<len(base.model.layers)-1 for i in row['mask'])
    key=hashlib.sha256(json.dumps({'model':ms['name'],'mask':sorted(row['mask'])},sort_keys=True).encode()).hexdigest();groups.setdefault(key,[]).append(row)
   for key,logical in sorted(groups.items()):
    check();assert shutil.disk_usage(root).free>s['disk_reserve_bytes']+16*1024**3;mask=sorted(logical[0]['mask']);reuse=next((v for v in plan['reuse_artifacts'] if v['model']==ms['name'] and v['mask']==mask),None);directory=root/(reuse['path'] if reuse else str(Path(s['artifact_root'])/key))
    if reuse:
     for v in reuse['members']:assert sha(directory/v['path'])==v['sha256']
    else:directory.mkdir(parents=True,exist_ok=False)
    removed_expected=sum(p.numel() for i in mask for p in base.model.layers[i].parameters())
    with cost.phase('export-'+key,'export_hash','method_production'):
     with session.physical_mask(base,mask):
      expected={k:v for k,v in base.state_dict().items()};parameters=sum(p.numel() for p in base.parameters());reference=probes(base)
      if not reuse:base.save_pretrained(directory,safe_serialization=True,max_shard_size='2GB')
     tokenizer_names=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','tokenizer.model.v3'] if ms['name']=='mistral' else ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
     for name in ([] if reuse else tokenizer_names):shutil.copyfile(root/ms['model_directory']/name,directory/name)
     for name in ([] if reuse else ['native_loader.py',ms['name']+'_precision_policy.py','tokenizer_loader.py']):shutil.copyfile(code/name,directory/name)
    with cost.phase('strict-reload-'+key,'reload','research_evaluation') as metrics:
     native=load_native(directory,'cuda:0');actual=native.state_dict();assert set(actual)==set(expected)
     for k in expected:assert torch.equal(actual[k],expected[k]),k
     assert sum(p.numel() for p in native.parameters())==parameters and total-parameters==removed_expected
     result=probes(native);errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(reference,result)];assert errors==[0.0]*4
     metrics['forward_calls']=measured(4,'native prefill and3cached decode','calls')
     del actual,native,expected,reference,result;gc.collect();torch.cuda.empty_cache()
    members=reuse['members'] if reuse else [{'path':str(p.relative_to(directory)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(directory.rglob('*')) if p.is_file()];bundle=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    artifact={'model':ms['name'],'topology_id':key,'mask':mask,'path':str(directory.relative_to(root)),'bundle_SHA256':bundle,'members':members,'native_bytes':sum(v['bytes'] for v in members),'parameters':parameters,'removed_parameters':total-parameters,'retained_tensor_exact':True,'strict_reload':True,'prefill_cached_max_abs_errors':errors,'logical_consumers':logical}
    artifact['reused_existing_weights']=reuse is not None
    if not reuse:(directory/'bundle_manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    j.complete_work('artifact-'+key,artifact);report['artifacts'].append(artifact)
    for v in logical:report['logical_points'].append(dict(v,topology_id=key,bundle_SHA256=bundle,path=str(directory.relative_to(root))))
    print(json.dumps({'model':ms['name'],'artifacts_done':len(report['artifacts']),'logical_done':len(report['logical_points'])}),flush=True)
   j._commit('summary',{'model':ms['name'],'unique':len(groups)});del base,tok;gc.collect();torch.cuda.empty_cache()
  assert len(report['logical_points'])==24;report['status']='PASS_K3_24_LOGICAL_STRICT_NATIVE_ARTIFACTS'
 except BaseException as e:report.update(status='FAIL',error=repr(e));raise
 finally:
  report['wall_seconds']=time.monotonic()-start;out.write_text(json.dumps(report,indent=2)+'\n')
  with tarfile.open(root/s['archive'],'w:gz') as tar:
   tar.add(out,arcname=str(out.relative_to(root)))
   for ms in s['models']:
    d=root/(s['journal_prefix']+'-'+ms['name'])
    if d.exists():tar.add(d,arcname=str(d.relative_to(root)))
if __name__=='__main__':main()

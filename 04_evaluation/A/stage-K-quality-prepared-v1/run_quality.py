from pathlib import Path
import json,time,hashlib,argparse,gc,tarfile,importlib
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from native_loader import load_native
from tokenizer_loader import load_tokenizer
from task_evaluation_v2 import evaluate_mc,evaluate_math
from controller_identity import controller_identity
from journal import Journal
from cost_recorder import Costs,measured
from resource_probe import ResourceProbe

def proof(model):
 with torch.no_grad():return model(input_ids=torch.tensor([[1,2,3,4,5,6,7,8]],device='cuda:0'),use_cache=False).logits.cpu()
def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);code=Path(__file__).parent;s=json.loads((code/'spec.json').read_text());start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for v in json.loads((code/'snapshot.json').read_text())['files']:assert sha(code/v['path'])==v['sha256']
 for v in s['inputs']:assert sha(root/v['path'])==v['sha256']
 c=controller_identity(s['project_id']);assert c['gpu_uuids']==s['required_GPU_UUIDs'];out=root/s['output'];assert not out.exists();source=json.loads((root/s['artifacts_report']).read_text());assert source['status']=='PASS_K3_24_LOGICAL_STRICT_NATIVE_ARTIFACTS'
 report={'status':'RUNNING','controller':c,'models':{},'logical_points':source['logical_points'],'new_rows':0,'reused_physical_rows':0,'quality_selection':False};reuse=json.loads((root/s['quality_reuse']).read_text())
 def check():
  if time.monotonic()-start>s['soft_timeout_seconds']:raise TimeoutError('whole quality soft cap')
 try:
  for ms in s['models']:
   name=ms['name'];session=importlib.import_module(name+'_model_session');importlib.import_module(name+'_precision_policy').configure('BF16-strict-fp32-attention');base=root/ms['model_directory'];session.verify_files(base,ms['model_files'])
   tok=load_tokenizer(base) if name=='mistral' else AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
   j=Journal(root/(s['journal_prefix']+'-'+name),{'controller':c,'split':'E1_followup','all_configurations_retained':True});j.model_ready({'base_verified':True,'stage':'quality','source':'separate evaluation origin; not method production reset'});cost=Costs(j,c['run_id'],gpu_uuids=c['gpu_uuids'],instrument_factory=ResourceProbe)
   with cost.phase('base-load','reload','research_evaluation'):dense=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval()
   datasets={t:[json.loads(l) for l in (root/path).read_text().splitlines()] for t,path in ms['task_files'].items()};assert {t:len(v) for t,v in datasets.items()}=={'piqa':128,'hellaswag':128,'gsm8k':32,'c4':64};demos=[json.loads(l) for l in (root/s['demonstration_file']).read_text().splitlines()]
   prior=json.loads((root/reuse[name]['report_path']).read_text());entry={'dense':prior['dense'],'artifacts':{}};report['models'][name]=entry;report['reused_physical_rows']+=352
   for a in [v for v in source['artifacts'] if v['model']==name]:
    check();key=a['topology_id'];old=next((v for v in reuse[name]['artifacts'] if v['mask']==a['mask']),None)
    if old:
     value=prior['artifacts'][old['old_key']];assert value['GPU_reload_max_abs_error']==0
     entry['artifacts'][key]={'quality':value['quality'],'bundle_SHA256':a['bundle_SHA256'],'reuse_source_report':reuse[name]['report_path'],'reuse_source_key':old['old_key'],'new_rows':0};report['reused_physical_rows']+=352;continue
    directory=root/a['path']
    for v in a['members']:assert sha(directory/v['path'])==v['sha256']
    with cost.phase(key+'-load','reload','research_evaluation'):model=load_native(directory,'cuda:0')
    with cost.phase(key+'-GPUproof','readiness_selfcheck','research_evaluation'):
     actual=proof(model);model.to('cpu');torch.cuda.empty_cache()
     with session.physical_mask(dense,a['mask']):dense.to('cuda:0');reference=proof(dense);dense.to('cpu');torch.cuda.empty_cache()
     error=float((actual.float()-reference.float()).abs().max());assert error==0;model.to('cuda:0');j.complete_work(key+'-GPUproof',{'max_abs_error':error,'bundle_SHA256':a['bundle_SHA256']});del actual,reference
    result={}
    for task,rows in datasets.items():
     values=[]
     for index,row in enumerate(rows):
      check();wid=key+'-'+task+'-'+str(index)
      with cost.phase(wid,'final_quality','research_evaluation') as metrics:
       if task=='c4':value=dict(session.document_nll(model,row['tokens']),id=row['id'])
       elif task=='gsm8k':value=evaluate_math(model,tok,row,demos)
       else:value=evaluate_mc(model,tok,row)
       metrics['input_tokens']=measured(value['input_tokens'],'explicit inputs','tokens');metrics['forward_calls']=measured(value['actual_forward_calls'],'actual scoring steps','calls');j.complete_work(wid,value)
      values.append(value);report['new_rows']+=1
     result[task]=values
    entry['artifacts'][key]={'quality':result,'bundle_SHA256':a['bundle_SHA256'],'GPU_reload_max_abs_error':error,'new_rows':352};model.to('cpu');del model;gc.collect();torch.cuda.empty_cache()
    print(json.dumps({'model':name,'models_done':len(entry['artifacts']),'new_rows':report['new_rows'],'reused_rows':report['reused_physical_rows']}),flush=True)
   del dense,tok;gc.collect();torch.cuda.empty_cache();j._commit('summary',{'new_rows':report['new_rows'],'model':name})
  assert report['new_rows']==s['expected_new_rows'];report['status']='PASS_K3_ALL_24_LOGICAL_QUALITY_WITH_AUDITED_REUSE'
 except BaseException as e:report.update(status='FAIL',error=repr(e));raise
 finally:
  report['wall_seconds']=time.monotonic()-start;out.write_text(json.dumps(report,indent=2)+'\n')
  with tarfile.open(root/s['archive'],'w:gz') as tar:
   tar.add(out,arcname=str(out.relative_to(root)))
   for ms in s['models']:
    d=root/(s['journal_prefix']+'-'+ms['name'])
    if d.exists():tar.add(d,arcname=str(d.relative_to(root)))
if __name__=='__main__':main()

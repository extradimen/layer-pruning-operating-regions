from pathlib import Path
import json,time,hashlib,argparse,gc,tarfile,importlib
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from controller_identity import controller_identity
from journal import Journal
from cost_recorder import Costs
from resource_probe import ResourceProbe
from native_loader import load_native
from tokenizer_loader import load_tokenizer
from one_active_pool import FullSwapPool
from paired_native_serving import paired_workload

def probe(model):
 with torch.no_grad():return model(input_ids=torch.tensor([[1,2,3,4,5,6,7,8]],device='cuda:0'),use_cache=False).logits.cpu()
def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);code=Path(__file__).parent;s=json.loads((code/'spec.json').read_text());start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for v in json.loads((code/'snapshot.json').read_text())['files']:assert sha(code/v['path'])==v['sha256']
 for v in s['inputs']:assert sha(root/v['path'])==v['sha256']
 c=controller_identity(s['project_id']);assert c['gpu_uuids']==s['required_GPU_UUIDs'];out=root/s['output'];assert not out.exists();source=json.loads((root/s['artifacts_report']).read_text());assert source['status']=='PASS_STAGE_B_AND_A_64_PHYSICAL_AGGREGATE';assert len(source['artifacts'])==s['expected_unique'];report={'status':'RUNNING','controller':c,'artifacts':{},'logical_points':source['logical_points'],'Final_read':False};pool=None
 def check():
  if time.monotonic()-start>s['soft_timeout_seconds']:raise TimeoutError('paired whole pool cap')
 try:
  for ms in s['models']:
   name=ms['name'];session=importlib.import_module(name+'_model_session');importlib.import_module(name+'_precision_policy').configure('BF16-strict-fp32-attention');base=root/ms['model_directory'];session.verify_files(base,ms['model_files']);tok=load_tokenizer(base) if name=='mistral' else AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
   j=Journal(root/(s['journal_prefix']+'-'+name),{'controller':c,'Final_read':False});cost=Costs(j,c['run_id'],gpu_uuids=c['gpu_uuids'],instrument_factory=ResourceProbe)
   dense=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval();j.model_ready({'base_verified':True,'stage':'shared_research_serving'})
   cal=[json.loads(l) for l in (root/s['calibration']).read_text().splitlines()];flat=[t for row in cal for t in tok.encode(row['text'],add_special_tokens=False)[:512]];assert len(flat)>=8192;rows=[{'tokens':flat[i*2048:(i+1)*2048]} for i in range(4)]
   for a in sorted([v for v in source['artifacts'] if v['model']==name],key=lambda v:v['topology_id']):
    check();key=a['topology_id'];directory=root/a['path']
    for v in a['members']:assert sha(directory/v['path'])==v['sha256']
    with cost.phase('load-'+key,'reload','research_evaluation'):candidate=load_native(directory,'cpu',len(dense.model.layers)-len(a['mask']));pool=FullSwapPool(dense,candidate)
    with cost.phase('native-reference-'+key,'readiness_selfcheck','research_evaluation'):
     actual=probe(pool.activate('candidate'));pool.release()
     with session.physical_mask(dense,a['mask']):dense.to('cuda:0');reference=probe(dense);dense.to('cpu');torch.cuda.empty_cache()
     error=float((actual.float()-reference.float()).abs().max());assert error==0;j.complete_work(key+'-GPUproof',{'max_abs_error':error,'bundle_SHA256':a['bundle_SHA256'],'same_UUID':c['gpu_uuids']});del actual,reference
    result={'model':name,'bundle_SHA256':a['bundle_SHA256'],'GPU_max_abs_error':error,'logical_consumers':a['logical_consumers'],'measurements':{}}
    for batch,prompt,label in [(1,512,'B1S512T128'),(4,2048,'B4S2048T128')]:result['measurements'][label]=paired_workload(pool,tok,rows,batch,prompt,key+'-'+label,j,cost,checkpoint=check)
    report['artifacts'][key]=result;j.complete_work(key+'-paired100',{'records':100});pool.release();pool=None;del candidate;gc.collect();torch.cuda.empty_cache();print(json.dumps({'model':name,'artifacts_done':len(report['artifacts']),'observations':len(report['artifacts'])*100}),flush=True)
   del dense,tok;gc.collect();torch.cuda.empty_cache()
  assert len(report['artifacts'])==s['expected_unique'];report.update(status='PASS_STAGE_B_ALL_K1_K2_K3_K4_PAIRED_SERVING',observations=len(report['artifacts'])*100)
 except BaseException as e:report.update(status='FAIL',error=repr(e));raise
 finally:
  if pool is not None:pool.release()
  report['wall_seconds']=time.monotonic()-start;out.write_text(json.dumps(report,indent=2)+'\n')
  with tarfile.open(root/s['archive'],'w:gz') as tar:
   tar.add(out,arcname=str(out.relative_to(root)))
   for ms in s['models']:
    d=root/(s['journal_prefix']+'-'+ms['name'])
    if d.exists():tar.add(d,arcname=str(d.relative_to(root)))
if __name__=='__main__':main()

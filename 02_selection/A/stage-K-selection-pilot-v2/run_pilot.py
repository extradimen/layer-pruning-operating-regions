from pathlib import Path
import argparse,json,hashlib,time,gc,importlib
import torch
from controller_identity import controller_identity
from journal import Journal
from cost_recorder import Costs
from resource_probe import ResourceProbe
from selection import block_influence

def fixtures():
 from transformers import Qwen2Config,Qwen2ForCausalLM,MistralConfig,MistralForCausalLM
 results=[]
 for name,C,M in [('qwen',Qwen2Config,Qwen2ForCausalLM),('mistral',MistralConfig,MistralForCausalLM)]:
  torch.manual_seed(11)
  c=C(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=5,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
  m=M(c).eval();ids=[1,2,3,4,5];session=importlib.import_module(name+'_model_session')
  bi=block_influence(m,ids)
  # Hidden states before final model norm; verify first four block hook values against independent output_hidden_states.
  with torch.no_grad():o=m(torch.tensor([ids]),output_hidden_states=True,use_cache=False)
  for i in range(4):
   x=o.hidden_states[i].float();y=o.hidden_states[i+1].float()
   reference=1-((x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-8)).mean().item()
   assert abs(reference-bi[i])<1e-6
  layers=list(m.model.layers);indices=[l.self_attn.layer_idx for l in layers];config=m.config
  try:
   with session.physical_mask(m,[1,3]):
    assert list(m.model.layers)==[layers[i] for i in [0,2,4]]
    assert [l.self_attn.layer_idx for l in m.model.layers]==[0,1,2]
    session.document_nll(m,ids)
    raise RuntimeError('intentional restoration fixture')
  except RuntimeError as e:assert str(e)=='intentional restoration fixture'
  assert list(m.model.layers)==layers and m.config is config
  assert [l.self_attn.layer_idx for l in layers]==indices
  assert not any(l._forward_hooks for l in layers)
  results.append({'model':name,'BI_reference_tolerance':1e-6,'mask_exception_restoration':True,'hook_cleanup':True})
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);code=Path(__file__).resolve().parent;s=json.loads((code/'spec.json').read_text());start=time.monotonic()
 for f in json.loads((code/'snapshot.json').read_text())['files']:assert hashlib.sha256((code/f['path']).read_bytes()).hexdigest()==f['sha256']
 identity=controller_identity(s['project_id']);assert len(identity['gpu_uuids'])==1
 out=root/s['output'];assert not out.exists();report={'status':'RUNNING','controller':identity,'synthetic_only':True,'Final_read':False,'models':[]}
 def check():
  if time.monotonic()-start>s['soft_timeout_seconds']:raise TimeoutError('pilot soft cap')
 try:
  report['fixtures']=fixtures()
  for ms in s['models']:
   check();session=importlib.import_module(ms['name']+'_model_session');precision=importlib.import_module(ms['name']+'_precision_policy');precision.configure('BF16-strict-fp32-attention')
   j=Journal(root/(s['journal_prefix']+'-'+ms['name']),{'controller':identity,'synthetic_only':True});cost=Costs(j,identity['run_id'],gpu_uuids=identity['gpu_uuids'],instrument_factory=ResourceProbe)
   model,tok,base=session.load_ready(root/ms['model_directory'],ms['model_files'],j,cost)
   tokens=[1,2]*256;rows=[]
   for repeat in range(3):
    check();torch.cuda.synchronize();begin=time.monotonic()
    with cost.phase('BI-'+str(repeat),'readiness_selfcheck','common'):bi=block_influence(model,tokens)
    torch.cuda.synchronize();rows.append({'kind':'BI','repeat':repeat,'seconds':time.monotonic()-begin,'layers':len(bi)})
    for mask in [[],[1],[1,2,3]]:
     check();torch.cuda.synchronize();begin=time.monotonic()
     with cost.phase('NLL-'+str(repeat)+'-'+str(len(mask)),'readiness_selfcheck','common'):
      with session.physical_mask(model,mask):nll=session.document_nll(model,tokens)
     torch.cuda.synchronize();rows.append({'kind':'NLL','repeat':repeat,'deleted_count':len(mask),'seconds':time.monotonic()-begin,'finite':True})
   report['models'].append({'name':ms['name'],'base_identity':base,'timings':rows,'parameters':sum(p.numel() for p in model.parameters())})
   j._commit('summary',report['models'][-1]);print(json.dumps({'model_done':ms['name'],'timings':rows}),flush=True)
   del model,tok,bi;gc.collect();torch.cuda.empty_cache()
  report['status']='PASS_TWO_MODEL_SELECTION_ENGINEERING'
 except BaseException as e:report.update(status='FAIL',error=repr(e));raise
 finally:
  report['wall_seconds']=time.monotonic()-start;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'wall_seconds':report['wall_seconds']}),flush=True)
if __name__=='__main__':main()

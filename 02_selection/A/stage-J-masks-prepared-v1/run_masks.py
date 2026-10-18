"""Governed scientific CPU mask search; no Final inputs and no model load."""
from pathlib import Path
import json,hashlib,argparse,time
from recipe_solver import solve_tabu
from starts import block_starts,branch_starts

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);here=Path(__file__).resolve().parent;s=json.loads((here/'spec.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for v in s['inputs']:assert sha(root/v['path'])==v['sha256']
 prior=json.loads((root/s['prior']).read_text());assert prior['status']=='PASS_96_DOCUMENTS_3_SEEDS_INDEPENDENT_BLOCK_BRANCH_PRIORS' and prior['documents_completed']==96 and prior['actual_calls']==18816
 out=root/s['output'];assert not out.exists();start=time.monotonic();records=[]
 for seed in [11,23,37]:
  for arm,k in [('k3',3),('k6',6),('k9',9),('branch',12)]:
   domain='branch' if arm=='branch' else 'block';v=prior['priors'][str(seed)][domain];assert v['documents']==32
   if domain=='block':starts=block_starts(k,seed);protected=[0,31];groups=None
   else:starts=branch_starts(seed);protected=[0,31,32,63];groups=[list(range(1,31)),list(range(33,63))]
   result=solve_tabu(v['a'],v['H'],k,starts,protected=protected,max_evaluations=100000,max_cpu_seconds=3600,quota_bins=groups)
   assert result['evaluations']==100000,'Frozen evaluation budget not completed; retain failure rather than substitute'
   mask=result['top_two'][0]['mask'];assert len(mask)==k and not set(mask)&set(protected)
   if domain=='branch':assert sum(i<32 for i in mask)==sum(i>=32 for i in mask)==6
   records.append({'arm':arm,'seed':seed,'domain':domain,'mask':mask,'solver':result,'q':0,'recovery_tokens':0,'Final_read':False});print(json.dumps({'done':len(records),'arm':arm,'seed':seed}),flush=True)
 report={'status':'PASS_FIXED_12_LOGICAL_MASKS_MODEL_OWN_PRIORS','records':records,'logical_compressed':12,'Dense':1,'all_configurations_retained':True,'Final_read':False,'prior_SHA256':sha(root/s['prior']),'wall_seconds':time.monotonic()-start};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

from pathlib import Path
import json,hashlib,random,time
from recipe_solver import solve_tabu
here=Path(__file__).resolve().parent;root=here.parents[1];s=json.loads((here/'spec.json').read_text())
for i in s['inputs']:assert hashlib.sha256((root/i['path']).read_bytes()).hexdigest()==i['sha256']
p=json.loads((root/s['prior_report']).read_text());results=[];start=time.time()
for seed in [11,23,37]:
 for k in [3,6,9]:
  rng=random.Random(seed);bins=[list(range(1,10)),list(range(10,19)),list(range(19,27))];starts=tuple(tuple(sorted(i for b in bins for i in rng.sample(b,k//3))) for _ in range(8));v=p['priors'][f'uniform-s{seed}']
  z=solve_tabu(v['a_by_objective']['NLL'],v['H'],k,starts,protected=(0,27),max_evaluations=100000,max_cpu_seconds=3600)
  assert z['evaluations']==100000 and all(x['evaluations']==12500 for x in z['restart_records'])
  x=dict(id=f'StageF-k{k}-s{seed}',seed=seed,k=k,mask=z['top_two'][0]['mask'],q=0,R=0,objective='NLL',selection='uniform',solver=z,calibration_IDs=v['doc_ids']);results.append(x);print(json.dumps({'done':len(results),'id':x['id'],'mask':x['mask'],'evaluations':z['evaluations']}),flush=True)
out=root/s['output'];out.parent.mkdir(exist_ok=True);assert not out.exists();out.write_text(json.dumps({'status':'PASS_STAGE_F9_COUNT_LIMITED_CPU_MASKS_NO_FINAL','results':results,'Final_read':False,'model_calls':0,'GPU_hours_charge':0,'wall_seconds':time.time()-start,'inputs':s['inputs']},indent=2)+'\n')

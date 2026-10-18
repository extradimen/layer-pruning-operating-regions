from pathlib import Path
import argparse,json,hashlib,csv,time
import numpy as np
from contrasts import METHODS,contrast,holm,cvar90
from cost_evidence import read_cost_events
p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);root=Path(p.parse_args().project_root);here=Path(__file__).resolve().parent;s=json.loads((here/'spec.json').read_text());start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for v in json.loads((here/'snapshot.json').read_text())['files']:assert sha(here/v['path'])==v['sha256']
for v in s['inputs']:assert sha(root/v['path'])==v['sha256']
q=json.loads((root/s['quality']).read_text());a=json.loads((root/s['artifacts']).read_text());paired=json.loads((root/s['paired']).read_text());assert q['status']=='PASS_K3_ALL_24_LOGICAL_QUALITY_WITH_AUDITED_REUSE' and paired['status']=='PASS_K3_ALL_PAIRED_SERVING';assert s['seed']==20261026 and s['replicates']==50000
lookup={(v['model'],v['method'],v['seed']):v['topology_id'] for v in a['logical_points']};assert len(lookup)==24;meta={v['topology_id']:v for v in a['artifacts']};seeds=[11,23,37];sizes={'piqa':128,'hellaswag':128,'gsm8k':32,'c4':64};cs=[];rows=[];serving={}
for key,v in paired['artifacts'].items():
 serving[key]={}
 for workload,obs in v['measurements'].items():
  D={x['trial']:x for x in obs if not x['warmup'] and x['model']=='dense'};C={x['trial']:x for x in obs if not x['warmup'] and x['model']=='candidate'};assert set(D)==set(C)==set(range(5,25))
  for t in D:assert D[t]['input_ids_sha256']==C[t]['input_ids_sha256'] and D[t]['pair_order']==C[t]['pair_order'] and D[t]['generated_tokens']==C[t]['generated_tokens']==128
  dv=np.array([D[t]['wall_seconds'] for t in sorted(D)]);cv=np.array([C[t]['wall_seconds'] for t in sorted(C)]);idx=np.random.default_rng(s['seed']).integers(0,20,size=(50000,20));ratio=dv[idx].mean(1)/cv[idx].mean(1)
  serving[key][workload]={'dense_mean_s':float(dv.mean()),'candidate_mean_s':float(cv.mean()),'speedup':float(dv.mean()/cv.mean()),'speedup_CI':np.quantile(ratio,[.025,.975]).tolist(),'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in C.values()),'pairs':20}
for name in ['qwen','mistral']:
 qm=q['models'][name];ids={t:sorted(v['id'] for v in vs) for t,vs in qm['dense'].items()};assert {t:len(v) for t,v in ids.items()}==sizes
 for key,v in qm['artifacts'].items():
  for t,vs in v['quality'].items():assert len(vs)==sizes[t] and sorted(x['id'] for x in vs)==ids[t] and len({x['id'] for x in vs})==sizes[t]
 for task in ['piqa','hellaswag','gsm8k']:
  arms={}
  for method in METHODS:
   vv=[]
   for seed in seeds:
    values={x['id']:x['correct'] for x in qm['artifacts'][lookup[name,method,seed]]['quality'][task]};vv.append([values[i] for i in ids[task]])
   arms[method]=vv
  for method in METHODS[1:]:
   diff=contrast(arms[METHODS[0]],arms[method]);idx=np.random.default_rng(s['seed']).integers(0,len(diff),size=(50000,len(diff)));obs=float(diff.mean());sim=(np.random.default_rng(s['seed']).choice([-1.,1.],size=(50000,len(diff)))*diff).mean(axis=1)
   cs.append({'model':name,'task':task,'comparison':'interaction_tabu-minus-'+method,'difference':obs,'nominal95CI':np.quantile(diff[idx].mean(1),[.025,.975]).tolist(),'p_two_sided':(int((np.abs(sim)>=abs(obs)-1e-12).sum())+1)/50001,'question_clusters':len(diff)})
 dense_nll={x['id']:x['nll'] for x in qm['dense']['c4']}
 for method,seed in [('Dense',None)]+[(m,s) for m in METHODS for s in seeds]:
  key=None if method=='Dense' else lookup[name,method,seed];v=qm['dense'] if key is None else qm['artifacts'][key]['quality'];gsm=v['gsm8k'];nll={x['id']:x['nll'] for x in v['c4']};delta=np.array([nll[i]-dense_nll[i] for i in ids['c4']]);parameters=7615616512 if name=='qwen' else 7248023552
  row={'model':name,'method':method,'seed':seed,'deleted_layers':0 if key is None else 3,'physical_key':key,'parameters':parameters if key is None else meta[key]['parameters'],'parameter_removed_ratio':0 if key is None else meta[key]['removed_parameters']/parameters,'C4_mean_NLL':float(np.mean(list(nll.values()))),'C4_delta_mean':float(delta.mean()),'C4_positive_tail_CVaR90':cvar90(np.maximum(delta,0)),'GSM_truncated':sum(bool(x['truncated']) for x in gsm),'GSM_parse_failures':sum(bool(x['parse_failure']) for x in gsm),'GSM_total_output_tokens':sum(x['output_tokens'] for x in gsm),'GSM_natural_wall_s_descriptive_only':sum(x['wall_seconds'] for x in gsm),'quality_reused':key is None or 'reuse_source_key' in qm['artifacts'][key]}
  for task in ['piqa','hellaswag','gsm8k']:row[task+'_accuracy']=sum(x['correct'] for x in v[task])/sizes[task]
  for workload in ['B1S512T128','B4S2048T128']:
   cell=dict(row,workload=workload)
   if key is not None:cell.update(serving[key][workload])
   rows.append(cell)
adj=holm([x['p_two_sided'] for x in cs])
for x,pv in zip(cs,adj):x.update(p_Holm18=pv,reject_Holm18=pv<.05)
assert len(rows)==52
cost_events=[]
for source in s['cost_archives']:
 cost_events.extend(read_cost_events(root/source['path'],source['sha256'],source['stage']))
lease_costs=[]
for source in s['cost_receipts']:
 receipt=json.loads((root/source['path']).read_text())
 amount=receipt.get('actual_GPUh',receipt.get('priors_GPU_hours'))
 assert amount is not None and amount>=0
 lease_costs.append({'source':source['path'],'role':source['role'],'run_id':receipt.get('run_id'),'actual_GPUh':amount,'OSwait':receipt.get('OSwait',receipt.get('OS_child_wait')),'shared_or_historical':source['shared_or_historical']})
assert len({x['source'] for x in lease_costs})==len(lease_costs)
result={'status':'PASS_K3_COMPLETE_ALL_METHOD_COMPARISON','contrasts':cs,'long_table':rows,'serving':serving,'cost_lease_lineage':lease_costs,'cost_event_count':len(cost_events),'cost_boundary':'Lease GPUh includes failure and setup; event wall/CPU is descriptive and must not be added to lease occupancy. Historical/shared priors are counted once, not assigned in full to each logical arm. Missing energy, price and CPU-child measurements remain unavailable.','family':{'size':18,'alpha':.05,'old_families_unchanged':True},'replicates':50000,'seed':20261026,'scope':'Previously observed E1 follow-up; fixed3seed mean question-cluster inference, no universal superiority or equivalence; reused quality natural wall not a controlled timing contrast','shared_physical_not_independent':True,'inputs':s['inputs'],'wall_seconds':time.monotonic()-start,'GPUh':0}
out=root/s['output'];out.mkdir(exist_ok=False);(out/'statistics.json').write_text(json.dumps(result,indent=2)+'\n');fields=sorted(set().union(*(row.keys() for row in rows)))
with (out/'all_configurations_long.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
if cost_events:
 with (out/'cost_events_long.csv').open('w') as f:
  fields=sorted(set().union(*(event.keys() for event in cost_events)))
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(cost_events)
print(json.dumps({'status':result['status'],'rows':len(rows),'contrasts':len(cs),'cost_events':len(cost_events),'cost_leases':len(lease_costs)}))

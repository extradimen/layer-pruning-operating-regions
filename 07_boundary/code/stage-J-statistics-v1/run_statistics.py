"""Frozen common-pool analysis. Only global-controller execution after SHA return."""
from pathlib import Path
import json,hashlib,time,csv
import numpy as np
from contrasts import LEVELS,DIRECTIONS,contrasts,holm,cvar90
here=Path(__file__).resolve().parent;root=here.parents[1];s=json.loads((here/'spec.json').read_text());start=time.monotonic()
for x in s['inputs']:assert hashlib.sha256((root/x['path']).read_bytes()).hexdigest()==x['sha256']
q=json.loads((root/s['quality']).read_text());plan=json.loads((root/s['plan']).read_text());assert q['status']=='PASS_ALL_MISTRAL_11_PHYSICAL_3872_ROWS_13_LOGICAL_CONFIGS' and len(q['artifacts'])==10 and len(plan['logical_points'])==12;assert s['replicates']==50000 and s['seed']==20261024
sizes={'piqa':128,'hellaswag':128,'gsm8k':32,'c4':64};ids={t:sorted(x['id'] for x in rows) for t,rows in q['dense'].items()};assert {t:len(v) for t,v in ids.items()}==sizes
for key,a in q['artifacts'].items():
 assert a['GPU_reload_max_abs_error']==0
 for t,rows in a['quality'].items():assert len(rows)==sizes[t] and sorted(x['id'] for x in rows)==ids[t] and len({x['id'] for x in rows})==sizes[t]
seeds=[11,23,37];lookup={(v['arm'],v['seed']):v['topology_id'] for v in plan['logical_points']};artifact_meta={v['topology_id']:v for v in plan['artifacts']};assert set(lookup)=={(arm,seed) for arm in LEVELS for seed in seeds};summary={};perseed=[];cs=[];raw=[1.]*9
for ti,task in enumerate(['piqa','hellaswag','gsm8k']):
 dense=np.array([next(x['correct'] for x in q['dense'][task] if x['id']==i) for i in ids[task]],float);idx=np.random.default_rng(s['seed']).integers(0,len(dense),size=(50000,len(dense)));arms={}
 for arm in LEVELS:
  columns=[]
  for seed in seeds:
   key=lookup[arm,seed];rows={x['id']:x for x in q['artifacts'][key]['quality'][task]};v=np.array([rows[i]['correct'] for i in ids[task]],float);columns.append(v);perseed.append({'arm':arm,'seed':seed,'task':task,'artifact':key,'accuracy':float(v.mean()),'correct':int(v.sum()),'n':len(v)})
  arms[arm]=np.array(columns)
 summary[task]={'n':len(dense),'dense_accuracy':float(dense.mean()),'arms':{arm:{'accuracy':float(v.mean()),'nominal95CI':np.quantile(v.mean(axis=0)[idx].mean(axis=1),[.025,.975]).tolist()} for arm,v in arms.items()}}
 for ci,diff in enumerate(contrasts(arms)):
  obs=float(diff.mean());sim=(np.random.default_rng(s['seed']).choice([-1.,1.],size=(50000,len(diff)))*diff).mean(axis=1);pv=(int(np.sum(np.abs(sim)>=abs(obs)-1e-12))+1)/50001;slot=ci*3+ti;raw[slot]=pv;cs.append({'task':task,'direction':DIRECTIONS[ci],'family_slot':slot,'accuracy_difference':obs,'nominal95CI':np.quantile(diff[idx].mean(axis=1),[.025,.975]).tolist(),'p_two_sided':pv,'question_clusters':len(diff)})
adj=holm(raw)
for c in cs:c.update(p_Holm9=adj[c['family_slot']],reject_Holm9=adj[c['family_slot']]<.05)
pool={};dense_nll={x['id']:x['nll'] for x in q['dense']['c4']}
for key,a in q['artifacts'].items():
 quality=a['quality'];nll={x['id']:x['nll'] for x in quality['c4']};delta=np.array([nll[i]-dense_nll[i] for i in ids['c4']]);gsm=quality['gsm8k'];pool[key]={'bundle_bytes':artifact_meta[key]['native_bytes'],'C4_mean_NLL':float(np.mean(list(nll.values()))),'C4_signed_delta_mean':float(delta.mean()),'C4_positive_delta_mean':float(np.maximum(delta,0).mean()),'C4_positive_delta_CVaR90':cvar90(np.maximum(delta,0)),'C4_paired_document_delta':[{'id':i,'delta':float(nll[i]-dense_nll[i])} for i in ids['c4']],'GSM_truncated':sum(bool(x['truncated']) for x in gsm),'GSM_parse_failures':sum(bool(x['parse_failure']) for x in gsm),'GSM_natural_total_wall_seconds':sum(x['wall_seconds'] for x in gsm),'GSM_output_tokens':[x['output_tokens'] for x in gsm],'logical_consumers':[p for p in plan['logical_points'] if p['topology_id']==key]}
result={'status':'PASS_FROZEN_MISTRAL_STAGE_J_COMMON_POOL_STATISTICS','validation':{'quality_records':3872,'physical_models':11,'compressed_logical':12,'shared_IDs_exact':True},'quality':summary,'per_seed_quality':perseed,'contrasts':cs,'pool':pool,'family':{'size':9,'alpha':.05,'raw_p':raw,'Holm_adjusted_p':adj,'old_families_unchanged':True},'replicates':50000,'seed':s['seed'],'inference_scope':'Conditional fixed3seed mean paired question clusters, not independent physical replicates; same-source holdout,17historicalgap/pretrainingunknown; noARC; no screening; not equivalence/fullspace','inputs':s['inputs'],'model':'Mistral-7B-Instruct-v0.3','cross_model_scope':'Separate within-model Holm9; no joint cross-model FWER or universal transfer claim','storage_scope':'SHA-manifested native payload bytes; bookkeeping bundle_manifest.json excluded and listed separately if full disk footprint reported','GPU_hours_charge':0,'wall_seconds':time.monotonic()-start}
out=root/s['output'];out.mkdir(exist_ok=False);(out/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'contrast_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['task','direction','family_slot','accuracy_difference','p_two_sided','p_Holm9']);w.writeheader();w.writerows({k:c[k] for k in w.fieldnames} for c in cs)
print(json.dumps({'status':result['status'],'comparisons':9,'outputs':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in out.iterdir()]}))

"""Execute only from an immutable, controller-admitted deployment."""
from pathlib import Path
import json,hashlib,csv,time
import numpy as np
from paired import transitions,paired_interval

def main():
 start=time.monotonic();here=Path(__file__).resolve().parent;root=here.parents[1];s=json.loads((here/'spec.json').read_text())
 for item in s['inputs']:assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
 definition=json.loads((root/s['definition']).read_text());assert definition['no_p_values'] and definition['bootstrap']['seed']==20261022 and definition['bootstrap']['replicates']==50000
 design=json.loads((root/s['design']).read_text());base=json.loads((root/s['baseline']).read_text());new=json.loads((root/s['diagnostic']).read_text());assert new['status']=='PASS_ALL128_NATURAL_LENGTH_DIAGNOSTIC'
 ids=design['selected_IDs'];assert len(ids)==len(set(ids))==16
 idx=np.random.default_rng(20261022).integers(0,16,size=(50000,16));summaries=[];comparisons=[];raw=[]
 for label,key in design['configs'].items():
  original=base['dense']['gsm8k'] if key is None else base['artifacts'][key]['quality']['gsm8k'];lookup={r['id']:r for r in original};observed=new['results'][label];assert observed['artifact']==key
  bylength={512:{i:lookup[i] for i in ids}}
  for length in [1024,2048]:
   rows=[r for r in observed['generations'] if r['max_new_tokens']==length];assert len(rows)==16 and {r['id'] for r in rows}==set(ids);bylength[length]={r['id']:r for r in rows}
  scores={}
  for length,rows in bylength.items():
   ordered=[rows[i] for i in ids];scores[length]=np.array([r['correct'] for r in ordered]);assert np.isin(scores[length],[0,1]).all()
   for i,r in zip(ids,ordered):
    assert r['rendered_input_sha256']==lookup[i]['rendered_input_sha256'];assert not r['truncated'] or r['correct']==0
    raw.append({'config':label,'artifact':key,'id':i,'length':length,**{k:r[k] for k in ['correct','truncated','parse_failure','output_tokens','wall_seconds','rendered_input_sha256']}})
   row={'config':label,'artifact':key,'length':length,'n':16,'correct':int(scores[length].sum()),'accuracy':float(scores[length].mean()),'truncated':sum(bool(r['truncated']) for r in ordered),'parse_failure':sum(bool(r['parse_failure']) for r in ordered),'flags_nonexclusive':True}
   for field in ['wall_seconds','output_tokens']:
    v=np.array([r[field] for r in ordered],float);row[field]={'total':float(v.sum()),'mean':float(v.mean()),'median':float(np.median(v)),'p90':float(np.quantile(v,.9))}
   summaries.append(row)
  for a,b in [(512,1024),(512,2048),(1024,2048)]:comparisons.append({'config':label,'from_length':a,'to_length':b,**transitions(scores[a],scores[b]),**paired_interval(scores[a],scores[b],idx)})
 assert len(raw)==192
 out=root/s['output'];out.mkdir(exist_ok=False);result={'status':'PASS_FROZEN_STAGE_I_LENGTH_DIAGNOSTIC_STATISTICS','definition':definition,'inputs':s['inputs'],'summaries':summaries,'comparisons':comparisons,'raw_pairs':raw,'new_generations':128,'reused_baseline_records':64,'GPU_hours_charge':0,'wall_seconds':time.monotonic()-start}
 (out/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
 with (out/'paired_records.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 print(json.dumps({'status':result['status'],'outputs':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in out.iterdir()]}))
if __name__=='__main__':main()

"""Atomic immutable work records and single model_ready origin per pipeline."""
from pathlib import Path
import os,json,hashlib,uuid,time,datetime

class Journal:
    def __init__(self,root,identity):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.identity=identity
        identity_file=self.root/'identity.json'
        if identity_file.exists():
            if json.loads(identity_file.read_text())!=identity: raise ValueError('journal identity changed')
        else: self._commit('identity',identity)
    def _commit(self,key,record):
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in key): raise ValueError('record key')
        target=self.root/(key+'.json'); temp=self.root/('.tmp-'+uuid.uuid4().hex)
        try:
            with temp.open('x') as f:
                json.dump(record,f,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.link(temp,target) # atomic no-overwrite commit on the same filesystem
            fd=os.open(self.root,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally:
            if temp.exists(): temp.unlink()
    def model_ready(self,probe_identity):
        self._commit('model_ready',{'identity':self.identity,'probe_identity':probe_identity,'monotonic_ns':time.monotonic_ns(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    def complete_work(self,work_id,payload,dependencies=()):
        for dep in dependencies:
            if not (self.root/('work-'+dep+'.json')).exists(): raise ValueError('incomplete dependency')
        ready=self.root/'model_ready.json'
        if not ready.exists(): raise ValueError('model_ready not recorded')
        self._commit('work-'+work_id,{'work_id':work_id,'identity':self.identity,'dependencies':list(dependencies),'model_ready_identity_sha256':hashlib.sha256(ready.read_bytes()).hexdigest(),'payload':payload})
    def completed(self):
        return {p.stem[5:]:json.loads(p.read_text()) for p in self.root.glob('work-*.json')}

def full_point_gate(records,expected,artifact_identity):
    """Exact IDs include warmups and timed requests; duplicate IDs fail."""
    failures=[]
    for kind,wanted in expected.items():
        rows=records.get(kind,[]); ids=[r.get('id') for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(wanted): failures.append(kind+':ID_SET')
        if any(r.get('artifact_identity')!=artifact_identity for r in rows): failures.append(kind+':ARTIFACT')
    return {'complete':not failures,'failures':failures,'deployment_pass':None}

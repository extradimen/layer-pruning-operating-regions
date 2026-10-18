"""Resolve own controller ancestor payload; never dump environment values."""
from pathlib import Path
import os,json
def controller_identity(project_id):
    pid=os.getpid()
    for _ in range(8):
        base=Path('/proc')/str(pid)
        try:
            command=[x.decode() for x in (base/'cmdline').read_bytes().split(b'\0') if x]
            if 'worker' in command:
                index=command.index('worker')
                if index+1<len(command):
                    path=Path(command[index+1])
                    if path.name=='payload.json' and path.is_absolute() and str(path).startswith('/root/.codex-experiment-control/'):
                        p=json.loads(path.read_text())
                        if p.get('project_id')!=project_id or Path(p['cwd']).resolve()!=Path.cwd().resolve():raise ValueError('own controller payload scope mismatch')
                        return {k:p[k] for k in ['run_id','fingerprint','node_id','gpu_ids','gpu_uuids']}
            status=(base/'status').read_text().splitlines();parent=int(next(x.split()[1] for x in status if x.startswith('PPid:')))
        except (FileNotFoundError,StopIteration):break
        if parent<=1 or parent==pid:break
        pid=parent
    raise RuntimeError('verified own global-controller worker ancestry not found')

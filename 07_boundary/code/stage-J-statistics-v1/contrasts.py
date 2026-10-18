import numpy as np
LEVELS=('k3','k6','k9','branch')
DIRECTIONS=('k6-k3','k9-k6','branch-k6block')
def contrasts(arms):
 if set(arms)!=set(LEVELS):raise ValueError('four fixed arms required')
 v={k:np.asarray(x,dtype=float) for k,x in arms.items()}
 if len({x.shape for x in v.values()})!=1 or any(x.ndim!=2 or x.shape[0]!=3 or not np.isfinite(x).all() for x in v.values()):raise ValueError('paired three fixed seeds by question required')
 v={k:x.mean(axis=0) for k,x in v.items()};return [v['k6']-v['k3'],v['k9']-v['k6'],v['branch']-v['k6']]
def holm(raw):
 if len(raw)!=9 or any(not 0<=p<=1 for p in raw):raise ValueError('fixed family9 required')
 out=[None]*9;running=0.
 for rank,i in enumerate(sorted(range(9),key=lambda i:(raw[i],i))):running=max(running,min(1.,(9-rank)*raw[i]));out[i]=running
 return out
def cvar90(values):
 v=np.sort(np.asarray(values,dtype=float))[::-1]
 if len(v)==0 or not np.isfinite(v).all():raise ValueError('finite nonempty observations required')
 mass=.1*len(v);n=int(np.floor(mass));frac=mass-n
 return float((v[:n].sum()+(frac*v[n] if frac>0 else 0))/mass)

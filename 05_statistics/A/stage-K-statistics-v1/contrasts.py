import numpy as np
METHODS=('interaction_tabu','shortgpt_bi_adapted','sleb_iterative_adapted','random')
def contrast(ours,other):
 a=np.asarray(ours,dtype=float);b=np.asarray(other,dtype=float)
 if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('paired three seeds by question required')
 return (a-b).mean(axis=0)
def holm(raw):
 if len(raw)!=18 or any(not 0<=p<=1 for p in raw):raise ValueError('frozen family18')
 out=[None]*18;running=0.
 for rank,i in enumerate(sorted(range(18),key=lambda i:(raw[i],i))):running=max(running,min(1.,(18-rank)*raw[i]));out[i]=running
 return out
def cvar90(values):
 v=np.sort(np.asarray(values,dtype=float))[::-1]
 if len(v)==0 or not np.isfinite(v).all():raise ValueError('finite observations')
 mass=.1*len(v);n=int(np.floor(mass));frac=mass-n
 return float((v[:n].sum()+(frac*v[n] if frac else 0))/mass)

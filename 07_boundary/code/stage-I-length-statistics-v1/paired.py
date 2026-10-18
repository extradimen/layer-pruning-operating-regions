"""Pure paired diagnostic helpers; no evidence execution on import."""
import numpy as np

def transitions(a,b):
 a=np.asarray(a,dtype=int);b=np.asarray(b,dtype=int)
 if a.shape!=b.shape or a.ndim!=1 or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():raise ValueError('paired binary vectors required')
 return {f'{i}_to_{j}':int(((a==i)&(b==j)).sum()) for i in [0,1] for j in [0,1]}

def paired_interval(a,b,indices):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if a.shape!=b.shape or a.ndim!=1:raise ValueError('paired vectors required')
 d=b-a
 return {'difference_pp':float(d.mean()*100),'nominal95CI_pp':(np.quantile(d[indices].mean(axis=1),[.025,.975])*100).tolist()}

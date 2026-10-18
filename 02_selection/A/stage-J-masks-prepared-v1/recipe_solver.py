"""Independent implementation of the frozen feasible-swap tabu recipe.

8 matched starts, <=100000 actual evaluations and <=10 process CPU seconds.
The tested toy search in core.py is not this scientific solver.
"""
import math,random,time
from core import validate_prior

class Exhausted(Exception): pass

def initial_pool(n,k,seed,restarts=8,protected=()):
    allowed=[i for i in range(n) if i not in protected]
    if not 0<=k<=len(allowed) or restarts<1: raise ValueError('initial pool')
    rng=random.Random(seed)
    return tuple(tuple(sorted(rng.sample(allowed,k))) for _ in range(restarts))

def solve_tabu(a,h,k,starts,*,protected=(),max_evaluations=100000,max_cpu_seconds=3600,quota_bins=None):
    validate_prior(a,h); n=len(a); protected=set(protected)
    if any(type(i) is not int or not 0<=i<n for i in protected): raise ValueError('protected unit')
    if type(k) is not int or not 0<=k<=n-len(protected): raise ValueError('cardinality')
    if not starts or max_evaluations<len(starts) or max_cpu_seconds<=0: raise ValueError('budget')
    for m in starts:
        if tuple(m)!=tuple(sorted(set(m))) or len(m)!=k or set(m)&protected or any(not 0<=i<n for i in m):
            raise ValueError('infeasible start')
    if quota_bins:
        groups={i:b for b,group in enumerate(quota_bins) for i in group}
        if set(groups)!=set(range(n))-protected:raise ValueError('quota domain')
        if any(any(sum(i in group for i in m)!=6 for group in quota_bins) for m in starts):raise ValueError('quota start')
    begin=time.process_time(); wall=time.monotonic(); count=0; cache={}; best=math.inf; records=[]
    quota=max_evaluations//len(starts); seconds=max_cpu_seconds/len(starts); tenure=5
    for restart,start in enumerate(starts):
        t=time.process_time(); before=count; status='converged'
        def evaluate(m):
            nonlocal count,best
            if count-before>=quota or time.process_time()-t>=seconds or time.process_time()-begin>=max_cpu_seconds or time.monotonic()-wall>=2*max_cpu_seconds+1:
                raise Exhausted()
            e=math.fsum(a[i] for i in m)+math.fsum(h[i][j] for pos,i in enumerate(m) for j in m[pos+1:])
            count+=1; cache[m]=e; best=min(best,e); return e
        try:
            current=tuple(start); evaluate(current); forbidden={}
            for step in range(quota):
                neighbors=[]; selected=set(current)
                for removed in current:
                    for added in range(n):
                        if added in selected or added in protected: continue
                        if quota_bins and groups[removed]!=groups[added]:continue
                        m=tuple(sorted((selected-{removed})|{added})); previous_best=best; e=evaluate(m)
                        if forbidden.get(added,-1)>step and e>=previous_best: continue
                        neighbors.append((e,m,removed))
                if not neighbors: break
                _,current,removed=min(neighbors)
                forbidden[removed]=step+tenure
        except Exhausted: status='budget_limited'
        records.append({'restart':restart,'evaluations':count-before,'cpu_seconds':time.process_time()-t,'status':status})
    proposals=sorted((e,m) for m,e in cache.items())[:2]
    return {'solver':'tabu','top_two':[{'energy':e,'mask':list(m)} for e,m in proposals],
            'evaluations':count,'cpu_seconds':time.process_time()-begin,'wall_seconds':time.monotonic()-wall,
            'initial_pool':[list(m) for m in starts],'restart_records':records,
            'tabu_tenure':tenure,'certified_global_optimum':False,'returned_candidates':'best actually evaluated; rank1 for main dose line'}

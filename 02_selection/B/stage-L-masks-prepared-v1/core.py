"""Independent mathematical core; no model, scheduler or historical imports."""
from itertools import combinations
from math import ceil, isfinite

def validate_prior(a, h):
    n=len(a)
    if not n or len(h)!=n or any(len(row)!=n for row in h):
        raise ValueError('shape')
    if any(not isfinite(x) for x in a) or any(not isfinite(x) for row in h for x in row):
        raise ValueError('nonfinite')
    if any(h[i][i]!=0 for i in range(n)) or any(h[i][j]!=h[j][i] for i in range(n) for j in range(n)):
        raise ValueError('prior must be symmetric with zero diagonal')

def edge_order(a,h):
    validate_prior(a,h)
    return tuple(sorted(combinations(range(len(a)),2),key=lambda ij:(-abs(h[ij[0]][ij[1]]),ij)))

def corrected(a,h,q,observe):
    """observe(i,j) returns true b_ij; q0 never touches observe."""
    order=edge_order(a,h)
    if type(q) is not int or not 0<=q<=len(order): raise ValueError('dose')
    out=[list(row) for row in h]
    for i,j in order[:q]:
        b=observe(i,j)
        if not isfinite(b): raise ValueError('nonfinite observation')
        out[i][j]=out[j][i]=b
    return out

def energy(mask,a,h):
    validate_prior(a,h)
    m=tuple(sorted(mask))
    if len(set(m))!=len(m) or any(type(i) is not int or not 0<=i<len(a) for i in m): raise ValueError('mask')
    return sum(a[i] for i in m)+sum(h[i][j] for i,j in combinations(m,2))

def exact_oracle(a,h,k):
    """Bounded test oracle, never the production tabu solver."""
    validate_prior(a,h)
    if type(k) is not int or not 0<=k<=len(a) or len(a)>12: raise ValueError('oracle bound')
    return min((energy(m,a,h),m) for m in combinations(range(len(a)),k))

def positive_cvar(deltas,alpha=.9):
    if not deltas or not 0<=alpha<1 or any(not isfinite(x) for x in deltas): raise ValueError('tail input')
    v=sorted((max(0,x) for x in deltas),reverse=True)
    mass=len(v)*(1-alpha)
    whole=int(mass)
    numerator=sum(v[:whole])
    if whole<len(v): numerator+=(mass-whole)*v[whole]
    return numerator/mass

def task_error(candidate,dense):
    if not 0<=candidate<=1 or not 0<=dense<=1: raise ValueError('accuracy')
    return {'absolute_error':1-candidate,'error_difference':dense-candidate,
            'multiple':(1-candidate)/(1-dense) if dense!=1 else None,
            'null_reason':'ZERO_DENSE_ERROR_DENOMINATOR' if dense==1 else None}

def tabu_search(a,h,k,initial,iterations,tenure):
    """Deterministic cardinality-preserving search with explicit fixed budget.

    Returns the best visited mask and the trace. No early quality stopping;
    ties use lexicographic masks. Tabu blocks reversal of recent membership
    changes, with aspiration only for strict global energy improvement.
    This interface is not yet bound to a frozen scientific solver schedule.
    """
    validate_prior(a,h)
    if type(k) is not int or not 0<=k<=len(a): raise ValueError('cardinality')
    if type(iterations) is not int or iterations<0 or type(tenure) is not int or tenure<0:
        raise ValueError('search budget')
    current=tuple(sorted(initial))
    current_energy=energy(current,a,h)
    if len(current)!=k: raise ValueError('initial cardinality')
    best=(current_energy,current); until={}; trace=[]; evaluations=1
    for step in range(iterations):
        selected=set(current); candidates=[]
        for removed in current:
            for added in range(len(a)):
                if added in selected: continue
                m=tuple(sorted((selected-{removed})|{added}))
                e=energy(m,a,h); evaluations+=1
                blocked=until.get(removed,-1)>step or until.get(added,-1)>step
                if not blocked or e<best[0]: candidates.append((e,m,removed,added))
        if not candidates:
            trace.append({'iteration':step,'mask':current,'energy':current_energy,'move':None})
            continue
        current_energy,current,removed,added=min(candidates)
        until[removed]=until[added]=step+tenure+1
        best=min(best,(current_energy,current))
        trace.append({'iteration':step,'mask':current,'energy':current_energy,'move':(removed,added)})
    return {'energy':best[0],'mask':best[1],'trace':trace,'energy_evaluations':evaluations}

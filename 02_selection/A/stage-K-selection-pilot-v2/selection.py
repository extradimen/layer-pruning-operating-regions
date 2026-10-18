"""Controlled whole-block selection adapters; no evaluation-data access."""
import hashlib, math

def random_mask(revision,seed,n,k):
    legal=list(range(1,n-1))
    if not 0<k<=len(legal):raise ValueError('k')
    return sorted(sorted(legal,key=lambda i:hashlib.sha256(f'selection-supplement-v1|{revision}|{seed}|{i}'.encode()).hexdigest())[:k])

def lowest(scores):
    if not scores or any(not math.isfinite(v) for v in scores.values()):raise ValueError('invalid scores')
    return min(scores,key=lambda i:(scores[i],i))

def greedy(n,k,evaluate,record=lambda x:None):
    deleted=[]
    for step in range(k):
        scores={}
        for i in range(1,n-1):
            if i in deleted:continue
            scores[i]=evaluate(sorted(deleted+[i]))
            record({'step':step,'candidate':i,'deleted':sorted(deleted+[i]),'score':scores[i]})
        chosen=lowest(scores);deleted.append(chosen)
        record({'step':step,'chosen':chosen,'deleted':sorted(deleted)})
    return sorted(deleted)

def block_influence(model,tokens):
    import torch
    values={};handles=[]
    def hook(i):
        def f(module,args,kwargs,output):
            x=(args[0] if args else kwargs['hidden_states']).float()
            y=(output[0] if isinstance(output,tuple) else output).float()
            if x.shape!=y.shape:raise ValueError('BI shape')
            denom=(x.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-8)
            v=1-((x*y).sum(-1)/denom).mean()
            if not torch.isfinite(v):raise FloatingPointError('BI nonfinite')
            if i in values:raise ValueError('layer invoked twice')
            values[i]=v.item()
        return f
    try:
        for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(hook(i),with_kwargs=True))
        with torch.no_grad():
            ids=torch.tensor([tokens],device=next(model.parameters()).device)
            model(input_ids=ids,use_cache=False)
        if set(values)!=set(range(len(model.model.layers))):raise ValueError('incomplete BI hooks')
        return values
    finally:
        for h in handles:h.remove()

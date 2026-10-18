"""Verified base loading and physical mask acquisition; no downloader.

Production use requires a controller CUDA lease, immutable artifact manifest,
new data roles and a cost-profile contract. Loading does not admit science.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib,copy,json,os
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
PARAMETERS=7615616512

def verify_files(directory,files):
    root=Path(directory).resolve();wanted={r['path'] for r in files}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=wanted:raise ValueError('base artifact member set')
    for r in files:
        p=root/r['path']
        if p.is_symlink() or p.stat().st_size!=r['bytes']:raise ValueError('base artifact size/symlink')
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=r['sha256']:raise ValueError('base artifact SHA')
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_ready(directory,files,journal,costs):
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:raise ValueError('one controller-allocated CUDA device required')
    if (journal.root/'model_ready.json').exists():raise ValueError('use explicit resumed reload P phase; do not reset t0')
    with costs.phase('base-identity','setup','common'):
        identity=verify_files(directory,files)
    with costs.phase('initial-base-load','base_load','common'):
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
        if model.config.model_type!='qwen2' or len(model.model.layers)!=28 or sum(p.numel() for p in model.parameters())!=PARAMETERS:raise ValueError('fixed model identity')
        if any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('fixed BF16')
    with costs.phase('readiness-probe','readiness_selfcheck','common') as metrics:
        # Predeclared common minimum probe, not calibration/teacher/capability/warmup.
        ids=torch.tensor([[1,2,3,4,5,6,7,8]],device='cuda:0',dtype=torch.long)
        with torch.no_grad():out=model(input_ids=ids,use_cache=False).logits
        if not torch.isfinite(out).all():raise FloatingPointError('readiness nonfinite')
        from cost_recorder import measured
        metrics['forward_calls']=measured(1,'explicit model call','calls')
        metrics['input_tokens']=measured(8,'ids.numel','tokens')
        del out,ids;torch.cuda.synchronize()
    journal.model_ready({'base_manifest_identity':identity,'revision':MODEL_REVISION,'dtype':'BF16','blocks':28,'attention':'eager','probe_tokens':[1,2,3,4,5,6,7,8],'CUDA_synchronized':True})
    return model,tokenizer,identity

@contextmanager
def physical_mask(model,deleted):
    """Acquisition semantics only. Serving must use exported/reloaded artifact."""
    layers=model.model.layers;old_model_config=model.config;old_decoder_config=model.model.config
    n=len(layers);deleted=tuple(deleted)
    if deleted!=tuple(sorted(set(deleted))) or any(type(i)is not int or not 0<=i<n for i in deleted):raise ValueError('mask')
    retained=[i for i in range(n) if i not in deleted]
    if not retained:raise ValueError('empty model')
    old_indices=[l.self_attn.layer_idx for l in layers]
    config=copy.deepcopy(model.config);config.num_hidden_layers=len(retained)
    if getattr(config,'layer_types',None)is not None:config.layer_types=[old_model_config.layer_types[i] for i in retained]
    try:
        model.model.layers=torch.nn.ModuleList([layers[i] for i in retained]);model.config=config;model.model.config=config
        for i,l in enumerate(model.model.layers):l.self_attn.layer_idx=i
        yield
    finally:
        model.model.layers=layers;model.config=old_model_config;model.model.config=old_decoder_config
        for l,i in zip(layers,old_indices):l.self_attn.layer_idx=i

def document_nll(model,tokens):
    if model.training or len(tokens)<2:raise ValueError('eval next-token document')
    ids=torch.tensor([tokens],device=next(model.parameters()).device,dtype=torch.long)
    with torch.no_grad():
        logits=model(input_ids=ids,use_cache=False).logits[:,:-1].float()
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten(),reduction='sum')
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite NLL')
    return {'loss_sum':loss.item(),'effective_loss_tokens':len(tokens)-1,'nll':loss.item()/(len(tokens)-1),'input_tokens':len(tokens),'actual_forward_calls':1}

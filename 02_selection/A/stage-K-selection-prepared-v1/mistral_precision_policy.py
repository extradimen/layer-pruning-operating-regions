"""Explicit runtime numerical policy; never modifies verified package files."""
import torch
from transformers.models.mistral import modeling_mistral
ORIGINAL_EAGER=modeling_mistral.eager_attention_forward
def fp32_attention_forward(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
    dtype=query.dtype
    key=modeling_mistral.repeat_kv(key,module.num_key_value_groups)
    value=modeling_mistral.repeat_kv(value,module.num_key_value_groups)
    scores=torch.matmul(query.float(),key.float().transpose(2,3))*scaling
    if attention_mask is not None:scores=scores+attention_mask[:,:,:,:key.shape[-2]].float()
    probabilities=torch.nn.functional.softmax(scores,dim=-1,dtype=torch.float32)
    probabilities=torch.nn.functional.dropout(probabilities,p=dropout,training=module.training)
    output=torch.matmul(probabilities,value.float()).to(dtype)
    return output.transpose(1,2).contiguous(),probabilities
def configure(name):
    if name not in ['BF16-default','BF16-strict','BF16-strict-fp32-attention']:raise ValueError('explicit precision variant')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=name=='BF16-default'
    modeling_mistral.eager_attention_forward=fp32_attention_forward if name=='BF16-strict-fp32-attention' else ORIGINAL_EAGER
    return {'name':name,'parameter_storage_dtype':'BF16','allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'attention_accumulation_dtype':'FP32' if name.endswith('fp32-attention') else 'native','package_files_modified':False}

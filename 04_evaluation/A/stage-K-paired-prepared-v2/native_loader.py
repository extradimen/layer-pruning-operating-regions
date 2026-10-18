"""Whole-block-only strict native loader; no missing parameters allowed."""
from pathlib import Path
import json,torch
from safetensors.torch import load_file
from transformers import AutoConfig,Qwen2ForCausalLM,MistralForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

def load_native(directory,device):
 d=Path(directory);c=AutoConfig.from_pretrained(d,local_files_only=True,trust_remote_code=False);assert c.model_type in ['qwen2','mistral'];assert c.num_hidden_layers==({'qwen2':25,'mistral':29}[c.model_type]);c._attn_implementation='eager'
 cls,rotary=(Qwen2ForCausalLM,Qwen2RotaryEmbedding) if c.model_type=='qwen2' else (MistralForCausalLM,MistralRotaryEmbedding)
 with torch.device('meta'):m=cls(c)
 index=d/'model.safetensors.index.json';names=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors'];state={}
 for name in names:
  for k,v in load_file(d/name).items():
   assert k not in state;state[k]=v
 expected=m.state_dict();assert set(state)==set(expected)
 for k,v in state.items():assert v.dtype==torch.bfloat16 and v.shape==expected[k].shape
 m.load_state_dict(state,strict=True,assign=True);m.model.rotary_emb=rotary(config=c,device='cpu');assert not any(v.is_meta for v in list(m.parameters())+list(m.buffers()))
 return m.to(device).eval()

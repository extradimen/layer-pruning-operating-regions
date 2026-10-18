"""Model-specific official serialized fast backend. Acceptance pending governed parity proof."""
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers import Tokenizer

def load_tokenizer(directory):
 directory=Path(directory)
 tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,add_prefix_space=None)
 reference=Tokenizer.from_file(str(directory/'tokenizer.json'))
 for text in ['Hello',' Hello','  Hello','\nline\n','a\tb','你好','café','<s>test</s>','']:
  assert tokenizer.encode(text,add_special_tokens=False)==reference.encode(text,add_special_tokens=False).ids
  assert tokenizer.encode(text,add_special_tokens=True)==[tokenizer.bos_token_id]+tokenizer.encode(text,add_special_tokens=False)
 return tokenizer

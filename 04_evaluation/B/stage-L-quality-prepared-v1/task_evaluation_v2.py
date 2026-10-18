"""Frozen independent prompt/scorer; no downloader, no hidden data selection."""
import re,decimal,hashlib,json,time
import torch
PROMPT_VERSION='independent-five-example-math-chat-v1'
GENERATION={'do_sample':False,'num_beams':1,'max_new_tokens':512,'use_cache':True}
def numeric(value):
    value=str(value).strip().replace(',','').replace('$','')
    try:
        x=decimal.Decimal(value)
        return x if x.is_finite() else None
    except decimal.InvalidOperation:return None
def parse_math(text):
    matches=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',text)
    return numeric(matches[-1]) if matches else None
def math_messages(question,demonstrations):
    if len(demonstrations)!=5 or len({x['id'] for x in demonstrations})!=5:raise ValueError('five frozen examples')
    examples='\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in demonstrations)
    content='Solve the math problem. Show your reasoning and end with #### followed by the numerical answer.\n\n'+examples+'\n\nQuestion: '+question+'\nAnswer:'
    return [{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':content}]
def evaluate_math(model,tokenizer,row,demonstrations):
    messages=math_messages(row['question'],demonstrations)
    tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    ids=torch.tensor([tokens],device=next(model.parameters()).device,dtype=torch.long)
    attention=torch.ones_like(ids);begin=time.monotonic()
    config=__import__('transformers').GenerationConfig(**GENERATION,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,bos_token_id=tokenizer.bos_token_id)
    with torch.no_grad():generated=model.generate(input_ids=ids,attention_mask=attention,generation_config=config,use_model_defaults=False)
    if ids.device.type=='cuda':torch.cuda.synchronize()
    output=generated[0,len(tokens):].tolist();text=tokenizer.decode(output,skip_special_tokens=True);prediction=parse_math(text);gold=numeric(row['gold_numeric']);truncated=len(output)>=GENERATION['max_new_tokens'] and (not output or output[-1]!=tokenizer.eos_token_id)
    correct=int(prediction is not None and gold is not None and prediction==gold and not truncated)
    return {'id':row['id'],'correct':correct,'prediction_numeric':str(prediction) if prediction is not None else None,'gold_numeric':str(gold) if gold is not None else None,'parse_failure':prediction is None,'truncated':truncated,'generated_text':text,'input_tokens':len(tokens),'output_tokens':len(output),'actual_forward_calls':len(output),'wall_seconds':time.monotonic()-begin,'rendered_input_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'prompt_version':PROMPT_VERSION}
def evaluate_mc(model,tokenizer,row):
    context=row['context'];scores=[];input_counts=[]
    for choice in row['choices']:
        # Character-normalized continuation log likelihood, independently fixed.
        clean_context=context.rstrip();spacing=context[len(clean_context):];prefix=tokenizer.encode(clean_context,add_special_tokens=False);whole=tokenizer.encode(clean_context+spacing+' '+choice,add_special_tokens=False);continuation=whole[len(prefix):]
        if not prefix or not continuation:raise ValueError('empty MC input')
        tokens=prefix+continuation;ids=torch.tensor([tokens],device=next(model.parameters()).device,dtype=torch.long)
        with torch.no_grad():logits=model(input_ids=ids,use_cache=False).logits[0,len(prefix)-1:-1].float();labels=ids[0,len(prefix):];nll=torch.nn.functional.cross_entropy(logits,labels,reduction='sum')
        if not torch.isfinite(nll):raise FloatingPointError('MC nonfinite')
        scores.append(-float(nll) if row['task']=='piqa' else -float(nll)/max(1,len(choice)));input_counts.append(len(tokens))
    prediction=max(range(len(scores)),key=lambda i:(scores[i],-i))
    return {'id':row['id'],'prediction':prediction,'correct':int(prediction==row['gold']),'scores':scores,'input_tokens':sum(input_counts),'actual_forward_calls':len(scores),'scoring_rule':'source-mapped task metric: PIQA raw continuation likelihood; ARC-Challenge and HellaSwag character-normalized; first-choice ties; independent implementation, no official-test or full-harness equivalence claimed'}

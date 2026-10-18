"""Native artifact, one active model, fixed paired 5+20 workload evaluation."""
import hashlib,json,time
import torch
from transformers import GenerationConfig
from cost_recorder import measured
def workload_ids(rows,batch,prompt,device):
    source=[]
    for row in rows:
        if len(row['tokens'])>=prompt:source.append(row['tokens'][:prompt])
        if len(source)==batch:break
    if len(source)!=batch:raise ValueError('predeclared serving source sequences insufficient')
    return torch.tensor(source,device=device,dtype=torch.long)
def paired_workload(pool,tokenizer,rows,batch,prompt,label,journal,costs,*,checkpoint=lambda:None):
    results=[]
    config=GenerationConfig(do_sample=False,num_beams=1,min_new_tokens=128,max_new_tokens=128,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id)
    for trial in range(25):
        checkpoint();warm=trial<5
        # Balanced predeclared pair order, independent of observed timing.
        order=['dense','candidate'] if trial%2==0 else ['candidate','dense']
        for name in order:
            identity=label+'-'+name+'-'+('warmup-' if warm else 'timed-')+str(trial)
            with costs.phase(identity+'-activate','swap_warmup','research_evaluation'):
                model=pool.activate(name);ids=workload_ids(rows,batch,prompt,'cuda:0');attention=torch.ones_like(ids)
                # Exactly one prefill rewarm after each physical weight swap,
                # recorded separately; not a hidden additional timed sample.
                with torch.no_grad():probe=model(input_ids=ids,attention_mask=attention,use_cache=False,logits_to_keep=1).logits
                del probe;torch.cuda.synchronize()
            with costs.phase(identity,'swap_warmup' if warm else 'serving','research_evaluation') as metrics:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
                with torch.no_grad():output=model.generate(input_ids=ids,attention_mask=attention,generation_config=config,use_model_defaults=False)
                torch.cuda.synchronize();elapsed=time.monotonic()-begin;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                if output.shape!=(batch,prompt+128):raise ValueError('full forced128 output required')
                row={'id':identity,'model':name,'workload':label,'trial':trial,'warmup':warm,'pair_order':order,'batch':batch,'prompt_tokens':prompt,'generated_tokens':128,'wall_seconds':elapsed,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'input_ids_sha256':hashlib.sha256(json.dumps(ids.cpu().tolist(),separators=(',',':')).encode()).hexdigest(),'one_active_native_model':True,'weight_swap_and_rewarm_seconds_excluded_from_serving_but_charged_separately':True}
                metrics['forward_calls']=measured(128,'forced autoregressive steps','calls');metrics['output_tokens']=measured(batch*128,'actual generation shape','tokens');metrics['input_tokens']=measured(batch*(prompt+127),'prefill and subsequent decoding input tokens','tokens');journal.complete_work(identity,row);results.append(row);del output
            del ids,attention
    if len(results)!=50 or sum(not r['warmup'] for r in results)!=40:raise ValueError('complete5 paired warmups and20 paired timed trials')
    return results

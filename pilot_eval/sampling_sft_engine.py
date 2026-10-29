"""Optional real engines for bounded qualification; imports stay behind job gates.

Versioned vLLM adapter uses public token-in/token-out, prompt_logprobs and
collective_rpc/apply_model APIs. No automatic installs, downloads or fallback.
"""
import importlib.metadata
import json
import os
import subprocess

from pilot_eval.sampling_sft_sampler import ProtocolError, proposer
from pilot_eval.workflow import _hash


def numeric_flags():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision())


def worker_evidence(model):
    """Executed inside the actual inference worker, not just the parent process."""
    import torch
    flags=numeric_flags()
    dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    devices=sorted({str(p.device) for p in model.parameters()})
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip().splitlines()
    if len(driver)!=1:raise ValueError('Require one visible physical GPU in the bounded runtime.')
    return dict(gpu=torch.cuda.get_device_name(),driver=driver[0],cuda=torch.version.cuda,
        torch_version=importlib.metadata.version('torch'),worker_numeric_flags=flags,
        parameter_dtypes=dtypes,parameter_devices=devices)


def check_worker(actual,expected):
    for key in ['gpu','driver','cuda','torch_version','worker_numeric_flags']:
        if actual[key]!=expected[key]:raise ValueError('Actual worker runtime mismatch: '+key)
    if actual['parameter_dtypes']!=['torch.float32'] or any(not d.startswith('cuda') for d in actual['parameter_devices']):
        raise ValueError('Worker weights must be unquantized FP32 on CUDA.')


def reset_memory(model):
    import torch
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()


def peak_memory(model):
    import torch
    torch.cuda.synchronize()
    return dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())


class VLLMEngine:
    def __init__(self,llm,settings):
        self.llm=llm;self.settings=settings

    def params(self,**kwargs):
        from vllm import SamplingParams
        r=self.settings
        return SamplingParams(n=1,top_k=r['top_k'],min_p=r['min_p'],
            presence_penalty=r['presence_penalty'],frequency_penalty=r['frequency_penalty'],
            detokenize=False,skip_special_tokens=False,ignore_eos=False,
            stop_token_ids=r['actual_stop_ids'],**kwargs)

    def generate(self,requests):
        if any(len(r['prompt_ids'])+r['max_tokens']>self.settings['max_model_len'] for r in requests):
            raise ProtocolError('Proposer context overflow; truncation forbidden.')
        prompts=[dict(prompt_token_ids=r['prompt_ids']) for r in requests]
        params=[self.params(temperature=0.6,top_p=1.0,repetition_penalty=1.1,
            max_tokens=r['max_tokens'],min_tokens=0,seed=r['seed']) for r in requests]
        outputs=self.llm.generate(prompts,params,use_tqdm=False)
        result=[]
        for request,out in zip(requests,outputs):
            if list(out.prompt_token_ids)!=request['prompt_ids'] or len(out.outputs)!=1:
                raise ProtocolError('vLLM prompt/order/output cardinality mismatch.')
            value=out.outputs[0]
            result.append(dict(token_ids=list(value.token_ids),finish_reason=value.finish_reason,
                               stop_reason=value.stop_reason,num_cached_tokens=out.num_cached_tokens))
        if len(result)!=len(requests):raise ProtocolError('vLLM generation cardinality mismatch.')
        return result

    def score(self,requests):
        from vllm import SamplingParams
        if any(len(r['prompt_ids'])+len(r['response_ids'])+1>self.settings['max_model_len'] for r in requests):
            raise ProtocolError('Scorer context overflow; truncation forbidden.')
        # Raw model probabilities: no proposer penalties/temperature applied.
        params=SamplingParams(n=1,max_tokens=1,temperature=1.0,top_p=1.0,top_k=-1,
            min_p=0.0,repetition_penalty=1.0,presence_penalty=0.0,frequency_penalty=0.0,
            prompt_logprobs=0,seed=0,detokenize=False)
        prompts=[dict(prompt_token_ids=r['prompt_ids']+r['response_ids']) for r in requests]
        outputs=self.llm.generate(prompts,params,use_tqdm=False)
        results=[]
        for request,out in zip(requests,outputs):
            ids=request['prompt_ids']+request['response_ids']
            if list(out.prompt_token_ids)!=ids or out.prompt_logprobs is None or len(out.prompt_logprobs)!=len(ids):
                raise ProtocolError('Missing/misaligned vLLM prompt_logprobs.')
            positions=list(range(len(request['prompt_ids']),len(ids)))
            try:values=[float(out.prompt_logprobs[i][ids[i]].logprob) for i in positions]
            except (KeyError,TypeError,AttributeError) as exc:
                raise ProtocolError('Missing target token in vLLM prompt_logprobs.') from exc
            results.append(dict(token_ids=request['response_ids'],positions=positions,logprobs=values,
                                num_cached_tokens=out.num_cached_tokens))
        if len(results)!=len(requests):raise ProtocolError('vLLM scoring cardinality mismatch.')
        return results

    def clear_cache(self):self.llm.reset_prefix_cache()

    def measurement_start(self):self.llm.apply_model(reset_memory)
    def measurement_end(self):return self.llm.apply_model(peak_memory)


class HFOracle:
    """Independent full-vocabulary FP32 log-softmax with explicit causal shift."""
    def __init__(self,model):self.model=model

    def score(self,requests):
        import torch
        results=[]
        with torch.inference_mode():
            for request in requests:
                ids=request['prompt_ids']+request['response_ids'];boundary=len(request['prompt_ids'])
                tokens=torch.tensor([ids],dtype=torch.long,device='cuda')
                # Avoid retaining a full sequence × vocab tensor: obtain residuals,
                # then apply the independent HF head one target position at a time.
                hidden=self.model.model(input_ids=tokens,use_cache=False).last_hidden_state
                values=[]
                for position in range(boundary,len(ids)):
                    logits=self.model.lm_head(hidden[:,position-1,:]).float()
                    values.append(float(torch.log_softmax(logits,dim=-1)[0,ids[position]].cpu()))
                results.append(dict(token_ids=request['response_ids'],positions=list(range(boundary,len(ids))),logprobs=values))
        return results


def open_engine(plan,root):
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from pilot_eval.sampling_sft import verify
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Bounded engine requires one visible CUDA GPU.')
    settings=plan['settings'];r=settings['runtime']
    if importlib.metadata.version('transformers')!='4.57.6':
        raise ValueError('Independent tokenizer/HF adapter requires transformers4.57.6.')
    audit=verify(root/plan['source_plan'],root)['audit'];identity=audit['training']['identity']
    model_path=root/settings['local_model'];tokenizer_path=root/settings['local_tokenizer']
    model_config=json.loads((model_path/'config.json').read_text())
    if model_config.get('quantization_config'):
        raise ValueError('Quantized snapshots cannot qualify this FP32 instrument.')
    # Model snapshots are staged offline under their immutable Hub revision name.
    if model_path.name!=identity['model_revision'] or tokenizer_path.name!=identity['tokenizer_revision']:
        raise ValueError('Local snapshot directory names must identify the audited immutable Hub revisions.')
    numeric_flags()
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
    if _hash(tokenizer.chat_template)!=audit['training']['prompt_mask_contract']['chat_template_sha256']:
        raise ValueError('Pinned tokenizer chat template changed.')
    actual=[tokenizer.convert_tokens_to_ids(s) for s in ['<|im_end|>','<|endoftext|>']]
    if actual!=r['actual_stop_ids'] or tokenizer.eos_token_id not in actual:
        raise ValueError('Declared actual stop IDs differ from pinned tokenizer.')
    maximum=max(len(proposer(p,[],tokenizer)) for p in plan['problems'])
    if r['max_model_len']<maximum+2048:
        raise ValueError('Context cannot accommodate the declared duplicated-prefix bound.')
    if settings['mode']=='oracle':
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
        evidence=worker_evidence(model);check_worker(evidence,r)
        engine=HFOracle(model)
    else:
        if importlib.metadata.version('vllm')!=r['vllm_version']:
            raise ValueError('Installed vLLM differs from declared supported version.')
        from vllm import LLM
        llm=LLM(model=str(model_path),tokenizer=str(tokenizer_path),dtype='float32',
            trust_remote_code=False,tensor_parallel_size=1,quantization=None,enforce_eager=True,
            seed=0,max_model_len=r['max_model_len'],gpu_memory_utilization=r['gpu_memory_utilization'],
            enable_prefix_caching=r['prefix_caching'],max_num_seqs=r['max_num_seqs'],
            max_num_batched_tokens=r['max_num_batched_tokens'],generation_config='vllm')
        evidence=llm.apply_model(worker_evidence)
        if len(evidence)!=1:raise ValueError('One actual inference worker required.')
        evidence=evidence[0];check_worker(evidence,r)
        engine=VLLMEngine(llm,r)
    return engine,tokenizer,dict(execution_kind='real-cuda',worker=evidence,
        transformers_version=importlib.metadata.version('transformers'),runtime_hash=plan['runtime_hash'],
        model_files='frozen local snapshot inventory; upstream download verification is preflight evidence')

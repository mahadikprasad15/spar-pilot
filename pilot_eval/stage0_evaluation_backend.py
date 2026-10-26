"""One lazy local-only FP32 model with explicitly selected PEFT variants."""
from pathlib import Path
import os

from pilot_eval.stage0_qualification_backend import QualificationDependencies
from pilot_eval.stage0_sources import inside
from pilot_eval.workflow import _hash


class EvaluationDependencies(QualificationDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':
            raise ValueError('A CPU fixture cannot launch real model inference.')
        return EvaluationEngine(config,root)


def verify_loaded_adapter(model,variant,directory):
    """Check every actual PEFT factor against its saved FP32 tensor payload."""
    import numpy as np
    from peft import get_peft_model_state_dict
    from safetensors.numpy import load_file
    expected=load_file(str(directory/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(model,adapter_name=variant)
    if set(actual)!=set(expected):
        raise ValueError(f'Actual adapter mapping differs from saved payload: {variant}')
    for name,value in actual.items():
        array=value.detach().cpu().numpy()
        if array.dtype!=np.float32 or not np.array_equal(array,expected[name]):
            raise ValueError(f'Actual loaded adapter factor differs from saved payload: {variant}; {name}')
    return dict(variant=variant,tensor_count=len(expected),exact_payload_match=True)


class EvaluationEngine:
    def __init__(self,config,root):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,enable_full_determinism
        from peft import PeftModel
        from pilot_eval.sft_backend import frozen_weight_hash
        self.torch=torch;self.config=config;self.root=Path(root).resolve()
        enable_full_determinism(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        snapshot=inside(self.root,config['weights_dir'])
        self.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        self.tokenizer.padding_side='left'
        decoding=config['decoding']
        if (self.tokenizer.eos_token_id!=decoding['eos_token_id']
                or self.tokenizer.pad_token_id!=decoding['pad_token_id']):
            raise ValueError('Pinned tokenizer EOS/PAD identity differs from Pilot 4.')
        template=config['prompt_contract'].get('chat_template_sha256')
        if template and _hash(self.tokenizer.chat_template)!=template:
            raise ValueError('Pinned tokenizer chat template differs from the source contract.')
        base=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
            dtype=torch.float32,attn_implementation='eager',device_map={'':0})
        order=config['variant_order'];first=order[0]
        self.model=PeftModel.from_pretrained(base,inside(self.root,config['variants'][first]['adapter_path']),
            adapter_name=first,is_trainable=False)
        for variant in order[1:]:
            self.model.load_adapter(inside(self.root,config['variants'][variant]['adapter_path']),
                adapter_name=variant,is_trainable=False)
        self.model.eval();self.model.config.use_cache=True
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError('All base and adapter weights must be FP32.')
        self.loaded_adapter_checks=[verify_loaded_adapter(self.model,v,inside(self.root,config['variants'][v]['adapter_path']))
                                    for v in order]
        self.initial_base=frozen_weight_hash(self.model)
        if self.initial_base!=config['model_identity']['base_sha256']:
            raise ValueError('Loaded model base hash differs from the audited source.')
        self.guard=self._versions();self.adapter_guard=self._versions(adapters=True);self.inputs={}
        print(f'Stage 0 greedy: one FP32 model loaded; {len(order)} verified adapters attached.',flush=True)

    def _versions(self,*,adapters=False):
        return {name:(id(p),p.data_ptr(),tuple(p.shape),p.dtype,str(p.device),p._version)
                for name,p in self.model.named_parameters() if ('lora_' in name)==adapters}

    def check_integrity(self,*,full=False):
        same=self._versions()==self.guard
        adapters=self._versions(adapters=True)==self.adapter_guard
        if full:
            from pilot_eval.sft_backend import frozen_weight_hash
            same=same and frozen_weight_hash(self.model)==self.initial_base
            try:
                for v in self.config['variant_order']:
                    verify_loaded_adapter(self.model,v,inside(self.root,self.config['variants'][v]['adapter_path']))
            except ValueError:
                adapters=False
        return dict(base_unchanged=same,adapters_unchanged=adapters,
                    kind='full-base-sha256-and-adapter-payloads' if full else 'parameter-identity-version-guard')

    def prepare_inputs(self,rows):
        result=[]
        for row in rows:
            tokens=self.tokenizer.encode(row['prompt'],add_special_tokens=False)
            if not tokens or len(tokens)+self.config['decoding']['max_new_tokens']>self.model.config.max_position_embeddings:
                raise ValueError('Frozen prompt is empty or would exceed the model context.')
            if 'prompt_tokens' in row and len(tokens)!=row['prompt_tokens']:
                raise ValueError('Original Pilot 4 prompt token count changed.')
            self.inputs[row['id']]=tokens
            result.append(dict(id=row['id'],input_ids=tokens,attention_mask=[1]*len(tokens)))
        return result

    def generate(self,rows,variant,seed):
        from transformers import set_seed
        if variant not in self.config['variants']:
            raise ValueError('Adapter is outside the frozen intervention catalogue.')
        self.model.set_adapter(variant)
        if self.model.active_adapters!=[variant]:
            raise ValueError('Model did not select exactly the requested adapter.')
        self.model.eval();set_seed(seed)
        sequences=[self.inputs[r['id']] for r in rows];width=max(map(len,sequences))
        policy=self.config['decoding'];pad=policy['pad_token_id'];eos=policy['eos_token_id']
        ids=self.torch.tensor([[pad]*(width-len(s))+s for s in sequences],device='cuda:0',dtype=self.torch.long)
        mask=self.torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences],device='cuda:0',dtype=self.torch.long)
        with self.torch.inference_mode():
            generated=self.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,
                num_beams=1,num_return_sequences=1,repetition_penalty=1.0,no_repeat_ngram_size=0,
                max_new_tokens=policy['max_new_tokens'],pad_token_id=pad,eos_token_id=eos,
                use_cache=policy['use_cache'])
        self.torch.cuda.synchronize()
        result=[]
        for continuation in generated[:,width:].tolist():
            end=next((i for i,t in enumerate(continuation) if t==eos),None)
            content=continuation if end is None else continuation[:end]
            raw=continuation if end is None else continuation[:end+1]
            result.append(dict(text=self.tokenizer.decode(content,skip_special_tokens=True),
                token_ids=content,generated_token_ids=raw,stop_reason='cap' if end is None else 'eos'))
        return result

    def close(self):
        import gc
        self.model=None;self.tokenizer=None;gc.collect();self.torch.cuda.empty_cache()

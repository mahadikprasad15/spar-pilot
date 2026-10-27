"""Local-only untuned HF FP32 sampler; historical eight-draw method is reused."""
import hashlib
from types import SimpleNamespace

from pilot_eval.activation_prepare import PROJECTIONS
from pilot_eval.grpo_baseline import HFSampler
from pilot_eval.stage0_qualification_backend import QualificationDependencies
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_weights import LocalWeights


def parameter_name(name):
    parts=name.split('.')
    if len(parts)>=5 and parts[-2] in PROJECTIONS and parts[-1] in ['weight','bias']:
        name='.'.join(parts[:-1]+['base_layer',parts[-1]])
    return 'base_model.model.'+name


class RolloutDependencies(QualificationDependencies,LocalWeights):
    def engine(self,config,root):
        if config['execution_kind']=='synthetic-cpu-fixture':raise ValueError('Synthetic inputs cannot launch real sampling.')
        return RolloutEngine(config,root)


class RolloutEngine(HFSampler):
    def __init__(self,config,root):
        import os
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,enable_full_determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';enable_full_determinism(config['generation_settings']['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.torch=torch;self.config=config;snapshot=inside(root,config['weights_dir'])
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True);tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        if type(tokenizer.eos_token_id) is not int or type(tokenizer.pad_token_id) is not int:
            raise ValueError('The pinned tokenizer must have scalar EOS/PAD IDs.')
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',device_map={'':0})
        if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Sampling base weights must be FP32.')
        model.eval();model.generation_config=GenerationConfig(do_sample=False,num_beams=1,num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,repetition_penalty=1.,no_repeat_ngram_size=0)
        self.backend=SimpleNamespace(model=model,tokenizer=tokenizer)
        self.guard=self._versions()
        if self._hash()!=config['model_identity']['base_sha256']:raise ValueError('Actual untuned base hash differs from the audited source.')
        print('Stage 0 R0: one verified local FP32 untuned model; eight draws per question.',flush=True)

    def _versions(self):
        return {n:(id(p),p.data_ptr(),p._version,tuple(p.shape),p.dtype) for n,p in self.backend.model.named_parameters()}

    def _hash(self):
        result=hashlib.sha256()
        for name,p in sorted((parameter_name(n),p) for n,p in self.backend.model.named_parameters()):
            result.update(name.encode()+b'\0');result.update(str(tuple(p.shape)).encode())
            result.update(p.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes())
        return result.hexdigest()

    def check_integrity(self,*,full=False):
        same=self._versions()==self.guard
        if full:same=same and self._hash()==self.config['model_identity']['base_sha256']
        return dict(base_unchanged=same,kind='base SHA256' if full else 'parameter identity/version guards')

    def special_tokens(self):
        return dict(eos_token_id=self.backend.tokenizer.eos_token_id,pad_token_id=self.backend.tokenizer.pad_token_id)

    def prepare_inputs(self,rows):
        result=[]
        for row in rows:
            ids=self.backend.tokenizer.encode(row['prompt'],add_special_tokens=False)
            if ids!=row['input_ids'][:row['prompt_tokens']]:raise ValueError('Sampling prompt IDs differ from the saved SFT inputs.')
            if len(ids)+1024>self.backend.model.config.max_position_embeddings:raise ValueError('Sampling context would overflow.')
            result.append(dict(id=row['id'],input_ids=ids))
        return result

    def sample(self,rows,settings,seed):
        if len(rows)!=1:raise ValueError('Historical generation requires one question per eight-draw group.')
        with self.torch.autocast('cuda',enabled=False):return super().sample(rows,settings,seed)

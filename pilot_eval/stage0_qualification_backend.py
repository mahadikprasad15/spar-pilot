"""Real, lazy CUDA backend for Ticket 04's bounded mixed-loader diagnostic."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


@contextmanager
def deadline_guard(seconds, enabled):
    """Bound a real notebook child even if a CUDA call stops making progress."""
    if not enabled or not hasattr(signal, 'SIGALRM'):
        yield
        return
    def expired(_signum, _frame):
        raise TimeoutError('Reserved qualification wall time expired; reconcile usage before resume.')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


class QualificationDependencies:
    """Runtime and one-model engine factory; importing this module needs no Torch."""
    def runtime(self):
        import torch
        from pilot_eval.sft_backend import PINS
        versions={name:importlib.metadata.version(name) for name in PINS}
        mismatches={name:(versions[name],expected) for name,expected in PINS.items()
                    if versions[name].split('+')[0]!=expected}
        if mismatches:
            raise ValueError(f'Qualification dependency versions differ from the validated pins: {mismatches}')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Real Stage 0 qualification requires exactly one visible CUDA GPU.')
        try:
            driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                text=True,stderr=subprocess.STDOUT).strip().splitlines()[0]
            commit=subprocess.check_output(['git','rev-parse','HEAD'],
                cwd=Path(__file__).resolve().parent.parent,text=True).strip()
        except (OSError,subprocess.CalledProcessError,IndexError) as exc:
            raise ValueError('Could not record CUDA driver and checked-out Git commit.') from exc
        return dict(execution_kind='real-cuda',session_id=_session_id(),python=sys.version,
            versions=versions,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
            driver=driver,git_commit=commit,memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            compute_capability=list(torch.cuda.get_device_capability(0)),precision='float32',
            autocast=False,tf32=False,attention='eager',deterministic=True,
            cublas_workspace_config=':4096:8')

    def preflight(self,config,root):
        """Prove artifact-root readback before reserving any inference work."""
        directory=Path(root)/config['run_path']/'preflight'
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'durable-readback.bin'
        payload=os.urandom(4096)
        with path.open('wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        if path.read_bytes()!=payload:
            raise OSError('Artifact-root write/readback failed; no model was loaded.')
        path.unlink()
        return dict(readback_bytes=len(payload),artifact_root=str(root),passed=True)

    def engine(self,config,root):
        return QualificationEngine(config,root)


def _session_id():
    boot=Path('/proc/sys/kernel/random/boot_id')
    if not boot.is_file():
        raise ValueError('Cannot identify a stable Colab runtime session; refuse GPU reservation.')
    return platform.node()+':'+boot.read_text().strip()


class QualificationEngine:
    def __init__(self,config,root):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,enable_full_determinism
        from peft import PeftModel
        from pilot_eval.sft_backend import frozen_weight_hash
        from pilot_eval.stage0_sources import inside
        self.torch=torch;self.config=config;self.root=Path(root).resolve();self.tokenizer=None
        enable_full_determinism(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_grad_enabled(False)
        model_info=config['model_identity']
        weights=inside(self.root,config['weights_dir'])
        self.tokenizer=AutoTokenizer.from_pretrained(weights,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(weights,dtype=torch.float32,
            attn_implementation='eager',device_map={'':0},local_files_only=True)
        if any(parameter.dtype!=torch.float32 for parameter in base.parameters()):
            raise ValueError('Qualification base model did not load in FP32.')
        first=config['variants']['V1']['adapter_path']
        second=config['variants']['V5']['adapter_path']
        self.model=PeftModel.from_pretrained(base,inside(self.root,first),adapter_name='sft',is_trainable=False)
        self.model.load_adapter(inside(self.root,second),adapter_name='grpo',is_trainable=False)
        self.model.eval();self.model.config.use_cache=False
        self.modules=config['merged_modules'];self.inputs={}
        actual=frozen_weight_hash(self.model)
        if actual!=model_info['base_sha256']:
            raise ValueError('Loaded model base hash differs from the source audit.')
        if len(self.model.get_base_model().model.layers)!=28:
            raise ValueError('Loaded model layer count differs from the audited Qwen architecture.')
        if self.tokenizer.eos_token_id is None:
            raise ValueError('Pinned tokenizer has no EOS token for teacher-forced answer positions.')
        if (self.tokenizer.eos_token_id!=config['decoding']['eos_token_id']
                or self.tokenizer.pad_token_id!=config['decoding']['pad_token_id']):
            raise ValueError('Loaded tokenizer special-token IDs differ from the frozen Pilot 4 evaluation config.')
        print('Stage 0 qualification model loaded once; SFT and GRPO edited adapters attached.',flush=True)

    def prepare_inputs(self,rows):
        manifest=[]
        for row in rows:
            prompt=self.tokenizer.encode(row['prompt'],add_special_tokens=False)
            answer=self.tokenizer.encode(row['gold'],add_special_tokens=False)+[self.tokenizer.eos_token_id]
            if not prompt or not answer:
                raise ValueError(f'Empty tokenized prompt/answer: {row["id"]}')
            full=prompt+answer
            if len(full)>int(self.model.config.max_position_embeddings):
                raise ValueError(f'Qualification sequence exceeds model context: {row["id"]}')
            prepared=dict(id=row['id'],input_ids=full,attention_mask=[1]*len(full),
                          prompt_tokens=len(prompt),answer_positions=list(range(len(prompt)-1,len(full)-1)))
            self.inputs[row['id']]=prepared
            manifest.append(dict(id=row['id'],input_ids=full,attention_mask=prepared['attention_mask'],
                prompt_tokens=len(prompt),answer_positions=prepared['answer_positions'],
                token_sha256=hashlib.sha256(json.dumps([full,prepared['attention_mask']],separators=(',',':')).encode()).hexdigest()))
        return dict(scope='16 frozen held-out prompts; exact token IDs and masks',sequences=manifest)

    @contextmanager
    def _mixed(self,arm,negative=False):
        self.model.set_adapter(arm)
        saved=[]
        try:
            with self.torch.no_grad():
                for name in self.modules:
                    layer=self.model.get_submodule(name)
                    if not hasattr(layer,'base_layer') or arm not in layer.lora_A:
                        raise ValueError(f'Merged loader target is not the expected LoRA layer: {name}')
                    weight=layer.base_layer.weight
                    original=weight.detach().clone()
                    a=layer.lora_A[arm].weight; b=layer.lora_B[arm].weight
                    scale=float(layer.scaling[arm])
                    delta=(b@a)*scale
                    weight.copy_(original+delta.to(dtype=weight.dtype))
                    old_disabled=layer._disable_adapters
                    if not negative:
                        layer._disable_adapters=True
                    saved.append((layer,weight,original,old_disabled))
            yield
        finally:
            with self.torch.no_grad():
                for layer,weight,original,old_disabled in reversed(saved):
                    weight.copy_(original);layer._disable_adapters=old_disabled
                    if not self.torch.equal(weight,original):
                        raise RuntimeError('Merged-layer weight restoration was not exact.')

    def _teacher(self,ids,arm,mode):
        self.model.set_adapter(arm)
        data=self.inputs[ids['id']]
        tokens=self.torch.tensor([data['input_ids']],device='cuda:0',dtype=self.torch.long)
        mask=self.torch.ones_like(tokens)
        context=self._mixed(arm,negative=(mode=='negative')) if mode in ['merged','negative'] else _null()
        with context,self.torch.inference_mode():
            logits=self.model(input_ids=tokens,attention_mask=mask,use_cache=False).logits[0]
        positions=data['answer_positions']
        return logits[positions].detach().cpu().double().numpy()

    def _generate(self,row,arm,mode):
        self.model.set_adapter(arm)
        data=self.inputs[row['id']];prompt=data['input_ids'][:data['prompt_tokens']]
        ids=self.torch.tensor([prompt],device='cuda:0',dtype=self.torch.long)
        mask=self.torch.ones_like(ids)
        context=self._mixed(arm) if mode=='merged' else _null()
        with context,self.torch.inference_mode():
            result=self.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,num_beams=1,
                max_new_tokens=self.config['decoding']['max_new_tokens'],
                pad_token_id=self.config['decoding']['pad_token_id'],
                eos_token_id=self.config['decoding']['eos_token_id'])
        return result[0,len(prompt):].detach().cpu().tolist()

    def _next_logits(self,row,arm,mode,prefix):
        self.model.set_adapter(arm)
        tokens=self.torch.tensor([prefix],device='cuda:0',dtype=self.torch.long)
        context=self._mixed(arm) if mode=='merged' else _null()
        with context,self.torch.inference_mode():
            return self.model(input_ids=tokens,attention_mask=self.torch.ones_like(tokens),use_cache=False).logits[0,-1].detach().cpu().double().numpy()

    def diagnose(self,row,arm,check_deadline):
        from pilot_eval.stage0_qualification_math import divergence
        ref=self._teacher(row,arm,'reference');merged=self._teacher(row,arm,'merged');negative=self._teacher(row,arm,'negative')
        check_deadline();near=False;divergence_evidence=None
        greedy_started=__import__('time').monotonic()
        reference_tokens=self._generate(row,arm,'reference');merged_tokens=self._generate(row,arm,'merged')
        greedy_seconds=__import__('time').monotonic()-greedy_started
        exact=reference_tokens==merged_tokens
        if not exact:
            index=next((i for i in range(min(len(reference_tokens),len(merged_tokens)))
                        if reference_tokens[i]!=merged_tokens[i]),min(len(reference_tokens),len(merged_tokens)))
            prompt=self.inputs[row['id']]['input_ids'][:self.inputs[row['id']]['prompt_tokens']]
            prefix=prompt+reference_tokens[:index]
            left=self._next_logits(row,arm,'reference',prefix);right=self._next_logits(row,arm,'merged',prefix)
            divergence_evidence=divergence(reference_tokens,merged_tokens,left,right,prefix=prefix)
            near=divergence_evidence['near_tie']
        check_deadline()
        self.torch.cuda.synchronize()
        return dict(reference=ref,merged=merged,negative=negative,positions=self.inputs[row['id']]['answer_positions'],
            greedy=dict(exact=exact,near_tie=near,reference_token_ids=reference_tokens,merged_token_ids=merged_tokens,
                reference_text=self.tokenizer.decode(reference_tokens,skip_special_tokens=False),
                merged_text=self.tokenizer.decode(merged_tokens,skip_special_tokens=False),divergence=divergence_evidence),
            integrity=True,generated_tokens=max(len(reference_tokens),len(merged_tokens)),
            greedy_seconds=greedy_seconds)

    def close(self):
        import gc
        self.model=None;self.tokenizer=None;gc.collect();self.torch.cuda.empty_cache()


@contextmanager
def _null():
    yield

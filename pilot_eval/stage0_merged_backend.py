"""Mixed full-matrix representation usable by the existing greedy engine API.

No model is loaded on import. Runtime qualification/approvals belong to Ticket 10.
"""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pilot_eval.stage0_evaluation_backend import EvaluationDependencies, EvaluationEngine
from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash


def runner_variants(catalogue):
    """Retain original references; do not schedule empty sensitivity variants."""
    if catalogue['state']=='not-applicable':
        raise ValueError('Conditional catalogue is not-applicable; no extension model run is scheduled.')
    return {**catalogue['references'],**{name:value for name,value in catalogue['variants'].items()
        if value['status']=='ready-for-qualification'}}


@contextmanager
def merged_scope(model,definition,root,torch,*,adapter_name,negative_double_adapter=False,matrix_cache=None):
    """Swap parameter objects, never mutate W0; restore on every exit path.

    This preserves the original base parameter identity/version guards. Loading a
    saved already-merged matrix does not apply its adapter again. Only the bounded
    unhalved diagnostic may deliberately enable that branch as a negative control.
    """
    if definition.get('status')!='ready-for-qualification':
        raise ValueError('Conditional variant has no eligible executable matrix scope.')
    modules=definition['merged_modules'];mask=definition['adapter_disable_modules']
    if mask!=sorted(modules) or not mask:raise ValueError('Merged module mapping/adapter-disable mask differs.')
    if negative_double_adapter and definition['mode']!='unhalved':
        raise ValueError('Double-adapter negative control is restricted to the unhalved diagnostic.')
    previous=list(model.active_adapters)
    if len(previous)!=1:raise ValueError('Mixed loader requires one explicit previous active adapter.')
    flags=[(layer,layer._disable_adapters) for _,layer in model.named_modules() if hasattr(layer,'_disable_adapters')]
    saved=[];mutated=False
    try:
        model.set_adapter(adapter_name)
        if list(model.active_adapters)!=[adapter_name]:raise ValueError('Mixed loader selected the wrong source adapter.')
        for layer,_ in flags:layer._disable_adapters=False
        for name in mask:
            descriptor=modules[name];path=inside(Path(root).resolve(),descriptor['path'])
            key=(str(path),descriptor['sha256'])
            matrix=None if matrix_cache is None else matrix_cache.get(key)
            if matrix is None:
                if file_hash(path)!=descriptor['sha256']:raise ValueError('Saved merged matrix payload changed.')
                matrix=np.load(path,allow_pickle=False)
                if matrix_cache is not None:
                    matrix.flags.writeable=False;matrix_cache[key]=matrix
            if matrix.dtype!=np.float32 or list(matrix.shape)!=descriptor['shape'] or not np.isfinite(matrix).all():
                raise ValueError('Saved merged matrix precision/shape/finiteness differs.')
            layer=model.get_submodule(name)
            if not hasattr(layer,'base_layer') or bool(getattr(layer,'merged',False)):
                raise ValueError('Merged loader requires an unmerged LoRA linear module.')
            original=layer.base_layer.weight
            if original.dtype!=torch.float32 or tuple(original.shape)!=matrix.shape:
                raise ValueError('Actual base module shape/FP32 dtype differs from the full matrix.')
            parameter=torch.nn.Parameter(torch.from_numpy(matrix.copy()).to(device=original.device,dtype=torch.float32),requires_grad=False)
            saved.append((layer,original,parameter,getattr(parameter,'_version',None)))
            layer.base_layer.weight=parameter
            layer._disable_adapters=not negative_double_adapter
        yield
    finally:
        for layer,original,parameter,version in reversed(saved):
            mutated=mutated or layer.base_layer.weight is not parameter or getattr(parameter,'_version',None)!=version
            layer.base_layer.weight=original
        try:model.set_adapter(previous[0])
        finally:
            for layer,disabled in flags:layer._disable_adapters=disabled
        if mutated:raise RuntimeError('A frozen mixed matrix was mutated during use; original source objects restored.')


class MergedEvaluationDependencies(EvaluationDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic catalogues cannot launch real model inference.')
        return MergedEvaluationEngine(config,root)


class MergedEvaluationEngine(EvaluationEngine):
    """The normal loader/inputs/integrity contract with scoped mixed generation."""
    def __init__(self,config,root):
        from pilot_eval.stage0_intruder_edits import verify
        catalogue=verify(config['mixed_catalogue_path'],root)
        expected=runner_variants(catalogue)
        if config.get('collection_kind')=='conditional-intruders' and 'controls' in config:
            from pilot_eval.stage0_intruder_edits import _load
            from pilot_eval.stage0_edits import _load as load_edits
            self_root=Path(root).resolve()
            frozen,_,_,_=_load(config['mixed_catalogue_path'],self_root)
            _,_,audit,_=load_edits(self_root/frozen['edits_path'],self_root)
            for arm in ['sft','grpo']:
                source=audit['sources'][arm]['checkpoints']['0']
                if not source['b_exact_zero']:raise ValueError('Mixed drift zero control is not zero.')
                expected[arm+'-zero']=dict(arm=arm,operation='zero-control',adapter_path=source['path'])

        if (config['model_identity']!=catalogue['model_identity'] or config['weights_dir']!=catalogue['weights_dir']
                or any(name not in expected or value!=expected[name] for name,value in config['variants'].items())
                or set(config['variant_order'])!=set(config['variants'])
                or len(config['variant_order'])!=len(config['variants'])):
            raise ValueError('Runner variants/model identity differ from the verified mixed catalogue.')
        if catalogue['execution_kind']=='synthetic-cpu-fixture':raise ValueError('Synthetic matrix construction does not qualify a real GPU runner.')
        self._matrix_cache={};self._cached_variant=None
        super().__init__(config,root)

    def matrix_cache(self,variant):
        if variant!=self._cached_variant:
            self._matrix_cache.clear();self._cached_variant=variant
        return self._matrix_cache

    def check_integrity(self,*,full=False):
        proof=super().check_integrity(full=full)
        if full:
            from pilot_eval.stage0_intruder_edits import verify
            try:verify(self.config['mixed_catalogue_path'],self.root)
            except (ValueError,OSError):proof['base_unchanged']=False;proof['mixed_sources_unchanged']=False
            else:proof['mixed_sources_unchanged']=True
        return proof

    def close(self):
        self._matrix_cache.clear()
        super().close()

    def generate(self,rows,variant,seed):
        if variant not in self.config['variants']:raise ValueError('Variant is outside the frozen conditional runner.')
        definition=self.config['variants'][variant]
        if definition.get('representation')!='mixed-full-matrices':
            return super().generate(rows,variant,seed)
        from transformers import set_seed
        with merged_scope(self.model,definition,self.root,self.torch,adapter_name=variant,matrix_cache=self.matrix_cache(variant)):
            self.model.eval();set_seed(seed)
            sequences=[self.inputs[r['id']] for r in rows];width=max(map(len,sequences))
            policy=self.config['decoding'];pad=policy['pad_token_id'];eos=policy['eos_token_id']
            ids=self.torch.tensor([[pad]*(width-len(s))+s for s in sequences],device='cuda:0',dtype=self.torch.long)
            mask=self.torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences],device='cuda:0',dtype=self.torch.long)
            with self.torch.inference_mode():
                generated=self.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,
                    num_beams=1,num_return_sequences=1,repetition_penalty=1.0,no_repeat_ngram_size=0,
                    max_new_tokens=policy['max_new_tokens'],pad_token_id=pad,eos_token_id=eos,use_cache=policy['use_cache'])
            self.torch.cuda.synchronize();result=[]
            for continuation in generated[:,width:].tolist():
                end=next((i for i,t in enumerate(continuation) if t==eos),None)
                content=continuation if end is None else continuation[:end]
                raw=continuation if end is None else continuation[:end+1]
                result.append(dict(text=self.tokenizer.decode(content,skip_special_tokens=True),token_ids=content,
                    generated_token_ids=raw,stop_reason='cap' if end is None else 'eos'))
            return result

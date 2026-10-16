"""One optional CUDA model; arbitrary frozen rank-one checkpoint inventory."""
from pilot_eval.activation_workflow import HFActivationDependencies
from pilot_eval.trace_profile_backend import ProfileEngine


class ProductionDependencies(HFActivationDependencies):
    def runtime(self):return dict(super().runtime(),execution_kind='real-cuda')
    def engine(self,config,root):return ProductionEngine(config,root)


class ProductionEngine(ProfileEngine):
    def __init__(self,config,root):
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from transformers import AutoModelForCausalLM,enable_full_determinism
        from peft import PeftModel
        from pilot_eval.trace_capture import HookedTraceCapture
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic contract cannot launch a real model.')
        enable_full_determinism(config['seed']);self.torch=torch;self.conditions=config['conditions']
        base=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['model_revision'],dtype=torch.float32,
            attn_implementation='eager',device_map={'':0})
        adapters=list(config['adapters']);first=adapters[0]
        self.model=PeftModel.from_pretrained(base,root/config['adapters'][first],adapter_name=first,is_trainable=False)
        for name in adapters[1:]:self.model.load_adapter(root/config['adapters'][name],adapter_name=name,is_trainable=False)
        self.model.eval();self.model.config.use_cache=False
        body=self.model.get_base_model().model
        if len(body.layers)!=28 or self.model.config.hidden_size!=config['width']:raise ValueError('Production model site/width mismatch.')
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):raise ValueError('Production requires FP32 weights.')
        self.hooks=HookedTraceCapture(self.model.get_input_embeddings(),list(body.layers))
        torch.cuda.reset_peak_memory_stats()

    def capture(self,rows,condition):
        from contextlib import nullcontext
        if condition not in self.conditions:raise ValueError('Condition not in frozen production inventory.')
        if condition!='base':self.model.set_adapter(condition)
        inputs={k:self.torch.tensor([r[k] for r in rows],dtype=self.torch.long,device='cuda:0') for k in ['input_ids','attention_mask','position_ids']}
        with (self.model.disable_adapter() if condition=='base' else nullcontext()),self.torch.inference_mode():
            return self.hooks.forward(lambda:self.model(**inputs,use_cache=False,output_hidden_states=False))

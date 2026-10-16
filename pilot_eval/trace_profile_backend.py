"""Optional one-model CUDA boundary for the approved endpoint cost profile."""
from pilot_eval.activation_workflow import HFActivationDependencies


class ProfileDependencies(HFActivationDependencies):
    def runtime(self):
        return dict(super().runtime(), execution_kind='real-cuda', profile_batches=[1,4])

    def engine(self, config, root):
        return ProfileEngine(config,root)


class ProfileEngine:
    def __init__(self,config,root):
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from transformers import AutoModelForCausalLM,enable_full_determinism
        from peft import PeftModel
        from pilot_eval.trace_capture import HookedTraceCapture
        enable_full_determinism(config['seed']);self.torch=torch
        base=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['model_revision'],
            dtype=torch.float32,attn_implementation='eager',device_map={'':0})
        self.model=PeftModel.from_pretrained(base,root/config['adapters']['sft-64'],adapter_name='sft-64',is_trainable=False)
        for arm in ['grpo-64','random-64']:
            self.model.load_adapter(root/config['adapters'][arm],adapter_name=arm,is_trainable=False)
        self.model.eval();self.model.config.use_cache=False
        body=self.model.get_base_model().model
        if len(body.layers)!=28 or self.model.config.hidden_size!=config['width']:
            raise ValueError('Profile model site/width mismatch.')
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError('Profile requires FP32 weights.')
        self.hooks=HookedTraceCapture(self.model.get_input_embeddings(),list(body.layers))
        torch.cuda.reset_peak_memory_stats()
        print('Profile model loaded once; three existing endpoint adapters attached.',flush=True)

    def integrity(self):
        from pilot_eval.sft_backend import frozen_weight_hash
        return frozen_weight_hash(self.model)

    def capture(self,rows,condition):
        from contextlib import nullcontext
        if condition not in ['base','sft-64','grpo-64','random-64']:
            raise ValueError('Unsupported profile condition.')
        if condition!='base':self.model.set_adapter(condition)
        context=self.model.disable_adapter() if condition=='base' else nullcontext()
        inputs={k:self.torch.tensor([r[k] for r in rows],dtype=self.torch.long,device='cuda:0')
                for k in ['input_ids','attention_mask','position_ids']}
        with context,self.torch.inference_mode():
            values=self.hooks.forward(lambda:self.model(**inputs,use_cache=False,output_hidden_states=False))
            self.torch.cuda.synchronize()
        return values

    def memory(self):
        return dict(peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),scope='cumulative since model load')

    def close(self):
        import gc
        self.model=None;self.hooks=None;gc.collect();self.torch.cuda.empty_cache()

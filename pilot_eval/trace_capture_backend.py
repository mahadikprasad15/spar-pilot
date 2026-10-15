"""Optional real-model boundary for the four-document zero diagnostic."""
from pilot_eval.activation_workflow import HFActivationDependencies


class CaptureDependencies(HFActivationDependencies):
    def runtime(self):
        return dict(super().runtime(),execution_kind='real-cuda',diagnostic_batch_size=1)

    def engine(self,config,root):
        return CaptureEngine(config,root)


class CaptureEngine:
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
        self.model=PeftModel.from_pretrained(base,root/config['adapters']['sft'],is_trainable=False)
        self.model.load_adapter(root/config['adapters']['grpo'],adapter_name='grpo-zero',is_trainable=False)
        self.model.eval();self.model.config.use_cache=False
        body=self.model.get_base_model().model
        if len(body.layers)!=28:raise ValueError('Expected exactly 28 decoder blocks.')
        self.capture_hooks=HookedTraceCapture(self.model.get_input_embeddings(),list(body.layers))
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):raise ValueError('Diagnostic requires FP32 weights.')
        for name,p in self.model.named_parameters():
            if 'lora_B' in name and torch.count_nonzero(p).item()!=0:
                raise ValueError(f'Loaded step-zero adapter has nonzero B: {name}')
        print('Diagnostic model loaded once; both zero adapters attached.',flush=True)

    def integrity(self):
        from pilot_eval.sft_backend import frozen_weight_hash
        import time
        started=time.monotonic();print('Diagnostic: checking frozen-base hash.',flush=True)
        value=frozen_weight_hash(self.model)
        print(f'Diagnostic: frozen-base hash read in {time.monotonic()-started:.1f}s.',flush=True)
        return value

    def capture(self,rows,condition):
        from contextlib import nullcontext
        from pilot_eval.trace_capture import CaptureFailure
        if condition not in ['base','sft-zero','grpo-zero']:raise ValueError('Unsupported diagnostic condition.')
        if condition!='base':self.model.set_adapter('default' if condition=='sft-zero' else 'grpo-zero')
        context=self.model.disable_adapter() if condition=='base' else nullcontext()
        result=[]
        with context,self.torch.inference_mode():
            for i,row in enumerate(rows):
                inputs={k:self.torch.tensor([row[k]],dtype=self.torch.long,device='cuda:0') for k in ['input_ids','attention_mask','position_ids']}
                try:
                    result.append(self.capture_hooks.forward(lambda:self.model(**inputs,use_cache=False,output_hidden_states=False)))
                except CaptureFailure as exc:
                    exc.diagnostics.update(example_id=row['id'],condition=condition)
                    raise
                print(f'Diagnostic {condition}: {i+1}/{len(rows)} documents; 29 sites captured.',flush=True)
        return result

    def close(self):
        import gc
        self.model=None;self.capture_hooks=None;gc.collect();self.torch.cuda.empty_cache()

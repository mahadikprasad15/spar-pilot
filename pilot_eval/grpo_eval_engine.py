"""Real pinned FP32 checkpoint inference, with no optimizer or training restore."""
from pilot_eval.grpo_gpu import GPUPreflight
from pilot_eval.sft_backend import frozen_weight_hash


class EvaluationDependencies:
    def load_evaluation(self,plan,settings,directory): return EvaluationEngine(plan,settings,directory)


class EvaluationEngine(GPUPreflight):
    def select_checkpoint(self,path,step):
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        weights=load_file(str(path/'adapter_model.safetensors'),device=str(self.model.device))
        expected=get_peft_model_state_dict(self.model)
        if set(weights)!=set(expected) or any(weights[k].shape!=expected[k].shape for k in weights):
            raise ValueError('evaluation adapter mapping/shape mismatch')
        if any(not self.torch.isfinite(w).all() for w in weights.values()): raise ValueError('nonfinite checkpoint adapter')
        if step==0 and any(self.torch.count_nonzero(w) for k,w in weights.items() if 'lora_B' in k):
            raise ValueError('checkpoint zero is not zero initialized')
        loaded=set_peft_model_state_dict(self.model,weights)
        if loaded.unexpected_keys: raise ValueError('unexpected adapter keys')
        self.model.eval()
        if self.model.is_gradient_checkpointing: self.model.gradient_checkpointing_disable()

    def generate(self,rows,policy,seed):
        # Check saved prompt length even though evaluation rows do not carry token IDs.
        for row in rows:
            if len(self.tokenizer.encode(row['prompt'],add_special_tokens=False))!=row['prompt_tokens']:
                raise ValueError('held-out prompt token count changed')
        settings={**self.settings,'generation_groups':policy['batch_size'],'evaluation_batch_size':policy['batch_size']}
        return self._generate(rows,settings,seed,draws=policy['draws'],greedy=policy['mode']=='greedy')

    def check_integrity(self):
        return dict(base_unchanged=frozen_weight_hash(self.model)==self.initial_base)

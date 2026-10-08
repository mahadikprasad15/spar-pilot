"""Real GRPO window execution and full-state checkpoint recovery."""
import random
import time
from pathlib import Path
import numpy as np

from pilot_eval.grpo_gpu import GPUPreflight
from pilot_eval.grpo_algorithm import gradient_consistency
from pilot_eval.grpo_baseline import _score
from pilot_eval.sft_backend import frozen_weight_hash
from pilot_eval.run import _write_json


class TrainingDependencies:
    def load_training(self,plan,settings,directory):return GRPOTrainingEngine(plan,settings,directory)


class GRPOTrainingEngine(GPUPreflight):
    """Reuses validated model/window construction, never disposable preflight updates."""
    def __init__(self,plan,settings,directory,*,sampler=None):
        super().__init__(plan,settings,directory,sampler=sampler)
        from transformers import enable_full_determinism
        enable_full_determinism(plan['seed'])
        self.optimizer=self._optimizer()
        self.scheduler=self.torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.0)
        self.window=self._window(settings,directory/'trainer')
        enable_full_determinism(plan['seed'])  # Trainer construction cannot consume scientific RNG.
        self.previous_gradient=None;self.global_step=0
        _write_json(directory/'resolved-trainer-args.json',self.window.resolved_args)

    def _sample(self,rows):
        """HF sampling consumes current saved Torch RNG; no per-step reseeding."""
        t=self.torch;tokenizer=self.tokenizer;outputs=[];self.model.eval()
        for start in range(0,len(rows),self.settings['generation_groups']):
            selected=rows[start:start+self.settings['generation_groups']]
            for row in selected:
                if tokenizer.encode(row['prompt'],add_special_tokens=False)!=row['input_ids'][:row['prompt_tokens']]:
                    raise ValueError('scientific prompt/tokenizer identity mismatch')
            inputs=tokenizer([r['prompt'] for r in selected for _ in range(8)],padding=True,
                add_special_tokens=False,return_tensors='pt').to(self.model.device)
            inputs.pop('token_type_ids',None)
            with t.inference_mode():
                result=self.model.generate(**inputs,max_new_tokens=self.settings['completion_limit'],do_sample=True,
                    temperature=1.0,top_p=self.settings['top_p'],top_k=self.settings['top_k'],num_beams=1,
                    num_return_sequences=1,repetition_penalty=1.0,no_repeat_ngram_size=0,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            eos=tokenizer.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
            for tokens in result[:,inputs['input_ids'].shape[1]:].tolist():
                end=next((i for i,x in enumerate(tokens) if x in eos),None)
                counted=tokens if end is None else tokens[:end];raw=tokens if end is None else tokens[:end+1]
                outputs.append(dict(text=tokenizer.decode(counted,skip_special_tokens=True),token_ids=counted,
                    generated_token_ids=raw,stop_reason='cap' if end is None else 'eos'))
            print(f'GRPO rollout: {min(start+self.settings["generation_groups"],8)}/8 prompts; 8 draws each',flush=True)
        return outputs

    def train_window(self,rows,step):
        if step!=self.global_step+1 or len(rows)!=8:raise ValueError('scientific optimizer step/order mismatch')
        self._sync();started=time.perf_counter();outputs=self._sample(rows);self._sync();generation=time.perf_counter()-started
        scored=[dict(step=step,prompt_id=r['id'],training_source_index=r.get('source_index'),
                     **_score(r,outputs[i*8+d],d)) for i,r in enumerate(rows) for d in range(8)]
        if any(len(o['generated_token_ids'])<1 for o in outputs):raise ValueError('empty raw training continuation')
        self.optimizer.zero_grad(set_to_none=True);begin=time.perf_counter()
        loss=self.window.backward([r['input_ids'][:r['prompt_tokens']] for r in rows],
            [o['generated_token_ids'] for o in outputs],[r['reward'] for r in scored],capped=[r['capped'] for r in scored])
        gradient=self.window.flat_gradient();norm=float(gradient.norm())
        cosine=gradient_consistency(self.previous_gradient,gradient)
        before=[p.detach().clone() for p in self.model.parameters() if p.requires_grad]
        self.torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
            self.plan['optimizer']['max_grad_norm'],error_if_nonfinite=True)
        lr=self.optimizer.param_groups[0]['lr'];self.optimizer.step();self.scheduler.step();self._sync()
        parameters=[p for p in self.model.parameters() if p.requires_grad]
        adapter_norm=float(sum(p.detach().double().square().sum() for p in parameters).sqrt())
        movement=float(sum((p.detach().double()-old.double()).square().sum() for p,old in zip(parameters,before)).sqrt())
        if not all(self.torch.isfinite(p).all() for p in parameters):raise ValueError('nonfinite adapter weights after update')
        backward=time.perf_counter()-begin
        for r,advantage in zip(scored,self.window.advantages):r['advantage']=advantage
        self.previous_gradient=gradient.clone();self.global_step=step
        return scored,dict(loss=loss,learning_rate=lr,pre_clip_gradient_norm=norm,gradient_consistency=cosine,
            adapter_norm=adapter_norm,adapter_norm_definition='L2 of concatenated A/B factors; not effective write norm',
            parameter_movement=movement,generation_seconds=generation,backward_update_seconds=backward,
            seconds=time.perf_counter()-started)

    def check_integrity(self):
        after=frozen_weight_hash(self.model)
        return dict(base_before=self.initial_base,base_after=after,base_unchanged=self.initial_base==after)

    def save_checkpoint(self,directory,step):
        if step!=self.global_step:raise ValueError('cannot save an inconsistent optimizer boundary')
        directory.mkdir(parents=True,exist_ok=True);t=self.torch
        self.model.save_pretrained(directory)
        t.save(self.optimizer.state_dict(),directory/'optimizer.pt')
        t.save(self.scheduler.state_dict(),directory/'scheduler.pt')
        t.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=t.get_rng_state(),
            cuda=t.cuda.get_rng_state_all() if self.model.device.type=='cuda' else []),directory/'rng_state.pth')
        t.save(dict(names=self.window.parameter_names,gradient=self.previous_gradient),directory/'prior-gradient.pt')
        _write_json(directory/'trainer_state.json',dict(global_step=step,accumulation_window_complete=True,
            resolved_args=self.window.resolved_args,base_sha256=self.initial_base))

    def restore_checkpoint(self,directory):
        import json
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict,get_peft_model_state_dict
        t=self.torch
        state=json.loads((directory/'trainer_state.json').read_text())
        if not state['accumulation_window_complete'] or state['base_sha256']!=self.initial_base:
            raise ValueError('checkpoint base/window mismatch')
        weights=load_file(str(directory/'adapter_model.safetensors'),device=str(self.model.device))
        expected=get_peft_model_state_dict(self.model)
        if set(weights)!=set(expected) or any(weights[k].shape!=expected[k].shape for k in weights):raise ValueError('adapter checkpoint mapping/shape mismatch')
        loaded=set_peft_model_state_dict(self.model,weights)
        if loaded.unexpected_keys:raise ValueError('unexpected adapter checkpoint keys')
        self.optimizer.load_state_dict(t.load(directory/'optimizer.pt',map_location=self.model.device,weights_only=True))
        self.scheduler.load_state_dict(t.load(directory/'scheduler.pt',weights_only=True))
        previous=t.load(directory/'prior-gradient.pt',map_location='cpu',weights_only=True)
        if previous['names']!=self.window.parameter_names:raise ValueError('checkpoint gradient parameter order mismatch')
        self.previous_gradient=previous['gradient'];self.global_step=state['global_step']
        rng=t.load(directory/'rng_state.pth',map_location='cpu',weights_only=False)
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);t.set_rng_state(rng['torch'])
        if self.model.device.type=='cuda':t.cuda.set_rng_state_all(rng['cuda'])

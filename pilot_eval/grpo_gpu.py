"""Real single-GPU FP32 preflight backend; no GPU-variant dispatch."""
import hashlib
import json
import time
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.workflow import _hash
from pilot_eval.grpo_algorithm import GRPOWindow, gradient_consistency
from pilot_eval.grpo_baseline import HFSampler, _score
from pilot_eval.sft_backend import frozen_weight_hash


class GPUPreflightDependencies:
    def load_preflight(self,plan,settings,directory):return GPUPreflight(plan,settings,directory)


class GPUPreflight:
    def __init__(self,plan,settings,directory,*,sampler=None):
        import torch
        from peft import LoraConfig,get_peft_model
        loading_start=time.perf_counter()
        self.torch=torch;self.plan=plan;self.settings=settings;self.directory=directory
        self.sampler=sampler or HFSampler(plan);self.tokenizer=self.sampler.backend.tokenizer
        model=self.sampler.backend.model
        expected={f'model.layers.{i}.{kind}.{proj}' for i in range(28) for kind,projs in
                  [('self_attn',['q_proj','k_proj','v_proj','o_proj']),('mlp',['gate_proj','up_proj','down_proj'])] for proj in projs}
        actual={n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.split('.')[-1] in plan['adapter']['projections']}
        if actual!=expected:raise ValueError('preflight requires exactly the intended 196 projections')
        self.model=get_peft_model(model,LoraConfig(r=1,lora_alpha=1,lora_dropout=0.,target_modules=sorted(actual),
                                   bias='none',task_type='CAUSAL_LM',init_lora_weights=True))
        self.sampler.backend.model=self.model
        self.model.config.pad_token_id=self.tokenizer.pad_token_id
        self.model.enable_input_require_grads()
        if any(p.requires_grad!=('lora_' in n) or p.dtype!=torch.float32 for n,p in self.model.named_parameters()):
            raise ValueError('only intended FP32 LoRA parameters may train; all weights must be FP32')
        self.initial={n:p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        if any(torch.count_nonzero(p) for n,p in self.model.named_parameters() if 'lora_B' in n):raise ValueError('nonzero initial adapter')
        self.initial_base=frozen_weight_hash(self.model)
        self.loading_seconds=time.perf_counter()-loading_start
        template_hash=_hash(self.tokenizer.chat_template)
        if template_hash!=plan['source_prompt_contract']['chat_template_sha256']:raise ValueError('chat template identity mismatch')

    def runtime(self):
        value=self.sampler.runtime()
        value.update(adapter_dtype='float32',total_memory_bytes=self.torch.cuda.get_device_properties(0).total_memory,
                     adapter_targets=196,gradient_checkpointing=self.settings['gradient_checkpointing'],
                     gradient_checkpointing_kwargs={'use_reentrant':False},generation_use_cache=True)
        _write_json(self.directory/'results/loading-timing.json',dict(seconds=self.loading_seconds))
        return value

    def _sync(self):
        if self.model.device.type=='cuda':self.torch.cuda.synchronize()

    def _reset(self):
        with self.torch.no_grad():
            for n,p in self.model.named_parameters():
                if p.requires_grad:p.copy_(self.initial[n])
        self.model.zero_grad(set_to_none=True);self.model.eval()
        if self.model.is_gradient_checkpointing:self.model.gradient_checkpointing_disable()

    def _generate(self,rows,settings,seed,*,draws=8,greedy=False,force_cap=False):
        from transformers import set_seed
        set_seed(seed);outputs=[];t=self.torch;tokenizer=self.tokenizer
        self.model.eval()
        group_batch=settings['evaluation_batch_size'] if greedy else settings['generation_groups']
        for start in range(0,len(rows),group_batch):
            selected=rows[start:start+group_batch]
            for row in selected:
                saved=row.get('input_ids',[])[:row.get('prompt_tokens',0)]
                if saved and tokenizer.encode(row['prompt'],add_special_tokens=False)!=saved:raise ValueError('saved prompt/tokenizer identity mismatch')
            inputs=tokenizer([r['prompt'] for r in selected for _ in range(draws)],return_tensors='pt',padding=True,
                             add_special_tokens=False).to(self.model.device)
            inputs.pop('token_type_ids',None)  # Qwen accepts token IDs and attention masks, not segment IDs.
            args=dict(max_new_tokens=settings['completion_limit'],num_beams=1,num_return_sequences=1,
                      repetition_penalty=1.0,no_repeat_ngram_size=0,pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,use_cache=True,do_sample=not greedy)
            if not greedy:args.update(temperature=1.0,top_p=settings['top_p'],top_k=settings['top_k'])
            if force_cap:args['min_new_tokens']=settings['completion_limit']
            with t.inference_mode():generated=self.model.generate(**inputs,**args)
            eos=tokenizer.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
            for tokens in generated[:,inputs['input_ids'].shape[1]:].tolist():
                end=next((i for i,x in enumerate(tokens) if x in eos),None)
                counted=tokens if end is None else tokens[:end]
                raw=tokens if end is None else tokens[:end+1]
                outputs.append(dict(text=tokenizer.decode(counted,skip_special_tokens=True),token_ids=counted,
                                    generated_token_ids=raw,stop_reason='cap' if end is None else 'eos'))
            print(f'preflight generation: {min(start+group_batch,len(rows))}/{len(rows)} prompts; draws {draws}; greedy {greedy}',flush=True)
        return outputs

    def zero_equivalence(self,rows,settings):
        self._reset();start=time.perf_counter()
        with self.model.disable_adapter():baseline=self._generate(rows,settings,self.plan['seed'],draws=1,greedy=True)
        zero=self._generate(rows,settings,self.plan['seed'],draws=1,greedy=True)
        elapsed=time.perf_counter()-start
        _write_json(self.directory/'results/zero-timing.json',dict(seconds=elapsed,completions=2*len(rows)))
        return [dict(id=r['id'],prompt=r['prompt'],gold=r['gold'],baseline_token_ids=a['generated_token_ids'],
                     zero_token_ids=b['generated_token_ids'],baseline_text=a['text'],zero_text=b['text'],
                     baseline_stop=a['stop_reason'],zero_stop=b['stop_reason']) for r,a,b in zip(rows,baseline,zero)]

    def _optimizer(self):
        o=self.plan['optimizer']
        return self.torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            lr=o['learning_rate'],betas=(o['beta1'],o['beta2']),eps=o['epsilon'],weight_decay=o['weight_decay'])

    def _window(self,settings,directory):
        return GRPOWindow(self.model,self.tokenizer,output_dir=directory,microbatch_groups=settings['backward_groups'],
                          seed=self.plan['seed'],completion_limit=settings['completion_limit'],
                          gradient_checkpointing=settings['gradient_checkpointing'])

    def _state_hash(self,optimizer):
        digest=hashlib.sha256()
        def visit(x):
            if isinstance(x,self.torch.Tensor):
                digest.update(str((tuple(x.shape),x.dtype)).encode());digest.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x,dict):
                for k in sorted(x,key=str):digest.update(str(k).encode());visit(x[k])
            elif isinstance(x,(list,tuple)):
                for v in x:visit(v)
            else:digest.update(repr(x).encode())
        visit({n:p for n,p in self.model.named_parameters() if p.requires_grad});visit(optimizer.state_dict())
        import random,numpy as np
        visit(self.torch.get_rng_state());visit(self.torch.cuda.get_rng_state_all() if self.model.device.type=='cuda' else [])
        visit(random.getstate());visit(np.random.get_state()[1].tolist())
        return digest.hexdigest()

    def trial(self,rows,settings,seed,directory):
        from transformers import enable_full_determinism
        from pilot_eval.grpo_controls import isolated_rng
        directory.mkdir(parents=True,exist_ok=True)
        t=self.torch;started=time.perf_counter()
        with isolated_rng():
            enable_full_determinism(seed);self._reset();before=frozen_weight_hash(self.model)
            optimizer=self._optimizer();window=self._window(settings,directory/'trainer')
            from transformers import enable_full_determinism
            enable_full_determinism(seed)
            _write_json(directory/'resolved-trainer-args.json',window.resolved_args)
            responses=[];steps=[];previous=None
            for step in range(2):
                self._sync();begin=time.perf_counter();selected=rows[step*8:(step+1)*8]
                outputs=self._generate(selected,settings,seed+step)
                scored=[_score(r,outputs[i*8+d],d) for i,r in enumerate(selected) for d in range(8)]
                optimizer.zero_grad(set_to_none=True)
                # Raw continuations include end-of-turn tokens; counted length excludes them.
                completions=[o['generated_token_ids'] for o in outputs]
                loss=window.backward([r['input_ids'][:r['prompt_tokens']] for r in selected],completions,
                                     [r['reward'] for r in scored],capped=[r['capped'] for r in scored])
                gradient=window.flat_gradient();norm=float(gradient.norm())
                consistency=gradient_consistency(previous,gradient)
                t.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],self.plan['optimizer']['max_grad_norm'],error_if_nonfinite=True)
                optimizer.step();self._sync()
                responses.extend([dict(step=step+1,**r) for r in scored])
                steps.append(dict(step=step+1,loss=loss,gradient_norm=norm,dead_group_fraction=window.dead_group_fraction,
                                  gradient_consistency=consistency,seconds=time.perf_counter()-begin))
                previous=gradient.clone();print(f'preflight disposable trial: step {step+1}/2',flush=True)
            state=self._state_hash(optimizer)
            import random,numpy as np
            self.model.save_pretrained(directory/'adapter')
            t.save(dict(optimizer=optimizer.state_dict(),torch_rng=t.get_rng_state(),cuda_rng=t.cuda.get_rng_state_all() if self.model.device.type=='cuda' else [],
                        python_rng=random.getstate(),numpy_rng=np.random.get_state()),directory/'optimizer-state.pt')
            after=frozen_weight_hash(self.model)
            (directory/'responses.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in responses))
            return dict(seed=seed,state_sha256=state,base_before=before,base_after=after,zero_initialized=True,
                        steps=steps,responses=responses,elapsed_seconds=time.perf_counter()-started)

    def capacity(self,rows,settings,directory):
        from pilot_eval.grpo_controls import isolated_rng
        directory.mkdir(parents=True,exist_ok=True);t=self.torch
        with isolated_rng():
            self._reset();started=time.perf_counter();before=frozen_weight_hash(self.model)
            t.cuda.empty_cache();t.cuda.reset_peak_memory_stats();self._sync();start=time.perf_counter()
            outputs=self._generate(rows,settings,settings['diagnostic_seed'],force_cap=True)
            self._sync();generation=time.perf_counter()-start
            if len(outputs)!=64 or any(len(o['generated_token_ids'])!=settings['completion_limit'] for o in outputs):raise ValueError('capacity generation did not reach final cap')
            # Artificial mixed rewards exercise a nonzero gradient; this is not a scientific rollout.
            optimizer=self._optimizer();window=self._window(settings,directory/'trainer');start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=window.backward([r['input_ids'][:r['prompt_tokens']] for r in rows],
                    [o['generated_token_ids'] for o in outputs],[i%2 for i in range(64)],capped=[False]*64)
            t.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],self.plan['optimizer']['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();self._sync();backward=time.perf_counter()-start
            peak=t.cuda.max_memory_allocated();reserved=t.cuda.max_memory_reserved();free,total=t.cuda.mem_get_info()
            self.model.save_pretrained(directory/'disposable-adapter')
            after=frozen_weight_hash(self.model)
            (directory/'synthetic-responses.jsonl').write_text(''.join(json.dumps(dict(diagnostic_only=True,**o))+'\n' for o in outputs))
            _write_json(directory/'resolved-trainer-args.json',window.resolved_args)
            return dict(generation_seconds=generation,backward_seconds=backward,elapsed_seconds=time.perf_counter()-started,
                        peak_allocated_bytes=peak,peak_reserved_bytes=reserved,
                        free_memory_bytes=min(free,total-reserved),synthetic_full_cap=True,completions=64,
                        completion_tokens=settings['completion_limit'],base_unchanged=before==after,loss=loss,
                        note='Longest eight saved prompts; forced-cap generation and artificial binary rewards stress memory. Not scientific training.')

    def close(self):
        del self.model,self.initial
        self.sampler.close()

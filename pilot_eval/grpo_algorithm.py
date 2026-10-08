"""Pilot 4 GRPO window contract, pinned TRL loss and gradient diagnostics.

Eight prompt groups are normalized together before any memory microbatching.
Direct backward on DAPO sums contributions; never divide again by accumulation.
"""
import math
import numpy as np


def group_advantages(rewards):
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or len(values) % 8 or not np.isfinite(values).all():
        raise ValueError('finite rewards in complete groups of eight required')
    groups = values.reshape(-1, 8)
    return ((groups - groups.mean(1, keepdims=True)) /
            (groups.std(1, ddof=1, keepdims=True) + 1e-4)).reshape(-1).tolist()


class GRPOWindow:
    """Public training boundary: one fresh 8-prompt/64-completion window.

    Uses the actual pinned GRPOTrainer.compute_loss. It owns no generation,
    optimizer or data order; callers supply a fresh rollout and step exactly once.
    """
    def __init__(self, model, tokenizer, *, output_dir, microbatch_groups, seed=42, completion_limit=512, gradient_checkpointing=False):
        import importlib.metadata
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
        if importlib.metadata.version('trl') != '0.26.2':
            raise ValueError('GRPO window requires TRL 0.26.2; use a separately resolved environment')
        if microbatch_groups not in [1, 2, 4, 8]:
            raise ValueError('microbatch must preserve whole groups and divide eight prompts')
        self.model, self.microbatch = model, microbatch_groups * 8
        self.parameter_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        if not self.parameter_names or any('lora_' not in n for n in self.parameter_names):
            raise ValueError('only LoRA parameters may train')
        args = GRPOConfig(output_dir=str(output_dir), use_cpu=next(model.parameters()).device.type == 'cpu',
            bf16=False, fp16=False, tf32=False, seed=seed, data_seed=seed,
            full_determinism=True, per_device_train_batch_size=self.microbatch,
            gradient_accumulation_steps=64 // self.microbatch, generation_batch_size=64,
            num_generations=8, num_iterations=1, loss_type='dapo', beta=0.0,
            scale_rewards='group', temperature=1.0, top_p=1.0, top_k=0,
            epsilon=.2, epsilon_high=.2, delta=None, importance_sampling_level='token',
            top_entropy_quantile=1.0, mask_truncated_completions=False,
            use_vllm=False, use_liger_kernel=False, disable_dropout=True,
            max_completion_length=completion_limit, gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1e-4, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
            weight_decay=0, lr_scheduler_type='constant', warmup_steps=0,
            max_grad_norm=1.0, max_steps=64, report_to=[], save_strategy='no',
            logging_strategy='no', dataloader_num_workers=0)
        self.trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=args,
            train_dataset=Dataset.from_dict({'prompt': ['bad'] * 8}),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions))
        if gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.resolved_args = args.to_dict()
        self.trainer.current_gradient_accumulation_steps = args.gradient_accumulation_steps
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError('explicit padding token required')

    def backward(self, prompts, completions, rewards, *, capped, sign=1):
        import torch
        if len(prompts) != 8 or len(completions) != 64 or len(rewards) != 64 or len(capped) != 64:
            raise ValueError('one optimizer window requires eight prompts and 64 draws')
        if sign not in [-1, 1] or any(r not in [0, 1] for r in rewards) or any(type(c) is not bool for c in capped):
            raise ValueError('binary rewards, explicit cap flags and a valid control sign required')
        if any(not ids or any(type(t) is not int or t < 0 for t in ids) for ids in prompts + completions):
            raise ValueError('nonempty token sequences required')
        effective = [0.0 if cap else float(r) for r, cap in zip(rewards, capped)]
        self.advantages = [sign * a for a in group_advantages(effective)]
        self.dead_group_fraction = sum(len(set(effective[i:i + 8])) == 1 for i in range(0, 64, 8)) / 8
        device = next(self.model.parameters()).device
        pw, cw = max(map(len, prompts)), max(map(len, completions))
        prompt_ids, prompt_mask = [], []
        for ids in prompts:
            prompt_ids.extend([[self.pad_id] * (pw - len(ids)) + ids] * 8)
            prompt_mask.extend([[0] * (pw - len(ids)) + [1] * len(ids)] * 8)
        batch = dict(prompt_ids=torch.tensor(prompt_ids, device=device),
                     prompt_mask=torch.tensor(prompt_mask, device=device),
                     completion_ids=torch.tensor([ids + [self.pad_id] * (cw - len(ids)) for ids in completions], device=device),
                     completion_mask=torch.tensor([[1] * len(ids) + [0] * (cw - len(ids)) for ids in completions], device=device),
                     advantages=torch.tensor(self.advantages, dtype=torch.float32, device=device),
                     num_items_in_batch=torch.tensor(sum(map(len, completions)), device=device))
        self.model.train()
        total = 0.0
        for start in range(0, 64, self.microbatch):
            micro = {k: v if k == 'num_items_in_batch' else v[start:start + self.microbatch] for k, v in batch.items()}
            loss = self.trainer.compute_loss(self.model, micro)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite GRPO policy loss')
            loss.backward()  # DAPO already uses the entire window denominator.
            total += float(loss.detach())
        for n, p in self.model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise ValueError('missing/nonfinite adapter gradient: ' + n)
        return total

    def flat_gradient(self):
        import torch
        by_name = dict(self.model.named_parameters())
        return torch.cat([by_name[n].grad.detach().reshape(-1).cpu().double() for n in self.parameter_names])


def gradient_consistency(previous, current):
    """Consecutive accumulated pre-clip gradients; undefined is never zero cosine."""
    if previous is None:
        return dict(cosine=None, reason='first_step', defined=False)
    if len(previous) != len(current):
        raise ValueError('gradient parameter order/size changed')
    a, b = np.asarray(previous, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite gradient consistency input')
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return dict(cosine=None, reason='zero_gradient', defined=False)
    return dict(cosine=float(np.clip(np.dot(a, b) / (na * nb), -1, 1)), reason=None, defined=True)

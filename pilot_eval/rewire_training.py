"""Sparse rewiring ticket 07: explicit training loop adapted from ``sampling_sft_training_backend.train``.

Carries over every safety check from ``sampling_sft_training_backend.train`` (spec test 19):
- FP32 weight check across all parameters
- Context-limit refusal (never truncate or skip)
- Frozen base-weight hash check before training and at every checkpoint
- ``gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})``
- ``AdamW(..., foreach=False)``

Adds the sparse-rewiring method (D1-D15, D20, D24):
- Two AdamW parameter groups: read factors (LR 3e-3 rank 1 / 3e-4 rank 0) and mask logits (LR 0.1),
  betas (0.9, 0.99), eps 1e-8, weight_decay 0.
- Shared linear warmup (20) + cosine decay to 0 at ``total_steps`` (1,024) via ``LambdaLR`` on both groups.
- Gradient clipping (1.0) on read factors ONLY; masks unclipped, norms logged separately.
- D14 shared-mask cache: microbatch 1, one uniform ``u`` per layer per group of 4 consecutive examples.
- Token-weighted step CE via ``sft_backend.completion_loss`` + ``lambda * expected_active`` once per step.
- Hardening/freeze after ``freeze_step`` (960): records ``hardening.json`` (pre/post counts and val CE),
  hardens ``log_alpha`` to binary ``+20/-20`` with ``requires_grad=False``, disables L0 penalty for the
  final 64 fixed-mask steps.
- Dynamics logging (Ticket 06): per-step rows, every-16-step FP16 ``log_alpha`` snapshots + validation CE
  hook, every-64-step settle diagnostics (logged only), D24 checkpoints (weights + masks + seal only).

Precision is NEVER set via legacy ``allow_tf32`` flags; it comes from ``TransformersBackend`` (ticket 14).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pilot_eval import attribution_units as au
from pilot_eval import hard_concrete as hc
from pilot_eval import rewire_dynamics as rd
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import sft_backend as sb
from pilot_eval.workflow import _hash


def warmup_cosine_multiplier(step_index: int, *, warmup_steps: int, total_steps: int) -> float:
    """Shared LR multiplier for 0-based optimizer step ``step_index``: linear warmup then cosine to 0."""
    if total_steps < 1:
        raise ValueError(f'total_steps must be >= 1, got {total_steps}')
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(f'warmup_steps ({warmup_steps}) must be in [0, total_steps={total_steps}]')
    if warmup_steps > 0 and step_index < warmup_steps:
        return float(step_index + 1) / float(warmup_steps)
    if step_index >= total_steps:
        return 0.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = float(step_index - warmup_steps) / float(decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def split_read_and_mask_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split ``rewire_`` trainable parameters into ``(read_params, mask_params)``."""
    read_params: list[torch.nn.Parameter] = []
    mask_params: list[torch.nn.Parameter] = []
    for name, param in rm.trainable_parameters(model):
        if name.endswith('rewire_log_alpha'):
            mask_params.append(param)
        else:
            read_params.append(param)
    if not read_params or not mask_params:
        raise ValueError('expected both read factors and mask logits among rewire trainable parameters')
    return read_params, mask_params


def _layer_norms(modules: Sequence[rm.RewireMLP]) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    gate_norms: dict[int, torch.Tensor] = {}
    up_norms: dict[int, torch.Tensor] = {}
    for mod in modules:
        z_det = hc.deterministic_gate(mod.rewire_log_alpha)
        if mod.rank == 0:
            gate_norms[mod.layer] = hc.row_change_norms(rank=0, w=mod.rewire_W_gate, z=z_det)
            up_norms[mod.layer] = hc.row_change_norms(rank=0, w=mod.rewire_W_up, z=z_det)
        else:
            gate_norms[mod.layer] = hc.row_change_norms(
                rank=mod.rank, a=mod.rewire_A_gate, b=mod.rewire_B_gate, z=z_det
            )
            up_norms[mod.layer] = hc.row_change_norms(
                rank=mod.rank, a=mod.rewire_A_up, b=mod.rewire_B_up, z=z_det
            )
    return gate_norms, up_norms


def _unclipped_grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    grads = [p.grad.detach().float() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_sq = sum(float(torch.sum(g * g).item()) for g in grads)
    return math.sqrt(total_sq)


def train_rewire(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    step_batches: Sequence[Sequence[str]],
    out_dir: str | Path,
    *,
    rank: int,
    read_lr: float,
    mask_lr: float = 0.1,
    l0_lambda: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    eps: float = 1e-8,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 20,
    freeze_step: int | None = 960,
    mask_seed: int = 1042,
    mask_group_size: int = hc.DEFAULT_GROUP_SIZE,
    checkpoint_steps: Sequence[int] = (0, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1024),
    snapshot_interval: int = rd.SNAPSHOT_INTERVAL,
    settle_start: int = rd.SETTLE_START_STEP,
    settle_interval: int = rd.SETTLE_INTERVAL,
    enable_settle_diagnostics: bool = True,
    gradient_checkpointing: bool = True,
    model_context_limit: int = 32768,
    pad_token_id: int = 0,
    expected_base_sha256: str | None = None,
    plan_sha256: str = 'plan-unset',
    runtime_sha256: str = 'runtime-unset',
    code_sha256: str = 'code-unset',
    val_ce_fn: Callable[[torch.nn.Module, int], Mapping[str, Any]] | None = None,
    after_step: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the explicit sparse-rewiring training loop and write logs, snapshots, checkpoints, and hardening record."""
    if not rows:
        raise ValueError('training rows must not be empty')
    if not step_batches:
        raise ValueError('step_batches must not be empty')
    if max(len(r['input_ids']) for r in rows) > int(model_context_limit):
        raise ValueError('Training item exceeds model_context_limit; never truncate or skip.')
    if not all(p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError('Every model and rewire weight must be FP32.')

    mods = rm.modules(model)
    if any(m.rank != int(rank) for m in mods):
        raise ValueError(f'installed rewire rank does not match requested rank {rank}')

    base_before = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
    if expected_base_sha256 is not None and base_before != expected_base_sha256:
        raise ValueError(f'Loaded frozen base hash {base_before} differs from expected {expected_base_sha256}.')

    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    read_params, mask_params = split_read_and_mask_parameters(model)
    device = next(model.parameters()).device
    total_steps = len(step_batches)

    optimizer = torch.optim.AdamW(
        [
            {'params': read_params, 'lr': float(read_lr)},
            {'params': mask_params, 'lr': float(mask_lr)},
        ],
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
        weight_decay=0.0,
        foreach=False,
    )
    lr_fn = lambda step_idx: warmup_cosine_multiplier(step_idx, warmup_steps=warmup_steps, total_steps=total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_fn, lr_fn])

    cache = hc.SharedMaskCache(seed=int(mask_seed), group_size=int(mask_group_size), device=device)
    rm.set_gate(model, cache.gate)
    layer_shapes = {m.layer: tuple(m.rewire_log_alpha.shape) for m in mods}

    by_id = {r['id']: r for r in rows}
    collator = sb.CompletionCollator(int(pad_token_id))
    out_path = Path(out_dir)
    steps_log_path = out_path / 'logs' / 'steps.jsonl'
    val_log_path = out_path / 'logs' / 'validation.jsonl'
    settle_log_path = out_path / 'logs' / 'settle.jsonl'
    ckpt_set = set(int(s) for s in checkpoint_steps)

    tracker = (
        rd.SettleTracker(start=int(settle_start), interval=int(settle_interval))
        if enable_settle_diagnostics and int(settle_start) >= int(settle_interval)
        else None
    )

    def _log_val(step: int) -> dict[str, Any] | None:
        if val_ce_fn is None:
            return None
        res = dict(val_ce_fn(model, step))
        res['step'] = int(step)
        rd.append_jsonl(val_log_path, res)
        return res

    def _save_checkpoint(step: int, *, frozen: bool) -> None:
        if step not in ckpt_set:
            return
        if sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter) != base_before:
            raise ValueError('Training modified a frozen base parameter.')
        ckpt_dir = out_path / 'checkpoints' / f'checkpoint-{step}'
        reval.save_rewire_checkpoint(
            ckpt_dir,
            model,
            step=step,
            rank=int(rank),
            plan_sha256=plan_sha256,
            runtime_sha256=runtime_sha256,
            code_sha256=code_sha256,
            frozen=frozen,
        )
        gate_norms, up_norms = _layer_norms(mods)
        la_map = {m.layer: m.rewire_log_alpha for m in mods}
        surv = hc.export_survivors(
            la_map,
            frozen=frozen,
            norms_gate_by_layer=gate_norms,
            norms_up_by_layer=up_norms,
            step=step,
            meta={'rank': int(rank)},
        )
        hc.seal_survivor_export(out_path / 'masks' / f'step-{step}', surv, filename=f'{step}.json')

    # Step 0 baseline snapshot, dynamics row, validation CE, and checkpoint
    frozen = False
    gate_norms_0, up_norms_0 = _layer_norms(mods)
    la_map_0 = {m.layer: m.rewire_log_alpha for m in mods}
    row0 = rd.step_dynamics_row(
        step=0,
        log_alpha_by_layer=la_map_0,
        norms_gate_by_layer=gate_norms_0,
        norms_up_by_layer=up_norms_0,
        frozen=False,
        lr_reads=float(optimizer.param_groups[0]['lr']),
        lr_masks=float(optimizer.param_groups[1]['lr']),
        examples_seen=0,
    )
    rd.append_jsonl(steps_log_path, row0)
    if tracker is not None:
        tracker.observe(0, la_map_0)
    if rd.should_snapshot(0, interval=snapshot_interval):
        rd.save_log_alpha_snapshot(out_path / 'masks' / 'trace' / '0.npz', la_map_0, step=0)
        _log_val(0)
    _save_checkpoint(0, frozen=False)

    model.train()
    examples_seen = 0
    hardening_record: dict[str, Any] | None = None
    step_records: list[dict[str, Any]] = []

    for step_idx, batch_ids in enumerate(step_batches, start=1):
        selected = [by_id[eid] for eid in batch_ids]
        if not selected:
            raise ValueError(f'step {step_idx} has an empty batch')
        denominator = sum(sum(int(v) != -100 for v in r['labels'][1:]) for r in selected)
        if denominator <= 0:
            raise ValueError(f'step {step_idx} has no supervised completion tokens')

        optimizer.zero_grad(set_to_none=True)
        boundaries = cache.begin_step(step_idx, len(selected))
        ce_parts: list[float] = []

        for start, end in boundaries:
            if not frozen:
                cache.draw_group(layer_shapes, dtype=torch.float32)
            for row in selected[start:end]:
                values = collator([row])
                values.pop('example_ids', None)
                labels = values.pop('labels').to(device)
                output = model(**{k: v.to(device) for k, v in values.items()}, use_cache=False)
                ce_loss = sb.completion_loss(output, labels, denominator)
                ce_parts.append(float(ce_loss.detach().item()))
                ce_loss.backward()
                del output, ce_loss
        cache.clear()

        ce_total = math.fsum(ce_parts)
        if not frozen and float(l0_lambda) > 0.0:
            exp_active_sum = sum(hc.expected_active(m.rewire_log_alpha) for m in mods)
            l0_term = float(l0_lambda) * exp_active_sum
            l0_val = float(l0_term.detach().item())
            l0_term.backward()
            del exp_active_sum, l0_term
        else:
            l0_val = 0.0

        # Check finiteness of all active gradients
        active_params = read_params if frozen else (read_params + mask_params)
        active_grads = [p.grad for p in active_params if p.grad is not None]
        if not active_grads or not all(torch.isfinite(g).all().item() for g in active_grads):
            raise ValueError(f'step {step_idx}: missing or nonfinite training gradients')

        # D4: clip read factors ONLY; masks remain unclipped
        grad_norm_reads = float(
            torch.nn.utils.clip_grad_norm_(read_params, float(max_grad_norm), error_if_nonfinite=True)
        )
        grad_norm_masks = 0.0 if frozen else _unclipped_grad_norm(mask_params)

        lr_reads_now = float(optimizer.param_groups[0]['lr'])
        lr_masks_now = float(optimizer.param_groups[1]['lr'])
        optimizer.step()
        scheduler.step()

        if scheduler.last_epoch != step_idx or not all(torch.isfinite(p).all().item() for p in active_params):
            raise ValueError(f'step {step_idx}: optimizer/scheduler progress or parameter finiteness failed')

        examples_seen += len(selected)
        gate_norms, up_norms = _layer_norms(mods)
        la_map = {m.layer: m.rewire_log_alpha for m in mods}

        # Dynamics row (D20)
        dyn_row = rd.step_dynamics_row(
            step=step_idx,
            log_alpha_by_layer=la_map,
            norms_gate_by_layer=gate_norms,
            norms_up_by_layer=up_norms,
            frozen=frozen,
            ce_loss=ce_total,
            l0_loss=l0_val,
            total_loss=ce_total + l0_val,
            lr_reads=lr_reads_now,
            lr_masks=lr_masks_now,
            grad_norm_reads=grad_norm_reads,
            grad_norm_masks=grad_norm_masks,
            examples_seen=examples_seen,
        )
        rd.append_jsonl(steps_log_path, dyn_row)
        step_records.append(dyn_row)

        if tracker is not None:
            settle_rec = tracker.observe(step_idx, la_map)
            if settle_rec is not None:
                rd.append_jsonl(settle_log_path, settle_rec)

        if rd.should_snapshot(step_idx, interval=snapshot_interval):
            rd.save_log_alpha_snapshot(out_path / 'masks' / 'trace' / f'{step_idx}.npz', la_map, step=step_idx)
            _log_val(step_idx)

        _save_checkpoint(step_idx, frozen=frozen)

        # Hardening / freeze at freeze_step (D5: 960 is last continuous-mask step)
        if freeze_step is not None and step_idx == int(freeze_step):
            pre_counts = hc.count_gates(la_map, frozen=False)
            val_before = dict(val_ce_fn(model, step_idx)) if val_ce_fn is not None else None
            for m in mods:
                hc.freeze_log_alpha(m.rewire_log_alpha)
            cache.freeze()
            rm.set_gate(model, None)
            frozen = True
            post_counts = hc.count_gates({m.layer: m.rewire_log_alpha for m in mods}, frozen=True)
            val_after = dict(val_ce_fn(model, step_idx)) if val_ce_fn is not None else None
            hardening_record = {
                'freeze_step': int(freeze_step),
                'rule': 'z_det > 0 (log_alpha > -ln(11)) -> +20.0 (binary 1), else -20.0 (binary 0)',
                'pre_expected_active': pre_counts['total']['expected_active'],
                'pre_nonzero': pre_counts['total']['nonzero_z_det'],
                'post_binary': post_counts['total']['binary_survivors'],
                'val_ce_before': val_before['val_ce'] if val_before is not None and 'val_ce' in val_before else None,
                'val_ce_after': val_after['val_ce'] if val_after is not None and 'val_ce' in val_after else None,
            }
            (out_path / 'hardening.json').write_bytes(au._encode(hardening_record))

        if after_step is not None:
            after_step(step_idx, dyn_row)

    base_after = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
    if base_after != base_before:
        raise ValueError('Training modified a frozen base parameter.')

    return {
        'successful_steps': total_steps,
        'examples_seen': examples_seen,
        'rank': int(rank),
        'frozen': frozen,
        'base_sha256_before': base_before,
        'base_sha256_after': base_after,
        'checkpoint_steps': sorted(s for s in ckpt_set if s <= total_steps),
        'hardening': hardening_record,
        'mask_group_logs': cache.step_logs,
        'plan_sha256': plan_sha256,
        'runtime_sha256': runtime_sha256,
        'code_sha256': code_sha256,
        'final_step_record': step_records[-1],
    }

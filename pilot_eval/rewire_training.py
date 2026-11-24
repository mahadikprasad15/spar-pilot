"""Sparse rewiring ticket 07: explicit training loop adapted from ``sampling_sft_training_backend.train``.

Carries over every safety check from ``sampling_sft_training_backend.train`` (spec test 19, lesson 2026-11-20):
- FP32 weight check across all parameters
- Context-limit refusal (never truncate or skip; ``model_context_limit`` is required)
- Required ``expected_base_sha256`` check on real runs (only ``execution_kind='cpu-fixture'`` may omit it)
- Step-0 zero-change assertion (every read coefficient ``B`` / ``W`` is exact zero)
- Frozen base-weight hash check before training, at every checkpoint, and after training
- ``gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})``
- ``AdamW(..., foreach=False)``
- Full determinism (D8): ``torch.use_deterministic_algorithms(True)`` and ``CUBLAS_WORKSPACE_CONFIG`` on CUDA

Integrates with PR #6 (``sparse_rewiring_plan``) and PR #16 (``rewire_checkpoint.CheckpointSink``):
- Drives ``LambdaLR`` directly from ``sparse_rewiring_plan.lr_multiplier`` / ``plan['steps'][s-1]['lr_multiplier']``
  so all 1,024 steps match the frozen plan identically (reaching exact 0 at step 1,024).
- Verifies ``plan['steps'][s-1]['mask_groups']`` and ``masks`` phase when a frozen plan is supplied.
- Calls ``sink(0, model)`` at step 0 and ``sink(step, model)`` after every optimizer step (ticket 08's
  ``CheckpointSink`` owns checkpoint writing and sealing; checkpoint 960 is saved before hardening and
  1,024 after).
- Logs D14 ``u`` seed and group boundaries in every step row of ``logs/steps.jsonl``.
- Precision is NEVER set via legacy TF32 flags; it comes from ``TransformersBackend`` (ticket 14).
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pilot_eval import attribution_units as au
from pilot_eval import hard_concrete as hc
from pilot_eval import rewire_dynamics as rd
from pilot_eval import rewire_module as rm
from pilot_eval import sft_backend as sb
from pilot_eval import sparse_rewiring_plan as sp
from pilot_eval.workflow import _hash


def rows_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Canonical SHA-256 of training rows (matching ``sparse_rewiring_plan.plan_from_files`` ``_hash(rows)``)."""
    return _hash([dict(r) for r in rows])


def warmup_cosine_multiplier(step_index: int, *, warmup_steps: int, total_steps: int) -> float:
    """Shared LR multiplier for 0-based ``LambdaLR`` count ``step_index``, matching ``sparse_rewiring_plan.lr_multiplier``."""
    step_1based = int(step_index) + 1
    if step_1based > int(total_steps):
        return 0.0
    if int(warmup_steps) == 0:
        if int(total_steps) == 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * (step_1based - 1) / (int(total_steps) - 1)))
    return float(sp.lr_multiplier(step_1based, warmup=int(warmup_steps), total=int(total_steps)))


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


def _assert_step0_zero_coefficients(modules: Sequence[rm.RewireMLP]) -> None:
    for mod in modules:
        for part, coeff in mod.coefficients().items():
            if torch.count_nonzero(coeff).item() != 0:
                raise ValueError(f'step 0 initialization check failed: layer {mod.layer} {part} coefficient is nonzero')


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
    out_dir: str | Path,
    *,
    plan: Mapping[str, Any] | None = None,
    step_batches: Sequence[Sequence[str]] | None = None,
    sink: Callable[[int, torch.nn.Module], None] | None = None,
    rank: int | None = None,
    read_lr: float | None = None,
    mask_lr: float = 0.1,
    l0_lambda: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    eps: float = 1e-8,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 20,
    freeze_step: int | None = 960,
    mask_seed: int = 1042,
    mask_group_size: int = hc.DEFAULT_GROUP_SIZE,
    checkpoint_steps: Sequence[int] | None = None,
    snapshot_interval: int = rd.SNAPSHOT_INTERVAL,
    settle_start: int = rd.SETTLE_START_STEP,
    settle_interval: int = rd.SETTLE_INTERVAL,
    enable_settle_diagnostics: bool = True,
    gradient_checkpointing: bool = True,
    model_context_limit: int,
    pad_token_id: int,
    expected_base_sha256: str | None = None,
    expected_rows_sha256: str | None = None,
    execution_kind: str = 'real-cuda',
    max_survivor_records: int | None = 10_000,
    val_ce_fn: Callable[[torch.nn.Module, int], Mapping[str, Any]] | None = None,
    after_step: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the explicit sparse-rewiring training loop and write logs, snapshots, survivor exports, and hardening record."""
    if not rows:
        raise ValueError('training rows must not be empty')
    val_steps_set: set[int] | None = None
    snap_steps_set: set[int] | None = None
    if plan is not None:
        resolved_rank = int(plan['rank']) if rank is None else int(rank)
        # Canonical Ticket 05 plan stores read LR at plan['optimizer']['reads']['lr'] (P1.2)
        if read_lr is not None:
            resolved_read_lr = float(read_lr)
        elif 'optimizer' in plan and 'reads' in plan['optimizer'] and 'lr' in plan['optimizer']['reads']:
            resolved_read_lr = float(plan['optimizer']['reads']['lr'])
        elif 'read_lr' in plan:
            resolved_read_lr = float(plan['read_lr'])
        else:
            raise ValueError('plan must provide optimizer.reads.lr')
        if 'optimizer' in plan:
            opt_cfg = plan['optimizer']
            if 'masks' in opt_cfg and 'lr' in opt_cfg['masks']:
                mask_lr = float(opt_cfg['masks']['lr'])
            if 'reads' in opt_cfg:
                if 'betas' in opt_cfg['reads']:
                    betas = (float(opt_cfg['reads']['betas'][0]), float(opt_cfg['reads']['betas'][1]))
                if 'eps' in opt_cfg['reads']:
                    eps = float(opt_cfg['reads']['eps'])
                if 'clip_norm' in opt_cfg['reads']:
                    max_grad_norm = float(opt_cfg['reads']['clip_norm'])
        plan_steps = list(plan['steps'])
        batches = [list(s['ids']) for s in plan_steps]
        resolved_ckpts = list(plan['checkpoints']) if checkpoint_steps is None else list(checkpoint_steps)
        if 'validation_steps' in plan:
            val_steps_set = set(int(s) for s in plan['validation_steps'])
        if 'gate_snapshot_steps' in plan:
            snap_steps_set = set(int(s) for s in plan['gate_snapshot_steps'])
        if 'settings' in plan and 'training' in plan['settings']:
            tr = plan['settings']['training']
            warmup_steps = int(tr['warmup_steps'])
            freeze_step = int(tr['freeze_after_step'])
        if 'settings' in plan and 'objective' in plan['settings']:
            l0_lambda = float(plan['settings']['objective']['l0_lambda'])
        if 'settings' in plan and 'seeds' in plan['settings']:
            mask_seed = int(plan['settings']['seeds']['mask_noise'])
        if 'settings' in plan and 'data' in plan['settings']:
            mask_group_size = int(plan['settings']['data']['examples_per_mask'])
        # P1.3: bind actual row tokens/labels to plan['inputs']['training_items_sha256']
        plan_rows_sha = (plan.get('inputs') or {}).get('training_items_sha256')
        if plan_rows_sha is not None:
            observed_rows_sha = rows_content_sha256(rows)
            if observed_rows_sha != plan_rows_sha:
                raise ValueError(
                    f'training rows content hash {observed_rows_sha} differs from plan training_items_sha256 {plan_rows_sha}'
                )
    else:
        if step_batches is None or rank is None or read_lr is None:
            raise ValueError('when plan is None, step_batches, rank, and read_lr are required')
        resolved_rank = int(rank)
        resolved_read_lr = float(read_lr)
        plan_steps = None
        batches = [list(b) for b in step_batches]
        resolved_ckpts = list(checkpoint_steps) if checkpoint_steps is not None else [0, len(batches)]

    if expected_rows_sha256 is not None:
        observed_rows_sha = rows_content_sha256(rows)
        if observed_rows_sha != expected_rows_sha256:
            raise ValueError(
                f'training rows content hash {observed_rows_sha} differs from expected {expected_rows_sha256}'
            )

    if not batches:
        raise ValueError('step_batches must not be empty')
    if max(len(r['input_ids']) for r in rows) > int(model_context_limit):
        raise ValueError('Training item exceeds model_context_limit; never truncate or skip.')
    if not all(p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError('Every model and rewire weight must be FP32.')

    mods = rm.modules(model)
    if any(m.rank != resolved_rank for m in mods):
        raise ValueError(f'installed rewire rank does not match requested rank {resolved_rank}')
    _assert_step0_zero_coefficients(mods)

    # Lesson 2026-11-20: expected_base_sha256 is required on real runs (only cpu-fixture may omit it)
    base_before = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
    if execution_kind != 'cpu-fixture':
        if not expected_base_sha256:
            raise ValueError('expected_base_sha256 is required when execution_kind != "cpu-fixture"')
    if expected_base_sha256 is not None and base_before != expected_base_sha256:
        raise ValueError(f'Loaded frozen base hash {base_before} differs from expected {expected_base_sha256}.')

    # D8: full determinism (and CUBLAS_WORKSPACE_CONFIG required on CUDA)
    device = next(model.parameters()).device
    if device.type == 'cuda' and not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be set before CUDA training for full determinism (D8)')
    torch.use_deterministic_algorithms(True)

    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    read_params, mask_params = split_read_and_mask_parameters(model)
    total_steps = len(batches)

    optimizer = torch.optim.AdamW(
        [
            {'params': read_params, 'lr': resolved_read_lr},
            {'params': mask_params, 'lr': float(mask_lr)},
        ],
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
        weight_decay=0.0,
        foreach=False,
    )
    if plan_steps is not None:
        plan_mults = [float(s['lr_multiplier']) for s in plan_steps]
        lr_fn = lambda i: plan_mults[i] if i < len(plan_mults) else 0.0
    else:
        lr_fn = lambda i: warmup_cosine_multiplier(i, warmup_steps=warmup_steps, total_steps=total_steps)
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
    ckpt_set = set(int(s) for s in resolved_ckpts)
    sealed_files: list[str] = ['logs/steps.jsonl']

    tracker = (
        rd.SettleTracker(start=int(settle_start), interval=int(settle_interval), freeze_step=freeze_step)
        if enable_settle_diagnostics and int(settle_start) >= int(settle_interval)
        else None
    )

    def _log_val(step: int) -> dict[str, Any] | None:
        if val_ce_fn is None:
            return None
        res = dict(val_ce_fn(model, step))
        res['step'] = int(step)
        rd.append_jsonl(val_log_path, res)
        if 'logs/validation.jsonl' not in sealed_files:
            sealed_files.append('logs/validation.jsonl')
        return res

    def _on_checkpoint_step(step: int, *, frozen: bool) -> None:
        if step in ckpt_set:
            if sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter) != base_before:
                raise ValueError('Training modified a frozen base parameter.')
            gate_norms, up_norms = _layer_norms(mods)
            la_map = {m.layer: m.rewire_log_alpha for m in mods}
            surv = hc.export_survivors(
                la_map,
                frozen=frozen,
                norms_gate_by_layer=gate_norms,
                norms_up_by_layer=up_norms,
                step=step,
                meta={'rank': resolved_rank},
                max_survivor_records=max_survivor_records,
            )
            rel = f'masks/step-{step}/{step}.json'
            hc.seal_survivor_export(out_path / 'masks' / f'step-{step}', surv, filename=f'{step}.json')
            if rel not in sealed_files:
                sealed_files.append(rel)
        if sink is not None:
            sink(step, model)

    # Step 0 baseline snapshot, dynamics row, validation CE, and sink(0, model)
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
    row0['mask_group_log'] = None
    rd.append_jsonl(steps_log_path, row0, strict_monotonic_step=True)
    if tracker is not None:
        tracker.observe(0, la_map_0)
    if (snap_steps_set is not None and 0 in snap_steps_set) or (
        snap_steps_set is None and rd.should_snapshot(0, interval=snapshot_interval)
    ):
        rd.save_log_alpha_snapshot(out_path / 'masks' / 'trace' / '0.npz', la_map_0, step=0)
    if (val_steps_set is not None and 0 in val_steps_set) or (
        val_steps_set is None and rd.should_snapshot(0, interval=snapshot_interval)
    ):
        _log_val(0)
    _on_checkpoint_step(0, frozen=False)

    model.train()
    examples_seen = 0
    hardening_record: dict[str, Any] | None = None
    step_records: list[dict[str, Any]] = []

    for step_idx, batch_ids in enumerate(batches, start=1):
        selected = [by_id[eid] for eid in batch_ids]
        if not selected:
            raise ValueError(f'step {step_idx} has an empty batch')
        denominator = sum(sum(int(v) != -100 for v in r['labels'][1:]) for r in selected)
        if denominator <= 0:
            raise ValueError(f'step {step_idx} has no supervised completion tokens')

        if plan_steps is not None:
            pstep = plan_steps[step_idx - 1]
            expected_frozen = pstep['masks'] == 'frozen-binary'
            if frozen != expected_frozen:
                raise ValueError(f'step {step_idx}: frozen={frozen} disagrees with plan masks={pstep["masks"]!r}')
            expected_mult = float(pstep['lr_multiplier'])
            if abs(optimizer.param_groups[0]['lr'] - resolved_read_lr * expected_mult) > 1e-12 * max(1.0, resolved_read_lr):
                raise ValueError(f'step {step_idx}: read LR disagrees with plan lr_multiplier={expected_mult}')

        optimizer.zero_grad(set_to_none=True)
        boundaries = cache.begin_step(step_idx, len(selected))
        if plan_steps is not None and 'mask_groups' in plan_steps[step_idx - 1]:
            expected_groups = [tuple(g) for g in plan_steps[step_idx - 1]['mask_groups']]
            if boundaries != expected_groups:
                raise ValueError(f'step {step_idx}: mask boundaries {boundaries} != plan {expected_groups}')

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
        dyn_row['mask_group_log'] = cache.step_logs[-1]
        rd.append_jsonl(steps_log_path, dyn_row, strict_monotonic_step=True)
        step_records.append(dyn_row)

        if tracker is not None:
            settle_rec = tracker.observe(step_idx, la_map)
            if settle_rec is not None:
                rd.append_jsonl(settle_log_path, settle_rec)
                if 'logs/settle.jsonl' not in sealed_files:
                    sealed_files.append('logs/settle.jsonl')

        if (snap_steps_set is not None and step_idx in snap_steps_set) or (
            snap_steps_set is None and rd.should_snapshot(step_idx, interval=snapshot_interval)
        ):
            rd.save_log_alpha_snapshot(out_path / 'masks' / 'trace' / f'{step_idx}.npz', la_map, step=step_idx)
        if (val_steps_set is not None and step_idx in val_steps_set) or (
            val_steps_set is None and rd.should_snapshot(step_idx, interval=snapshot_interval)
        ):
            _log_val(step_idx)

        # Save checkpoint BEFORE hardening so checkpoint 960 holds continuous log_alpha
        _on_checkpoint_step(step_idx, frozen=frozen)

        # Hardening / freeze after freeze_step (D5: 960 is last continuous-mask step)
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
            if 'hardening.json' not in sealed_files:
                sealed_files.append('hardening.json')

        if after_step is not None:
            after_step(step_idx, dyn_row)

    base_after = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
    if base_after != base_before:
        raise ValueError('Training modified a frozen base parameter.')

    return {
        'successful_steps': total_steps,
        'examples_seen': examples_seen,
        'rank': resolved_rank,
        'frozen': frozen,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'base_sha256_before': base_before,
        'base_sha256_after': base_after,
        'checkpoint_steps': sorted(s for s in ckpt_set if s <= total_steps),
        'hardening': hardening_record,
        'mask_group_logs': cache.step_logs,
        'sealed_files': sealed_files,
        'final_step_record': step_records[-1],
    }

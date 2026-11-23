"""Sparse rewiring ticket 06: training-dynamics logs and settle diagnostics (D5, D20).

Pure, side-effect-free functions that turn gate and read-factor state into:
- Per-step rows for ``logs/steps.jsonl`` (gate counts, per-layer nonzero counts, gauge-invariant
  read-change norms over all neurons and over the currently nonzero survivor subset).
- Every-16-step FP16 ``log_alpha`` snapshots for ``masks/trace/<step>.npz``.
- Every-64-step settle diagnostics from step 256 for ``logs/settle.jsonl`` (logged only; D5).
- Checkpoint-to-checkpoint mask churn (Jaccard, switched on, switched off).

Zero-signal safety: at step 0 (all read changes zero) or when survivor sets are empty, all norms,
medians, ratios, and Jaccard overlaps are well-defined finite numbers (never NaN or divide-by-zero).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_eval import attribution_units as au
from pilot_eval import hard_concrete as hc

SNAPSHOT_INTERVAL: int = 16
SETTLE_START_STEP: int = 256
SETTLE_INTERVAL: int = 64
SETTLE_JACCARD_REF: float = 0.99
SETTLE_REL_CHANGE_REF: float = 0.02
STEP_LOG_SCHEMA: str = 'sparse-rewiring-step-v1'
SETTLE_LOG_SCHEMA: str = 'sparse-rewiring-settle-v1'


def _stack_layers(
    values: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    *,
    attr: str | Sequence[str] | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Normalize per-layer 1D tensors into ``(sorted_layer_ids, stacked [L, W] float32 CPU tensor)``."""
    by_layer = hc._normalize_layer_tensors(values, attr=attr)
    if not by_layer:
        raise ValueError('expected at least one layer')
    layer_ids = sorted(by_layer)
    shapes = {tuple(by_layer[l].shape) for l in layer_ids}
    if len(shapes) != 1:
        raise ValueError(f'inconsistent layer widths: {sorted(shapes)}')
    stacked = torch.stack([by_layer[l].detach().cpu().float() for l in layer_ids], dim=0)
    return layer_ids, stacked


def _norm_summary(values: torch.Tensor) -> dict[str, Any]:
    """Return finite ``{count, sum, median, max}`` for a 1D norm tensor; empty -> zeros."""
    flat = values.detach().cpu().float().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return {'count': 0, 'sum': 0.0, 'median': 0.0, 'max': 0.0}
    return {
        'count': n,
        'sum': float(flat.sum().item()),
        'median': float(flat.median().item()),
        'max': float(flat.max().item()),
    }


def step_dynamics_row(
    *,
    step: int,
    log_alpha_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    norms_gate_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    norms_up_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    frozen: bool = False,
    ce_loss: float | None = None,
    l0_loss: float | None = None,
    total_loss: float | None = None,
    lr_reads: float | None = None,
    lr_masks: float | None = None,
    grad_norm_reads: float | None = None,
    grad_norm_masks: float | None = None,
    examples_seen: int | None = None,
    beta: float = hc.BETA,
    gamma: float = hc.GAMMA,
    zeta: float = hc.ZETA,
) -> dict[str, Any]:
    """Build one per-step dynamics log row without mutating inputs or RNG state."""
    la_layers, la_stack = _stack_layers(log_alpha_by_layer, attr=hc.LOG_ALPHA_ATTRS)
    gate_layers, gate_stack = _stack_layers(norms_gate_by_layer)
    up_layers, up_stack = _stack_layers(norms_up_by_layer)
    if la_layers != gate_layers or la_layers != up_layers:
        raise ValueError('layer keys must match across log_alpha, gate norms, and up norms')
    if gate_stack.shape != la_stack.shape or up_stack.shape != la_stack.shape:
        raise ValueError(
            f'norm shapes {tuple(gate_stack.shape)}, {tuple(up_stack.shape)} must match log_alpha {tuple(la_stack.shape)}'
        )

    counts = hc.count_gates(
        {l: la_stack[i] for i, l in enumerate(la_layers)},
        frozen=frozen,
        beta=beta,
        gamma=gamma,
        zeta=zeta,
    )
    z_det = hc.deterministic_gate(la_stack, gamma=gamma, zeta=zeta)
    surv_mask = z_det > 0.0

    row: dict[str, Any] = {
        'schema': STEP_LOG_SCHEMA,
        'step': int(step),
        'frozen': bool(frozen),
        'ce_loss': float(ce_loss) if ce_loss is not None else None,
        'l0_loss': float(l0_loss) if l0_loss is not None else None,
        'total_loss': float(total_loss) if total_loss is not None else None,
        'lr_reads': float(lr_reads) if lr_reads is not None else None,
        'lr_masks': float(lr_masks) if lr_masks is not None else None,
        'grad_norm_reads': float(grad_norm_reads) if grad_norm_reads is not None else None,
        'grad_norm_masks': float(grad_norm_masks) if grad_norm_masks is not None else None,
        'examples_seen': int(examples_seen) if examples_seen is not None else None,
        'expected_active': counts['total']['expected_active'],
        'nonzero_z_det': counts['total']['nonzero_z_det'],
        'binary_survivors': counts['total']['binary_survivors'],
        'per_layer_nonzero': [pl['nonzero_z_det'] for pl in counts['per_layer']],
        'per_layer_expected_active': [pl['expected_active'] for pl in counts['per_layer']],
        'norms': {
            'gate_all': _norm_summary(gate_stack),
            'up_all': _norm_summary(up_stack),
            'gate_survivors': _norm_summary(gate_stack[surv_mask]),
            'up_survivors': _norm_summary(up_stack[surv_mask]),
        },
    }
    return row


def should_snapshot(step: int, *, interval: int = SNAPSHOT_INTERVAL) -> bool:
    """True on step 0 and every ``interval`` steps (default 16)."""
    if interval < 1:
        raise ValueError(f'interval must be >= 1, got {interval}')
    return int(step) >= 0 and int(step) % int(interval) == 0


def save_log_alpha_snapshot(
    path: str | Path,
    log_alpha_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    *,
    step: int,
    gamma: float = hc.GAMMA,
    zeta: float = hc.ZETA,
) -> Path:
    """Save stacked ``[L, W]`` FP16 ``log_alpha`` AND packed exact FP32 ``z_det > 0`` hard mask (~31 KB).

    Saving the exact packed FP32 hard mask prevents FP16 rounding near ``-ln(11)`` (~0.002 spacing)
    from flipping boundary neurons across ``z_det > 0`` when recomputing counts or churn.
    """
    layer_ids, stacked = _stack_layers(log_alpha_by_layer, attr=hc.LOG_ALPHA_ATTRS)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr_fp16 = stacked.numpy().astype(np.float16)
    hard_mask = hc.hard_survivor_mask(stacked, gamma=gamma, zeta=zeta).numpy().astype(bool)
    packed = np.packbits(hard_mask.ravel())
    np.savez(
        out_path,
        step=np.int64(step),
        layers=np.asarray(layer_ids, dtype=np.int64),
        log_alpha=arr_fp16,
        hard_mask_bits=packed,
        hard_mask_shape=np.asarray(hard_mask.shape, dtype=np.int64),
    )
    return out_path


def load_log_alpha_snapshot(path: str | Path) -> dict[str, Any]:
    """Load an FP16 ``log_alpha`` snapshot and unpack its exact FP32 ``hard_mask`` without pickle."""
    with np.load(Path(path), allow_pickle=False) as npz:
        la = np.asarray(npz['log_alpha'], dtype=np.float16)
        if 'hard_mask_bits' in npz.files and 'hard_mask_shape' in npz.files:
            shape = tuple(int(x) for x in np.asarray(npz['hard_mask_shape'], dtype=np.int64).tolist())
            n_bits = int(np.prod(shape))
            hard_mask = np.unpackbits(np.asarray(npz['hard_mask_bits'], dtype=np.uint8), count=n_bits).astype(bool).reshape(shape)
        else:
            hard_mask = hc.hard_survivor_mask(torch.as_tensor(la, dtype=torch.float32)).numpy().astype(bool)
        return {
            'step': int(npz['step']),
            'layers': np.asarray(npz['layers'], dtype=np.int64).tolist(),
            'log_alpha': la,
            'hard_mask': hard_mask,
        }


def recompute_counts_from_snapshot(
    snapshot_or_log_alpha: Mapping[str, Any] | np.ndarray | torch.Tensor,
    *,
    hard_mask: np.ndarray | None = None,
    frozen: bool = False,
    beta: float = hc.BETA,
    gamma: float = hc.GAMMA,
    zeta: float = hc.ZETA,
) -> dict[str, Any]:
    """Recompute the three gate counts from a saved snapshot (using exact packed ``hard_mask`` when present)."""
    if isinstance(snapshot_or_log_alpha, Mapping):
        la_arr = snapshot_or_log_alpha['log_alpha']
        if hard_mask is None and 'hard_mask' in snapshot_or_log_alpha:
            hard_mask = np.asarray(snapshot_or_log_alpha['hard_mask'], dtype=bool)
    else:
        la_arr = snapshot_or_log_alpha
    t = torch.as_tensor(la_arr, dtype=torch.float32)
    if t.ndim != 2:
        raise ValueError(f'expected 2D [L, W] log_alpha array, got shape {tuple(t.shape)}')
    counts = hc.count_gates([t[i] for i in range(t.shape[0])], frozen=frozen, beta=beta, gamma=gamma, zeta=zeta)
    if hard_mask is not None:
        hm = np.asarray(hard_mask, dtype=bool)
        if hm.shape != tuple(t.shape):
            raise ValueError(f'hard_mask shape {hm.shape} != log_alpha shape {tuple(t.shape)}')
        for i, pl in enumerate(counts['per_layer']):
            nz = int(hm[i].sum())
            pl['nonzero_z_det'] = nz
            if frozen:
                pl['binary_survivors'] = nz
        total_nz = int(hm.sum())
        counts['total']['nonzero_z_det'] = total_nz
        if frozen:
            counts['total']['binary_survivors'] = total_nz
    return counts


def _to_bool_array(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    return arr.astype(bool, copy=False)


def mask_churn(prev_mask: np.ndarray | torch.Tensor, curr_mask: np.ndarray | torch.Tensor) -> dict[str, Any]:
    """Compute Jaccard overlap and switched-on / switched-off counts between two boolean masks.

    When both masks are empty (``union == 0``), ``jaccard`` is defined as ``1.0`` (zero-signal safe).
    """
    p = _to_bool_array(prev_mask)
    c = _to_bool_array(curr_mask)
    if p.shape != c.shape:
        raise ValueError(f'mask shape mismatch: {p.shape} vs {c.shape}')
    inter = int(np.logical_and(p, c).sum())
    union = int(np.logical_or(p, c).sum())
    switched_on = int(np.logical_and(c, np.logical_not(p)).sum())
    switched_off = int(np.logical_and(p, np.logical_not(c)).sum())
    jaccard = float(inter / union) if union > 0 else 1.0
    return {
        'prev_count': int(p.sum()),
        'curr_count': int(c.sum()),
        'intersection': inter,
        'union': union,
        'switched_on': switched_on,
        'switched_off': switched_off,
        'jaccard': jaccard,
    }


def should_check_settle(
    step: int,
    *,
    start: int = SETTLE_START_STEP,
    interval: int = SETTLE_INTERVAL,
) -> bool:
    """True at ``start`` (default 256) and every ``interval`` (default 64) steps thereafter."""
    if interval < 1:
        raise ValueError(f'interval must be >= 1, got {interval}')
    s = int(step)
    return s >= int(start) and (s - int(start)) % int(interval) == 0


def settle_diagnostic(
    *,
    step: int,
    prev_step: int,
    prev_log_alpha: np.ndarray | torch.Tensor,
    curr_log_alpha: np.ndarray | torch.Tensor,
    freeze_step: int | None = None,
    jaccard_ref: float = SETTLE_JACCARD_REF,
    rel_change_ref: float = SETTLE_REL_CHANGE_REF,
    beta: float = hc.BETA,
    gamma: float = hc.GAMMA,
    zeta: float = hc.ZETA,
) -> dict[str, Any]:
    """Compute D5 settle diagnostic between ``prev_step`` and ``step`` (logged only, never stops training)."""
    prev_t = torch.as_tensor(prev_log_alpha, dtype=torch.float32)
    curr_t = torch.as_tensor(curr_log_alpha, dtype=torch.float32)
    if prev_t.shape != curr_t.shape:
        raise ValueError(f'log_alpha shape mismatch: {tuple(prev_t.shape)} vs {tuple(curr_t.shape)}')

    prev_hard = hc.hard_survivor_mask(prev_t, gamma=gamma, zeta=zeta)
    curr_hard = hc.hard_survivor_mask(curr_t, gamma=gamma, zeta=zeta)
    churn = mask_churn(prev_hard, curr_hard)

    prev_exp = float(hc.expected_active(prev_t, beta=beta, gamma=gamma, zeta=zeta).item())
    curr_exp = float(hc.expected_active(curr_t, beta=beta, gamma=gamma, zeta=zeta).item())
    if prev_exp > 0.0:
        rel_change: float | None = abs(curr_exp - prev_exp) / prev_exp
    elif curr_exp == 0.0:
        rel_change = 0.0
    else:
        rel_change = None

    meets = bool(
        churn['jaccard'] >= float(jaccard_ref)
        and rel_change is not None
        and rel_change < float(rel_change_ref)
    )
    return {
        'schema': SETTLE_LOG_SCHEMA,
        'step': int(step),
        'prev_step': int(prev_step),
        'jaccard': churn['jaccard'],
        'switched_on': churn['switched_on'],
        'switched_off': churn['switched_off'],
        'prev_nonzero': churn['prev_count'],
        'curr_nonzero': churn['curr_count'],
        'prev_expected_active': prev_exp,
        'curr_expected_active': curr_exp,
        'expected_active_rel_change': rel_change,
        'jaccard_ref': float(jaccard_ref),
        'rel_change_ref': float(rel_change_ref),
        'meets_reference': meets,
        'post_freeze': bool(freeze_step is not None and int(step) > int(freeze_step)),
        'diagnostic_only': True,
    }


class SettleTracker:
    """Read-only tracker for D5 settle diagnostics (first check at 256, then every 64 steps).

    Records a detached CPU copy of ``log_alpha`` every ``interval`` steps starting at
    ``start - interval`` (192 by default), so step 256 compares against 192, 320 against 256, etc.
    Never modifies model parameters, gradients, or RNG state.
    """

    def __init__(
        self,
        *,
        start: int = SETTLE_START_STEP,
        interval: int = SETTLE_INTERVAL,
        freeze_step: int | None = 960,
        jaccard_ref: float = SETTLE_JACCARD_REF,
        rel_change_ref: float = SETTLE_REL_CHANGE_REF,
        beta: float = hc.BETA,
        gamma: float = hc.GAMMA,
        zeta: float = hc.ZETA,
    ) -> None:
        if interval < 1:
            raise ValueError(f'interval must be >= 1, got {interval}')
        if start < interval:
            raise ValueError(f'start ({start}) must be >= interval ({interval})')
        self.start = int(start)
        self.interval = int(interval)
        self.freeze_step = int(freeze_step) if freeze_step is not None else None
        self.jaccard_ref = float(jaccard_ref)
        self.rel_change_ref = float(rel_change_ref)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self._baseline_step: int | None = None
        self._baseline_log_alpha: torch.Tensor | None = None
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def observe(
        self,
        step: int,
        log_alpha_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    ) -> dict[str, Any] | None:
        """Observe ``step``; return a settle diagnostic dict on check steps, else ``None``."""
        s = int(step)
        anchor_step = self.start - self.interval
        if s < anchor_step or (s - anchor_step) % self.interval != 0:
            return None
        _, curr_stack = _stack_layers(log_alpha_by_layer, attr=hc.LOG_ALPHA_ATTRS)
        record: dict[str, Any] | None = None
        if should_check_settle(s, start=self.start, interval=self.interval):
            if self._baseline_step is None or self._baseline_log_alpha is None:
                raise RuntimeError(f'SettleTracker reached check step {s} without baseline at {s - self.interval}')
            if s - self._baseline_step != self.interval:
                raise RuntimeError(
                    f'SettleTracker expected baseline at {s - self.interval}, found {self._baseline_step}'
                )
            record = settle_diagnostic(
                step=s,
                prev_step=self._baseline_step,
                prev_log_alpha=self._baseline_log_alpha,
                curr_log_alpha=curr_stack,
                freeze_step=self.freeze_step,
                jaccard_ref=self.jaccard_ref,
                rel_change_ref=self.rel_change_ref,
                beta=self.beta,
                gamma=self.gamma,
                zeta=self.zeta,
            )
            self._records.append(record)
        self._baseline_step = s
        self._baseline_log_alpha = curr_stack.clone()
        return record


def append_jsonl(
    path: str | Path,
    row: Mapping[str, Any],
    *,
    strict_monotonic_step: bool = False,
    fsync: bool = False,
) -> None:
    """Append one JSON-encoded row (NumPy-safe via ``attribution_units._json_default``) to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if strict_monotonic_step and 'step' in row and out_path.is_file():
        existing = read_jsonl(out_path)
        if existing and 'step' in existing[-1] and int(row['step']) <= int(existing[-1]['step']):
            raise ValueError(
                f'refusing non-monotonic append to {out_path}: step {row["step"]} <= last logged {existing[-1]["step"]}'
            )
    line = json.dumps(dict(row), sort_keys=True, default=au._json_default) + '\n'
    with out_path.open('a', encoding='utf-8') as stream:
        stream.write(line)
        if fsync:
            stream.flush()
            os.fsync(stream.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]

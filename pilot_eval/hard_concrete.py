"""Sparse rewiring ticket 03: hard-concrete gate, shared mask cache (D14) and survivor export.

Clean-room implementation of the stretched hard-concrete L0 gate from Louizos, Welling & Kingma (2018)
("Learning Sparse Neural Networks through L_0 Regularization", ICLR 2018) and the sparse-rewiring
parameterisation described in ``maxsloef/owl-rewired-neurons`` at ``a6dc314`` (README.md / RANK0.md).

Key contracts:
- Gate hyperparameters: ``beta = 2/3``, ``gamma = -0.1``, ``zeta = 1.1``.
- Freeze rule: neurons with ``z_det > 0`` (equivalently ``log_alpha > -ln(11)``) become binary 1
  (``log_alpha = +20``), all others become binary 0 (``log_alpha = -20``), with ``requires_grad = False``.
- Shared mask noise (D14): one uniform ``u`` per layer is sampled per group of 4 consecutive examples
  from a dedicated generator on the model device, cached, and reused across all 4 single-example passes
  and across gradient-checkpointing recomputes. ``z`` is recomputed from the cached ``u`` and live
  ``log_alpha`` on every pass so gradients reach ``log_alpha``.
- Survivor export: zero-based ``(layer, neuron)`` with ``log_alpha``, ``z_det``, binary state,
  gauge-invariant gate/up row norms, and three counts (expected-active, nonzero ``z_det``, binary)
  total and per layer; sealed with ``attribution_units.seal_run`` / ``_encode``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pilot_eval import attribution_units as au

BETA: float = 2.0 / 3.0
GAMMA: float = -0.1
ZETA: float = 1.1
LOG_ALPHA_INIT: float = 1.0
FREEZE_LOG_ALPHA_ON: float = 20.0
FREEZE_LOG_ALPHA_OFF: float = -20.0
FREEZE_LOG_ALPHA_THRESHOLD: float = -math.log(-ZETA / GAMMA)  # -ln(11) ~= -2.3978952728
U_EPS: float = 1e-6
DEFAULT_GROUP_SIZE: int = 4
SURVIVOR_EXPORT_SCHEMA: str = 'sparse-rewiring-survivors-v1'


def _validate_gate_params(beta: float, gamma: float, zeta: float) -> None:
    if beta <= 0.0:
        raise ValueError(f'beta must be > 0, got {beta}')
    if not (gamma < 0.0 and zeta > 1.0):
        raise ValueError(f'stretched hard-concrete requires gamma < 0 and zeta > 1, got ({gamma}, {zeta})')


def expected_active_probs(
    log_alpha: torch.Tensor,
    *,
    beta: float = BETA,
    gamma: float = GAMMA,
    zeta: float = ZETA,
) -> torch.Tensor:
    """Per-neuron probability P(z > 0) = sigmoid(log_alpha - beta * ln(-gamma / zeta))."""
    _validate_gate_params(beta, gamma, zeta)
    offset = beta * math.log(-gamma / zeta)
    return torch.sigmoid(log_alpha - offset)


def expected_active(
    log_alpha: torch.Tensor,
    *,
    beta: float = BETA,
    gamma: float = GAMMA,
    zeta: float = ZETA,
) -> torch.Tensor:
    """Scalar L0 penalty term: sum_i P(z_i > 0)."""
    return expected_active_probs(log_alpha, beta=beta, gamma=gamma, zeta=zeta).sum()


def deterministic_gate(
    log_alpha: torch.Tensor,
    *,
    gamma: float = GAMMA,
    zeta: float = ZETA,
) -> torch.Tensor:
    """Deterministic evaluation gate: clip(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)."""
    _validate_gate_params(BETA, gamma, zeta)
    s = torch.sigmoid(log_alpha)
    return torch.clamp(s * (zeta - gamma) + gamma, 0.0, 1.0)


def sample_gate_from_u(
    log_alpha: torch.Tensor,
    u: torch.Tensor,
    *,
    beta: float = BETA,
    gamma: float = GAMMA,
    zeta: float = ZETA,
    eps: float = U_EPS,
) -> torch.Tensor:
    """Compute stochastic hard-concrete gate from a pre-drawn uniform tensor ``u``.

    Recomputing ``z`` from a cached ``u`` and live ``log_alpha`` ensures gradients flow into ``log_alpha``
    on every microbatch pass and under gradient-checkpointing recomputation (D14).
    """
    _validate_gate_params(beta, gamma, zeta)
    if u.shape != log_alpha.shape:
        raise ValueError(f'cached u shape {tuple(u.shape)} does not match log_alpha shape {tuple(log_alpha.shape)}')
    if u.device != log_alpha.device:
        raise ValueError(f'cached u device {u.device} does not match log_alpha device {log_alpha.device}')
    u_safe = u.clamp(eps, 1.0 - eps)
    logit_u = torch.log(u_safe) - torch.log1p(-u_safe)
    s = torch.sigmoid((logit_u + log_alpha) / beta)
    return torch.clamp(s * (zeta - gamma) + gamma, 0.0, 1.0)


def sample_gate(
    log_alpha: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    beta: float = BETA,
    gamma: float = GAMMA,
    zeta: float = ZETA,
    eps: float = U_EPS,
) -> torch.Tensor:
    """Draw a fresh uniform ``u`` on ``log_alpha.device`` and return the stochastic gate."""
    u = torch.empty_like(log_alpha).uniform_(eps, 1.0 - eps, generator=generator)
    return sample_gate_from_u(log_alpha, u, beta=beta, gamma=gamma, zeta=zeta, eps=eps)


def hard_survivor_mask(
    log_alpha: torch.Tensor,
    *,
    gamma: float = GAMMA,
    zeta: float = ZETA,
) -> torch.Tensor:
    """Boolean mask of neurons with positive deterministic gate (z_det > 0, i.e. log_alpha > -ln(11))."""
    return deterministic_gate(log_alpha, gamma=gamma, zeta=zeta) > 0.0


def freeze_log_alpha(
    log_alpha: torch.Tensor,
    *,
    gamma: float = GAMMA,
    zeta: float = ZETA,
    on_value: float = FREEZE_LOG_ALPHA_ON,
    off_value: float = FREEZE_LOG_ALPHA_OFF,
) -> torch.Tensor:
    """Harden ``log_alpha`` in place to binary survivors (+20 / -20) and disable its gradients."""
    with torch.no_grad():
        survivors = hard_survivor_mask(log_alpha, gamma=gamma, zeta=zeta)
        hardened = torch.where(
            survivors,
            torch.full_like(log_alpha, float(on_value)),
            torch.full_like(log_alpha, float(off_value)),
        )
        log_alpha.copy_(hardened)
    log_alpha.requires_grad_(False)
    log_alpha.grad = None
    return survivors


def group_boundaries(num_examples: int, group_size: int = DEFAULT_GROUP_SIZE) -> list[tuple[int, int]]:
    """Partition ``num_examples`` into consecutive groups of size ``group_size`` (last group may be shorter)."""
    if num_examples < 0:
        raise ValueError(f'num_examples must be >= 0, got {num_examples}')
    if group_size < 1:
        raise ValueError(f'group_size must be >= 1, got {group_size}')
    return [(start, min(start + group_size, num_examples)) for start in range(0, num_examples, group_size)]


class SharedMaskCache:
    """Per-group uniform ``u`` cache implementing D14 (one shared mask draw per 4 consecutive examples).

    Usage per training step:
        boundaries = cache.begin_step(step, num_examples)
        for start, end in boundaries:
            cache.draw_group(layer_shapes, dtype=torch.float32)
            for idx in range(start, end):
                # forward + backward on example idx; each layer calls cache.gate(layer_idx, log_alpha)
        cache.clear()
    """

    def __init__(
        self,
        *,
        seed: int = 1042,
        group_size: int = DEFAULT_GROUP_SIZE,
        device: str | torch.device = 'cpu',
        eps: float = U_EPS,
        beta: float = BETA,
        gamma: float = GAMMA,
        zeta: float = ZETA,
    ) -> None:
        _validate_gate_params(beta, gamma, zeta)
        if group_size < 1:
            raise ValueError(f'group_size must be >= 1, got {group_size}')
        self.seed = int(seed)
        self.group_size = int(group_size)
        # Normalise once (e.g. 'cuda' -> torch.device('cuda:0')) so comparisons against real tensors match.
        self.device = torch.empty(0, device=device).device
        self.eps = float(eps)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.seed)
        self._cached_u: dict[int, torch.Tensor] = {}
        self._frozen: bool = False
        self._step_logs: list[dict[str, Any]] = []

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def step_logs(self) -> list[dict[str, Any]]:
        return list(self._step_logs)

    def begin_step(self, step: int, num_examples: int) -> list[tuple[int, int]]:
        """Record group boundaries for ``step`` and clear any previous group's cached ``u``."""
        boundaries = group_boundaries(num_examples, self.group_size)
        self._cached_u.clear()
        self._step_logs.append(
            {
                'step': int(step),
                'seed': self.seed,
                'group_size': self.group_size,
                'num_examples': int(num_examples),
                'num_groups': len(boundaries),
                'boundaries': [[s, e] for s, e in boundaries],
            }
        )
        return boundaries

    def draw_group(
        self,
        layer_shapes: Mapping[int, tuple[int, ...]] | Sequence[tuple[int, ...]],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[int, torch.Tensor]:
        """Sample and cache one uniform ``u`` per layer on ``self.device`` for the current group."""
        if isinstance(layer_shapes, Mapping):
            items = sorted((int(k), tuple(v)) for k, v in layer_shapes.items())
        else:
            items = [(idx, tuple(shape)) for idx, shape in enumerate(layer_shapes)]
        self._cached_u = {}
        for layer_idx, shape in items:
            u = torch.empty(shape, dtype=dtype, device=self.device).uniform_(
                self.eps, 1.0 - self.eps, generator=self._generator
            )
            self._cached_u[layer_idx] = u
        return dict(self._cached_u)

    def set_group_u(self, u_by_layer: Mapping[int, torch.Tensor]) -> None:
        """Install pre-drawn ``u`` tensors (e.g. for equivalence tests); verifies device placement."""
        installed: dict[int, torch.Tensor] = {}
        for layer_idx, u in u_by_layer.items():
            if u.device != self.device:
                raise ValueError(f'layer {layer_idx} u is on {u.device}, expected {self.device}')
            installed[int(layer_idx)] = u.detach()
        self._cached_u = installed

    def cached_u(self, layer: int) -> torch.Tensor:
        if layer not in self._cached_u:
            raise RuntimeError(f'SharedMaskCache has no cached u for layer {layer}; call draw_group() first')
        return self._cached_u[layer]

    def gate(
        self,
        layer: int,
        log_alpha: torch.Tensor,
        *,
        training: bool = True,
        frozen: bool | None = None,
    ) -> torch.Tensor:
        """Return the gate vector for ``layer`` using the cached ``u`` (training) or deterministic gate."""
        is_frozen = self._frozen if frozen is None else bool(frozen)
        if is_frozen or not training:
            return deterministic_gate(log_alpha, gamma=self.gamma, zeta=self.zeta)
        u = self.cached_u(layer)
        return sample_gate_from_u(
            log_alpha,
            u,
            beta=self.beta,
            gamma=self.gamma,
            zeta=self.zeta,
            eps=self.eps,
        )

    def clear(self) -> None:
        """Drop cached ``u`` tensors at the end of a group or step."""
        self._cached_u.clear()

    def freeze(self) -> None:
        """Switch the cache into frozen deterministic binary mode."""
        self._frozen = True
        self._cached_u.clear()


def row_change_norms(
    *,
    rank: int,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gauge-invariant per-neuron row-change L2 norm ``||Delta w_i||_2``.

    - rank 0: ``||w_i||_2`` (``w`` of shape ``[n, d]``)
    - rank 1: ``|b_i| * ||a||_2`` (``a`` of shape ``[d]`` or ``[1, d]``, ``b`` of shape ``[n]`` or ``[n, 1]``)
    - rank k > 1: ``||b_i @ A||_2`` (``a`` of shape ``[r, d]``, ``b`` of shape ``[n, r]``)

    If ``z`` (shape ``[n]``) is supplied, scales each row norm by ``z_i``.
    """
    if rank < 0:
        raise ValueError(f'rank must be >= 0, got {rank}')
    if rank == 0:
        if w is None or w.ndim != 2:
            raise ValueError('rank 0 requires a 2D weight tensor w of shape [n, d]')
        norms = torch.linalg.vector_norm(w.detach().float(), dim=-1)
    elif rank == 1:
        if a is None or b is None:
            raise ValueError('rank 1 requires both a and b')
        a_vec = a.detach().float().reshape(-1)
        b_vec = b.detach().float().reshape(-1)
        norms = b_vec.abs() * torch.linalg.vector_norm(a_vec)
    else:
        if a is None or b is None:
            raise ValueError(f'rank {rank} requires both a [r, d] and b [n, r]')
        a_mat = a.detach().float()
        b_mat = b.detach().float()
        if a_mat.ndim != 2 or b_mat.ndim != 2 or b_mat.shape[1] != a_mat.shape[0]:
            raise ValueError(f'rank {rank} expects b [n, r] and a [r, d], got {tuple(b_mat.shape)} and {tuple(a_mat.shape)}')
        norms = torch.linalg.vector_norm(b_mat @ a_mat, dim=-1)
    if z is not None:
        z_vec = z.detach().float().reshape(-1)
        if z_vec.shape != norms.shape:
            raise ValueError(f'z shape {tuple(z_vec.shape)} does not match norms shape {tuple(norms.shape)}')
        norms = norms * z_vec
    return norms


def _normalize_layer_tensors(
    values: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    *,
    attr: str | Sequence[str] | None = None,
) -> dict[int, torch.Tensor]:
    if isinstance(values, Mapping):
        raw = {int(k): v for k, v in values.items()}
    else:
        raw = {idx: v for idx, v in enumerate(values)}
    attrs = (attr,) if isinstance(attr, str) else tuple(attr or ())
    out: dict[int, torch.Tensor] = {}
    for layer_idx in sorted(raw):
        item = raw[layer_idx]
        if attrs and not isinstance(item, torch.Tensor):
            for name in attrs:
                if hasattr(item, name):
                    item = getattr(item, name)
                    break
            else:
                raise AttributeError(f'layer {layer_idx}: {type(item).__name__} has none of attributes {attrs}')
        if not isinstance(item, torch.Tensor):
            raise TypeError(f'layer {layer_idx}: expected torch.Tensor, got {type(item).__name__}')
        out[layer_idx] = item.detach().float().reshape(-1)
    return out


LOG_ALPHA_ATTRS: tuple[str, str] = ('rewire_log_alpha', 'log_alpha')


def count_gates(
    log_alpha_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    *,
    frozen: bool = False,
    beta: float = BETA,
    gamma: float = GAMMA,
    zeta: float = ZETA,
) -> dict[str, Any]:
    """Compute the three gate counts (expected-active, nonzero z_det, binary survivors) per layer and total."""
    layers = _normalize_layer_tensors(log_alpha_by_layer, attr=LOG_ALPHA_ATTRS)
    per_layer: list[dict[str, Any]] = []
    total_expected = 0.0
    total_nonzero = 0
    total_binary: int | None = 0 if frozen else None

    for layer_idx, la in layers.items():
        exp = float(expected_active(la, beta=beta, gamma=gamma, zeta=zeta).item())
        z_det = deterministic_gate(la, gamma=gamma, zeta=zeta)
        nz = int((z_det > 0.0).sum().item())
        if frozen:
            is_binary = bool(((z_det == 0.0) | (z_det == 1.0)).all().item())
            if not is_binary:
                raise ValueError(f'layer {layer_idx} marked frozen=True but z_det is not strictly binary 0/1')
            bin_count: int | None = int((z_det == 1.0).sum().item())
            assert total_binary is not None
            total_binary += bin_count
        else:
            bin_count = None
        total_expected += exp
        total_nonzero += nz
        per_layer.append(
            {
                'layer': layer_idx,
                'num_neurons': int(la.numel()),
                'expected_active': exp,
                'nonzero_z_det': nz,
                'binary_survivors': bin_count,
            }
        )

    return {
        'frozen': bool(frozen),
        'num_layers': len(per_layer),
        'total': {
            'expected_active': total_expected,
            'nonzero_z_det': total_nonzero,
            'binary_survivors': total_binary,
        },
        'per_layer': per_layer,
    }


def export_survivors(
    log_alpha_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | Sequence[Any],
    *,
    frozen: bool = False,
    norms_gate_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | None = None,
    norms_up_by_layer: Sequence[torch.Tensor] | Mapping[int, torch.Tensor] | None = None,
    step: int | None = None,
    meta: Mapping[str, Any] | None = None,
    max_survivor_records: int | None = None,
    beta: float = BETA,
    gamma: float = GAMMA,
    zeta: float = ZETA,
) -> dict[str, Any]:
    """Build the JSON-serialisable survivor export with zero-based ``(layer, neuron)`` records.

    Compatible with ``rewire_attribution.py`` (reads ``export['survivors']`` as ``[{'layer', 'neuron', ...}]``).
    When ``not frozen`` and ``max_survivor_records`` is set and exceeded (e.g. checkpoint 0 where all
    250,880 neurons are nonzero), per-neuron records are omitted while exact per-layer counts are kept.
    """
    layers = _normalize_layer_tensors(log_alpha_by_layer, attr=LOG_ALPHA_ATTRS)
    gate_norms = _normalize_layer_tensors(norms_gate_by_layer) if norms_gate_by_layer is not None else None
    up_norms = _normalize_layer_tensors(norms_up_by_layer) if norms_up_by_layer is not None else None
    counts = count_gates(layers, frozen=frozen, beta=beta, gamma=gamma, zeta=zeta)

    omit_records = (
        not frozen
        and max_survivor_records is not None
        and counts['total']['nonzero_z_det'] > int(max_survivor_records)
    )
    survivors: list[dict[str, Any]] = []
    if not omit_records:
        for layer_idx, la in layers.items():
            z_det = deterministic_gate(la, gamma=gamma, zeta=zeta)
            ng = gate_norms[layer_idx] if gate_norms is not None else None
            nu = up_norms[layer_idx] if up_norms is not None else None
            if ng is not None and ng.shape != la.shape:
                raise ValueError(f'layer {layer_idx} gate norms shape {tuple(ng.shape)} != {tuple(la.shape)}')
            if nu is not None and nu.shape != la.shape:
                raise ValueError(f'layer {layer_idx} up norms shape {tuple(nu.shape)} != {tuple(la.shape)}')
            active_indices = torch.nonzero(z_det > 0.0, as_tuple=False).reshape(-1).tolist()
            for neuron_idx in active_indices:
                zd_val = float(z_det[neuron_idx].item())
                survivors.append(
                    {
                        'layer': int(layer_idx),
                        'neuron': int(neuron_idx),
                        'log_alpha': float(la[neuron_idx].item()),
                        'z_det': zd_val,
                        'binary': bool(zd_val == 1.0) if frozen else None,
                        'norm_gate': float(ng[neuron_idx].item()) if ng is not None else None,
                        'norm_up': float(nu[neuron_idx].item()) if nu is not None else None,
                    }
                )

    out: dict[str, Any] = {
        'schema': SURVIVOR_EXPORT_SCHEMA,
        'step': int(step) if step is not None else None,
        'frozen': bool(frozen),
        'gate_params': {
            'beta': float(beta),
            'gamma': float(gamma),
            'zeta': float(zeta),
            'freeze_threshold': float(-math.log(-zeta / gamma)),
        },
        'counts': counts,
        'survivors': survivors,
        'survivors_omitted': omit_records,
        'meta': dict(meta or {}),
    }
    if omit_records:
        out['survivors_omitted_reason'] = (
            f'pre-freeze nonzero count {counts["total"]["nonzero_z_det"]} exceeds cap {int(max_survivor_records)}'
        )
    return out


def seal_survivor_export(
    directory: str | Path,
    export_data: Mapping[str, Any],
    *,
    filename: str = 'survivors.json',
) -> str:
    """Write and seal a survivor export directory using ``attribution_units.seal_run`` / ``_encode``."""
    return au.seal_run(Path(directory), {filename: dict(export_data)})

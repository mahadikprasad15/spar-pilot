"""Sparse rewiring ticket 12 (D22): standalone analysis of each rank. CPU, NumPy only.

Covers the three D22 groups from ``spec.md`` §Standalone analysis:
1. **Selection**: survivor counts, per-layer distribution vs uniform reference, ``log_alpha``
   histograms, settle step (from D5 settle diagnostics and from 16-step gate snapshots, marked
   ``proposed``), and checkpoint stability (Jaccard vs final set, first-on and stayed-on steps).
2. **What survivors learned**: gauge-invariant change size (gate vs up, per-layer ratio), rank-1
   shared direction ``a`` sign-free gauge-invariant cosine with gold LoRA's ``gate_proj`` /
   ``up_proj`` ``lora_A`` (loaded via ``neuron_audit_collector.load_saved_factors``), rank-0
   effective rank (participation ratio ``(sum sigma)^2 / sum sigma^2`` and 90%-energy count, marked
   ``proposed``), and activation-shift summary (signed mean + mean-absolute, survivors vs random
   non-survivor reference).
3. **Behaviour**: validation CE/accuracy summary, cohort150 accuracy/invalid/capped/length summary,
   and per-question 2x2 flip tables vs ``gold-sft`` and ``untuned`` with paired bootstrap
   (``c_attribution_stability.resample_counts``) and exact McNemar p
   (``c_attribution_generation.mcnemar_exact``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import c_attribution_stability as cs
from pilot_eval import neuron_audit_collector as nac

SCHEMA: str = 'sparse-rewiring-standalone-v1'
SETTLE_JACCARD_REF: float = 0.99
SETTLE_REL_CHANGE_REF: float = 0.02
BOOTSTRAP_DRAWS: int = 10_000
BOOTSTRAP_SEED: int = 20261125
RANDOM_COSINE_DRAWS: int = 2_000
RANDOM_COSINE_SEED: int = 20261125


def _jaccard_sets(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _pairs_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f'expected 2D [L, W] boolean mask, got shape {arr.shape}')
    layers, neurons = np.nonzero(arr)
    return {(int(l), int(n)) for l, n in zip(layers.tolist(), neurons.tolist())}


# --- 1. Selection ----------------------------------------------------------------------------------

def per_layer_distribution(survivors: Sequence[Any], num_layers: int) -> dict[str, Any]:
    """Per-layer survivor counts against a uniform-by-layer reference."""
    if num_layers < 1:
        raise ValueError(f'num_layers must be >= 1, got {num_layers}')
    counts = [0] * int(num_layers)
    for item in survivors:
        layer = int(item['layer']) if isinstance(item, Mapping) else int(item[0])
        if not (0 <= layer < num_layers):
            raise ValueError(f'layer {layer} out of range [0, {num_layers})')
        counts[layer] += 1
    total = sum(counts)
    uniform = float(total / num_layers)
    deviations = [c - uniform for c in counts]
    return {
        'num_layers': int(num_layers),
        'total_survivors': total,
        'uniform_reference': uniform,
        'per_layer_counts': counts,
        'max_abs_deviation': float(max((abs(d) for d in deviations), default=0.0)),
    }


def log_alpha_histogram(log_alpha: np.ndarray, *, bins: int = 40, value_range: tuple[float, float] = (-20.0, 20.0)) -> dict[str, Any]:
    """Histogram of ``log_alpha`` values across all neurons."""
    arr = np.asarray(log_alpha, dtype=np.float64).ravel()
    counts, edges = np.histogram(arr, bins=int(bins), range=value_range)
    return {
        'bins': int(bins),
        'edges': edges.tolist(),
        'counts': counts.astype(int).tolist(),
        'min': float(arr.min()) if arr.size else None,
        'max': float(arr.max()) if arr.size else None,
    }


def compute_settle_step(
    *,
    settle_records: Sequence[Mapping[str, Any]] = (),
    snapshot_masks: Mapping[int, np.ndarray] | None = None,
    final_survivors: Sequence[Any] | None = None,
    freeze_step: int = 960,
    jaccard_ref: float = SETTLE_JACCARD_REF,
    rel_change_ref: float = SETTLE_REL_CHANGE_REF,
) -> dict[str, Any]:
    """Compute the two proposed D22 settle-step definitions (marked ``proposed``).

    1. ``from_diagnostics``: earliest D5 settle check (<= ``freeze_step``) after which every check
       up to ``freeze_step`` satisfies ``jaccard >= jaccard_ref`` and
       ``expected_active_rel_change < rel_change_ref``.
    2. ``from_snapshots``: earliest 16-step snapshot (<= ``freeze_step``) after which the hard
       survivor set keeps ``jaccard >= jaccard_ref`` with ``final_survivors`` at every later
       snapshot up to ``freeze_step``.
    """
    # Definition 1: from D5 settle diagnostics
    diag_filtered = sorted(
        (r for r in settle_records if int(r['step']) <= int(freeze_step)),
        key=lambda r: int(r['step']),
    )
    from_diag: int | None = None
    if diag_filtered:
        ok = [
            (
                float(r['jaccard']) >= float(jaccard_ref)
                and r.get('expected_active_rel_change') is not None
                and float(r['expected_active_rel_change']) < float(rel_change_ref)
            )
            for r in diag_filtered
        ]
        # Earliest index i such that ok[j] is True for all j >= i
        for i in range(len(ok)):
            if all(ok[i:]):
                from_diag = int(diag_filtered[i]['step'])
                break

    # Definition 2: from 16-step snapshots vs final binary set
    from_snaps: int | None = None
    snapshot_curve: list[dict[str, Any]] = []
    if snapshot_masks is not None and final_survivors is not None:
        final_set = {
            (int(x['layer']), int(x['neuron'])) if isinstance(x, Mapping) else (int(x[0]), int(x[1]))
            for x in final_survivors
        }
        steps = sorted(int(s) for s in snapshot_masks if int(s) <= int(freeze_step))
        jaccards: list[float] = []
        for s in steps:
            jac = _jaccard_sets(_pairs_from_mask(snapshot_masks[s]), final_set)
            jaccards.append(jac)
            snapshot_curve.append({'step': s, 'jaccard_vs_final': jac})
        for i in range(len(steps)):
            if all(j >= float(jaccard_ref) for j in jaccards[i:]):
                from_snaps = steps[i]
                break

    return {
        'definition_status': 'proposed',
        'freeze_step': int(freeze_step),
        'jaccard_ref': float(jaccard_ref),
        'rel_change_ref': float(rel_change_ref),
        'from_diagnostics': from_diag,
        'from_snapshots': from_snaps,
        'snapshot_curve': snapshot_curve,
    }


def survivor_stability(
    checkpoint_masks: Mapping[int, np.ndarray],
    final_survivors: Sequence[Any],
) -> dict[str, Any]:
    """Jaccard of each checkpoint's nonzero set with the final set, plus first-on / stayed-on steps."""
    final_list = sorted(
        {
            (int(x['layer']), int(x['neuron'])) if isinstance(x, Mapping) else (int(x[0]), int(x[1]))
            for x in final_survivors
        }
    )
    final_set = set(final_list)
    steps = sorted(int(s) for s in checkpoint_masks)
    sets_by_step = {s: _pairs_from_mask(checkpoint_masks[s]) for s in steps}

    by_checkpoint = [
        {
            'step': s,
            'nonzero_count': len(sets_by_step[s]),
            'jaccard_vs_final': _jaccard_sets(sets_by_step[s], final_set),
        }
        for s in steps
    ]

    per_survivor: list[dict[str, Any]] = []
    for pair in final_list:
        on_flags = [pair in sets_by_step[s] for s in steps]
        first_on = next((steps[i] for i, flag in enumerate(on_flags) if flag), None)
        stayed_on: int | None = None
        for i in range(len(steps)):
            if all(on_flags[i:]):
                stayed_on = steps[i]
                break
        per_survivor.append(
            {
                'layer': pair[0],
                'neuron': pair[1],
                'first_on_step': first_on,
                'stayed_on_from_step': stayed_on,
            }
        )

    return {
        'by_checkpoint': by_checkpoint,
        'per_survivor': per_survivor,
    }


# --- 2. What survivors learned ---------------------------------------------------------------------

def summarize_change_sizes(survivors: Sequence[Mapping[str, Any]], num_layers: int) -> dict[str, Any]:
    """Distribution of gauge-invariant ``||Δw_i||`` over survivors and per-layer gate/up ratio."""
    gate_vals = np.asarray([float(s['norm_gate']) for s in survivors if s.get('norm_gate') is not None], dtype=np.float64)
    up_vals = np.asarray([float(s['norm_up']) for s in survivors if s.get('norm_up') is not None], dtype=np.float64)

    def _dist(v: np.ndarray) -> dict[str, Any]:
        if v.size == 0:
            return {'count': 0, 'sum': 0.0, 'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'max': 0.0}
        return {
            'count': int(v.size),
            'sum': float(v.sum()),
            'mean': float(v.mean()),
            'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90)),
            'max': float(v.max()),
        }

    per_layer: list[dict[str, Any]] = []
    for layer in range(int(num_layers)):
        g_sum = float(sum(float(s['norm_gate']) for s in survivors if int(s['layer']) == layer and s.get('norm_gate') is not None))
        u_sum = float(sum(float(s['norm_up']) for s in survivors if int(s['layer']) == layer and s.get('norm_up') is not None))
        ratio = float(g_sum / u_sum) if u_sum > 0.0 else None
        per_layer.append({'layer': layer, 'gate_sum': g_sum, 'up_sum': u_sum, 'gate_over_up': ratio})

    return {
        'gate': _dist(gate_vals),
        'up': _dist(up_vals),
        'per_layer': per_layer,
    }


def abs_cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Sign-free, gauge-invariant absolute cosine similarity ``|u . v| / (||u|| ||v||)``.

    Zero-signal safe: returns ``0.0`` when either vector has zero norm.
    """
    a = np.asarray(u, dtype=np.float64).reshape(-1)
    b = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f'vector shape mismatch: {a.shape} vs {b.shape}')
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(min(1.0, abs(float(np.dot(a, b))) / (na * nb)))


def load_gold_lora_a(adapter_dir: str | Path, layers: Sequence[int], *, scale: float = 1.0) -> dict[str, dict[int, np.ndarray]]:
    """Load gold-sft's saved ``gate_proj`` and ``up_proj`` ``lora_A`` rows via ``neuron_audit_collector``."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for part, proj in (('gate', 'gate_proj'), ('up', 'up_proj')):
        loaded = nac.load_saved_factors(str(adapter_dir), list(layers), scale, projection=proj)
        out[part] = {int(layer): np.asarray(vals['a'], dtype=np.float64) for layer, vals in loaded['layers'].items()}
    return out


def rank1_shared_direction_cosine(
    rewire_a: Mapping[str, Mapping[int, np.ndarray]],
    gold_lora_a: Mapping[str, Mapping[int, np.ndarray]],
    *,
    random_draws: int = RANDOM_COSINE_DRAWS,
    seed: int = RANDOM_COSINE_SEED,
) -> dict[str, Any]:
    """Compare rank-1 shared direction ``a`` against gold LoRA ``lora_A`` per layer and part."""
    rows: list[dict[str, Any]] = []
    dim: int | None = None
    for part in ('gate', 'up'):
        if part not in rewire_a or part not in gold_lora_a:
            continue
        layers = sorted(set(int(l) for l in rewire_a[part]) & set(int(l) for l in gold_lora_a[part]))
        for layer in layers:
            va = np.asarray(rewire_a[part][layer], dtype=np.float64).reshape(-1)
            vg = np.asarray(gold_lora_a[part][layer], dtype=np.float64).reshape(-1)
            dim = va.size if dim is None else dim
            rows.append({'part': part, 'layer': layer, 'abs_cosine': abs_cosine(va, vg)})

    ref: dict[str, Any] | None = None
    if dim is not None and dim > 0 and random_draws > 0:
        rng = np.random.default_rng(int(seed))
        x = rng.normal(size=(int(random_draws), dim))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # Unit vector e_0 has identical marginal |cos| distribution by rotational invariance
        vals = np.abs(x[:, 0])
        ref = {
            'dim': int(dim),
            'draws': int(random_draws),
            'seed': int(seed),
            'mean': float(vals.mean()),
            'p95': float(np.percentile(vals, 95)),
        }

    return {
        'rows': rows,
        'random_reference': ref,
    }


def effective_rank_of_rows(rows: np.ndarray, *, energy_threshold: float = 0.90) -> dict[str, Any]:
    """Rank-0 effective rank (D22, marked ``proposed``): participation ratio and 90%-energy count.

    Given stacked survivor rows ``W_surv`` of shape ``[k, d]``:
    - Participation ratio: ``(sum sigma)^2 / sum(sigma^2)`` (0.0 if ``k == 0`` or all-zero).
    - ``count_for_energy``: smallest number of singular values whose squared sum reaches
      ``energy_threshold * sum(sigma^2)``.
    """
    mat = np.asarray(rows, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f'expected 2D [k, d] matrix, got shape {mat.shape}')
    k = int(mat.shape[0])
    if k == 0:
        return {
            'definition_status': 'proposed',
            'num_rows': 0,
            'participation_ratio': 0.0,
            'count_for_energy': 0,
            'energy_threshold': float(energy_threshold),
            'singular_values': [],
        }
    sv = np.linalg.svd(mat, compute_uv=False)
    sv2 = sv ** 2
    total_energy = float(sv2.sum())
    if total_energy == 0.0:
        pr = 0.0
        n_energy = 0
    else:
        pr = float((sv.sum() ** 2) / total_energy)
        cum = np.cumsum(sv2) / total_energy
        n_energy = int(np.searchsorted(cum, float(energy_threshold) - 1e-12) + 1)
    return {
        'definition_status': 'proposed',
        'num_rows': k,
        'participation_ratio': pr,
        'count_for_energy': n_energy,
        'energy_threshold': float(energy_threshold),
        'singular_values': sv.tolist(),
    }


def summarize_activation_shift(
    survivor_deltas: np.ndarray,
    non_survivor_deltas: np.ndarray,
) -> dict[str, Any]:
    """Summarize teacher-forced post-SiLU activation shifts (signed mean + mean-absolute, lesson 2026-11-11)."""
    def _stats(arr: np.ndarray) -> dict[str, Any]:
        x = np.asarray(arr, dtype=np.float64).ravel()
        if x.size == 0:
            return {'count': 0, 'mean': 0.0, 'mean_abs': 0.0, 'median_abs': 0.0, 'max_abs': 0.0}
        ax = np.abs(x)
        return {
            'count': int(x.size),
            'mean': float(x.mean()),
            'mean_abs': float(ax.mean()),
            'median_abs': float(np.median(ax)),
            'max_abs': float(ax.max()),
        }

    return {
        'survivors': _stats(survivor_deltas),
        'non_survivor_reference': _stats(non_survivor_deltas),
    }


# --- 3. Behaviour and per-question flips -----------------------------------------------------------

def summarize_behaviour_responses(responses: Mapping[str, Mapping[str, Any]], question_ids: Sequence[str]) -> dict[str, Any]:
    """Accuracy, invalid count, capped count, and response length summary."""
    ids = list(question_ids)
    if not ids:
        raise ValueError('question_ids must not be empty')
    rows = [responses[q] for q in ids]
    rewards = np.asarray([float(r['reward']) for r in rows], dtype=np.float64)
    tokens = np.asarray([int(r.get('n_tokens', 0)) for r in rows], dtype=np.float64)
    return {
        'questions': len(ids),
        'accuracy': float(rewards.mean()),
        'invalid': int(sum(bool(r.get('invalid', False)) for r in rows)),
        'capped': int(sum(bool(r.get('capped', False)) for r in rows)),
        'tokens_mean': float(tokens.mean()),
        'tokens_median': float(np.median(tokens)),
    }


def flip_table(
    arm_responses: Mapping[str, Any],
    ref_responses: Mapping[str, Any],
    question_ids: Sequence[str],
    *,
    arm_label: str = 'arm',
    ref_label: str = 'reference',
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """2x2 per-question flip table (fixed, broken, both, neither) with paired bootstrap and McNemar p."""
    ids = list(question_ids)
    if not ids:
        raise ValueError('question_ids must not be empty')
    arm_r = np.asarray([float(arm_responses[q]['reward'] if isinstance(arm_responses[q], Mapping) else arm_responses[q]) for q in ids], dtype=np.float64)
    ref_r = np.asarray([float(ref_responses[q]['reward'] if isinstance(ref_responses[q], Mapping) else ref_responses[q]) for q in ids], dtype=np.float64)

    fixed = int(((arm_r == 1.0) & (ref_r == 0.0)).sum())
    broken = int(((arm_r == 0.0) & (ref_r == 1.0)).sum())
    both = int(((arm_r == 1.0) & (ref_r == 1.0)).sum())
    neither = int(((arm_r == 0.0) & (ref_r == 0.0)).sum())

    counts = cs.resample_counts(len(ids), B=int(bootstrap_draws), seed=int(seed))
    diff = arm_r - ref_r
    boot = (counts @ diff) / len(ids)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        'arm_label': arm_label,
        'ref_label': ref_label,
        'questions': len(ids),
        'arm_accuracy': float(arm_r.mean()),
        'ref_accuracy': float(ref_r.mean()),
        'fixed': fixed,
        'broken': broken,
        'both': both,
        'neither': neither,
        'delta_pp': {
            'estimate': float(100.0 * diff.mean()),
            'p2.5': float(100.0 * lo),
            'p97.5': float(100.0 * hi),
        },
        'mcnemar_exact_p': float(cg.mcnemar_exact(fixed, broken)),
    }


def load_arm_artifacts(
    arm_dir: str | Path,
    plan: Mapping[str, Any],
    *,
    runtime_sha256: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Read and hash-verify a completed production/rehearsal arm directory for standalone analysis (P2.2).

    Restores ``arm_evidence`` if needed, runs ``rewire_checkpoint.verify_arm``, loads step/validation/settle
    logs, ``hardening.json``, every ``masks/trace/*.npz`` snapshot, the final sealed binary survivor export,
    and the final checkpoint tensors.
    """
    import json

    from pilot_eval import rewire_chain as rc
    from pilot_eval import rewire_checkpoint as rckpt
    from pilot_eval import rewire_dynamics as rd

    root = Path(arm_dir)
    if (root / rc.ARM_EVIDENCE_DIR / 'complete.json').is_file():
        rc.restore_arm_evidence(root)
    arm_seal = rckpt.verify_arm(root, plan, runtime_sha256=runtime_sha256, code_commit=code_commit)
    final_step = int(plan['checkpoints'][-1])
    final_ckpt_seal, final_tensors = rckpt.load_tensors(
        rckpt.checkpoint_dir(root, final_step),
        plan,
        runtime_sha256=arm_seal['runtime_sha256'],
        code_commit=arm_seal['code_commit'],
    )
    final_mask_dir = root / 'masks' / f'step-{final_step}'
    au.verify_run(final_mask_dir)
    final_export = json.loads((final_mask_dir / f'{final_step}.json').read_text(encoding='utf-8'))
    if final_export.get('frozen') is not True or final_export.get('survivors_omitted') is True:
        raise ValueError(f'{final_mask_dir}: final survivor export must be frozen=True with full survivor records')

    snapshots: dict[int, dict[str, Any]] = {}
    trace_dir = root / 'masks' / 'trace'
    if trace_dir.is_dir():
        for npz in sorted(trace_dir.glob('*.npz')):
            snap = rd.load_log_alpha_snapshot(npz)
            snapshots[int(snap['step'])] = snap

    hardening = (
        json.loads((root / 'hardening.json').read_text(encoding='utf-8'))
        if (root / 'hardening.json').is_file()
        else None
    )
    return {
        'arm': plan['arm'],
        'rank': int(plan['rank']),
        'arm_seal': arm_seal,
        'final_checkpoint_seal': final_ckpt_seal,
        'final_tensors': final_tensors,
        'final_export': final_export,
        'step_rows': rd.read_jsonl(root / 'logs' / 'steps.jsonl'),
        'val_ce_rows': rd.read_jsonl(root / 'logs' / 'validation.jsonl'),
        'settle_rows': rd.read_jsonl(root / 'logs' / 'settle.jsonl'),
        'hardening': hardening,
        'snapshots': snapshots,
    }


def seal_standalone_analysis(
    directory: str | Path,
    analysis: Mapping[str, Any],
    *,
    filename: str = 'standalone_analysis.json',
) -> str:
    """Seal a standalone analysis dict using ``attribution_units.seal_run``."""
    payload = {'schema': SCHEMA, **dict(analysis)}
    return au.seal_run(Path(directory), {filename: payload})

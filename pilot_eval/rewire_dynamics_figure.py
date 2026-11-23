"""Sparse rewiring ticket 13: training-dynamics figure (D20, D21).

Builds the four-panel training-dynamics figure on a shared step axis (plus a per-layer nonzero
count heatmap), both for a single arm and side-by-side for rank 1 vs rank 0 on shared y axes.

Strict contract:
- Refuses step logs with missing, duplicate, or out-of-order steps instead of interpolating.
- Shows every real evaluation point as an explicit marker.
- Styling comes from ``c_attribution_report._style()``.
- Outputs PNG, SVG, the plotted CSV table, and a provenance JSON with SHA-256 hashes and the
  required descriptive caption ("co-movement, not causation").
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_report as car

CAPTION: str = (
    'Descriptive co-movement of gate shutting, gauge-invariant read-change norms, and performance '
    'within one run; co-movement is not evidence that one quantity causes another.'
)
PROVENANCE_SCHEMA: str = 'sparse-rewiring-dynamics-figure-v1'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contiguous_steps(step_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate that ``step_rows`` are strictly contiguous (no missing, duplicate, or unsorted steps)."""
    if not step_rows:
        raise ValueError('step_rows must not be empty')
    rows = [dict(r) for r in step_rows]
    steps = [int(r['step']) for r in rows]
    if steps != sorted(steps):
        raise ValueError(f'step_rows are not sorted by step: {steps[:10]}')
    if len(set(steps)) != len(steps):
        raise ValueError('step_rows contain duplicate step numbers')
    expected = list(range(steps[0], steps[-1] + 1))
    if steps != expected:
        missing = sorted(set(expected) - set(steps))
        raise ValueError(f'step_rows have missing steps (refusing to interpolate): {missing[:10]}')
    return rows


def _extract_series(arm_data: Mapping[str, Any]) -> dict[str, Any]:
    rows = validate_contiguous_steps(arm_data['step_rows'])
    steps = np.asarray([int(r['step']) for r in rows], dtype=np.int64)
    expected_active = np.asarray([float(r['expected_active']) for r in rows], dtype=np.float64)
    nonzero = np.asarray([int(r['nonzero_z_det']) for r in rows], dtype=np.int64)
    binary = np.asarray(
        [float(r['binary_survivors']) if r.get('binary_survivors') is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    per_layer = np.asarray([r['per_layer_nonzero'] for r in rows], dtype=np.float64).T  # [L, T]

    def _norm(group: str, stat: str) -> np.ndarray:
        return np.asarray([float(r['norms'][group][stat]) for r in rows], dtype=np.float64)

    val_ce_rows = sorted(arm_data.get('val_ce_rows', ()), key=lambda r: int(r['step']))
    val_acc_rows = sorted(arm_data.get('val_acc_rows', ()), key=lambda r: int(r['step']))
    cohort_rows = sorted(arm_data.get('cohort150_rows', ()), key=lambda r: int(r['step']))
    settle_rows = sorted(arm_data.get('settle_rows', ()), key=lambda r: int(r['step']))
    churn_rows = sorted(arm_data.get('churn_rows', ()), key=lambda r: int(r['step']))

    return {
        'label': str(arm_data.get('label', 'arm')),
        'steps': steps,
        'expected_active': expected_active,
        'nonzero_z_det': nonzero,
        'binary_survivors': binary,
        'per_layer_nonzero': per_layer,
        'gate_all_sum': _norm('gate_all', 'sum'),
        'up_all_sum': _norm('up_all', 'sum'),
        'gate_surv_median': _norm('gate_survivors', 'median'),
        'up_surv_median': _norm('up_survivors', 'median'),
        'gate_surv_sum': _norm('gate_survivors', 'sum'),
        'up_surv_sum': _norm('up_survivors', 'sum'),
        'val_ce': [(int(r['step']), float(r['val_ce'])) for r in val_ce_rows],
        'val_acc': [(int(r['step']), float(r['accuracy'])) for r in val_acc_rows],
        'cohort150_acc': [(int(r['step']), float(r['accuracy'])) for r in cohort_rows],
        'settle': [(int(r['step']), float(r['jaccard']), r.get('expected_active_rel_change')) for r in settle_rows],
        'churn': [(int(r['step']), float(r['jaccard']), int(r['switched_on']), int(r['switched_off'])) for r in churn_rows],
        'references': dict(arm_data.get('references') or {}),
    }


def _write_plotted_csv(path: Path, series_list: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        'arm',
        'step',
        'expected_active',
        'nonzero_z_det',
        'binary_survivors',
        'gate_all_sum',
        'up_all_sum',
        'gate_surv_sum',
        'up_surv_sum',
        'gate_surv_median',
        'up_surv_median',
        'val_ce',
        'val_acc',
        'cohort150_acc',
        'settle_jaccard',
        'churn_jaccard',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for s in series_list:
            val_ce_map = dict(s['val_ce'])
            val_acc_map = dict(s['val_acc'])
            c150_map = dict(s['cohort150_acc'])
            settle_map = {st: jac for st, jac, _ in s['settle']}
            churn_map = {st: jac for st, jac, _, _ in s['churn']}
            for i, step in enumerate(s['steps'].tolist()):
                b_val = s['binary_survivors'][i]
                writer.writerow(
                    {
                        'arm': s['label'],
                        'step': step,
                        'expected_active': float(s['expected_active'][i]),
                        'nonzero_z_det': int(s['nonzero_z_det'][i]),
                        'binary_survivors': '' if np.isnan(b_val) else int(b_val),
                        'gate_all_sum': float(s['gate_all_sum'][i]),
                        'up_all_sum': float(s['up_all_sum'][i]),
                        'gate_surv_sum': float(s['gate_surv_sum'][i]),
                        'up_surv_sum': float(s['up_surv_sum'][i]),
                        'gate_surv_median': float(s['gate_surv_median'][i]),
                        'up_surv_median': float(s['up_surv_median'][i]),
                        'val_ce': val_ce_map.get(step, ''),
                        'val_acc': val_acc_map.get(step, ''),
                        'cohort150_acc': c150_map.get(step, ''),
                        'settle_jaccard': settle_map.get(step, ''),
                        'churn_jaccard': churn_map.get(step, ''),
                    }
                )


def _plot_arm_column(axes: Sequence[Any], s: dict[str, Any], *, freeze_step: int | None) -> list[Any]:
    """Plot the 5 rows (Gates, Per-layer heatmap, Read-change norms, Performance, Events) into ``axes``."""
    ax_gate, ax_heat, ax_norm, ax_perf, ax_evt = axes
    steps = s['steps']

    # 1. Gates (log y axis)
    ax_gate.set_title(f"{s['label']} — 1. Gates (log scale)")
    ax_gate.plot(steps, np.maximum(s['expected_active'], 0.5), label='expected-active', color='#2a78d6')
    ax_gate.plot(steps, np.maximum(s['nonzero_z_det'].astype(float), 0.5), label='nonzero z_det', color='#d95f02')
    valid_bin = ~np.isnan(s['binary_survivors'])
    if valid_bin.any():
        ax_gate.plot(
            steps[valid_bin],
            np.maximum(s['binary_survivors'][valid_bin], 0.5),
            'o',
            markersize=4,
            label='binary survivors',
            color='#1b9e77',
        )
    ax_gate.set_yscale('log')
    ax_gate.set_ylabel('neuron count')
    ax_gate.legend(loc='upper right', fontsize=8)

    # 1b. Per-layer nonzero heatmap
    ax_heat.set_title('1b. Per-layer nonzero z_det count')
    grid = s['per_layer_nonzero']
    extent = [float(steps[0]), float(steps[-1]) if steps.size > 1 else float(steps[0]) + 1.0, grid.shape[0] - 0.5, -0.5]
    ax_heat.imshow(grid, aspect='auto', cmap='viridis', extent=extent, interpolation='nearest')
    ax_heat.set_ylabel('layer')
    ax_heat.grid(False)

    # 2. Gauge-invariant read-change norms
    ax_norm.set_title('2. Gauge-invariant read-change norms ||Δw||')
    ax_norm.plot(steps, s['gate_all_sum'], label='gate sum (all)', color='#2a78d6')
    ax_norm.plot(steps, s['up_all_sum'], label='up sum (all)', color='#86b6ef', linestyle='--')
    ax_norm.plot(steps, s['gate_surv_median'], label='gate median (surv)', color='#d95f02')
    ax_norm.plot(steps, s['up_surv_median'], label='up median (surv)', color='#e7298a', linestyle=':')
    ax_norm.set_ylabel('L2 norm')
    ax_norm.legend(loc='upper left', fontsize=8, ncol=2)

    # 3. Performance (Val CE + greedy accuracy on twin axis, explicit markers only)
    ax_perf.set_title('3. Performance (val CE & greedy accuracy)')
    if s['val_ce']:
        xs, ys = zip(*s['val_ce'])
        ax_perf.plot(xs, ys, 'o-', markersize=3.5, label='val CE', color='#2a78d6')
    ax_perf.set_ylabel('validation CE')
    refs = s['references']
    if 'untuned_val_ce' in refs and refs['untuned_val_ce'] is not None:
        ax_perf.axhline(float(refs['untuned_val_ce']), color='#888888', linestyle=':', linewidth=1.2, label='untuned CE')
    if 'gold_sft_val_ce' in refs and refs['gold_sft_val_ce'] is not None:
        ax_perf.axhline(float(refs['gold_sft_val_ce']), color='#444444', linestyle='--', linewidth=1.2, label='gold-sft s64 CE')

    ax_acc = ax_perf.twinx()
    ax_acc.spines['top'].set_visible(False)
    if s['val_acc']:
        xs, ys = zip(*s['val_acc'])
        ax_acc.plot(xs, ys, 's', markersize=4.5, label='val greedy acc', color='#1b9e77')
    if s['cohort150_acc']:
        xs, ys = zip(*s['cohort150_acc'])
        ax_acc.plot(xs, ys, '^', markersize=6.0, label='cohort150 (exploratory)', color='#e7298a')
    if 'untuned_val_acc' in refs and refs['untuned_val_acc'] is not None:
        ax_acc.axhline(float(refs['untuned_val_acc']), color='#1b9e77', linestyle=':', linewidth=1.0, alpha=0.6)
    if 'gold_sft_val_acc' in refs and refs['gold_sft_val_acc'] is not None:
        ax_acc.axhline(float(refs['gold_sft_val_acc']), color='#1b9e77', linestyle='--', linewidth=1.0, alpha=0.6)
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_ylim(-0.02, 1.02)
    h1, l1 = ax_perf.get_legend_handles_labels()
    h2, l2 = ax_acc.get_legend_handles_labels()
    if h1 or h2:
        ax_perf.legend(h1 + h2, l1 + l2, loc='best', fontsize=7.5)

    # 4. Events (settle Jaccard & checkpoint churn)
    ax_evt.set_title('4. Events: settle diagnostics & checkpoint churn')
    if s['settle']:
        xs = [t[0] for t in s['settle']]
        ys = [t[1] for t in s['settle']]
        ax_evt.plot(xs, ys, 'o-', markersize=4.5, label='settle Jaccard (64-step)', color='#2a78d6')
    if s['churn']:
        xs = [t[0] for t in s['churn']]
        ys = [t[1] for t in s['churn']]
        ax_evt.plot(xs, ys, 'D', markersize=4.5, label='checkpoint churn Jaccard', color='#d95f02')
    ax_evt.set_ylim(-0.02, 1.05)
    ax_evt.set_ylabel('Jaccard')
    ax_evt.set_xlabel('training step')
    ax_evt.legend(loc='lower right', fontsize=8)

    if freeze_step is not None:
        for ax in (ax_gate, ax_heat, ax_norm, ax_perf, ax_evt):
            ax.axvline(float(freeze_step), color='#a32a2a', linestyle='--', linewidth=1.1, alpha=0.8)

    return [ax_acc]


def _save_bundle(
    fig: Any,
    plt: Any,
    out_dir: Path,
    *,
    name: str,
    series_list: Sequence[dict[str, Any]],
    freeze_step: int | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f'{name}.png'
    svg_path = out_dir / f'{name}.svg'
    csv_path = out_dir / f'{name}.csv'
    prov_path = out_dir / f'{name}_provenance.json'

    fig.suptitle(CAPTION, y=0.01, fontsize=8, color='#555555', style='italic')
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)

    _write_plotted_csv(csv_path, series_list)
    provenance = {
        'schema': PROVENANCE_SCHEMA,
        'name': name,
        'caption': CAPTION,
        'freeze_step': int(freeze_step) if freeze_step is not None else None,
        'arms': [
            {
                'label': s['label'],
                'step_start': int(s['steps'][0]),
                'step_end': int(s['steps'][-1]),
                'num_steps': int(s['steps'].size),
                'num_layers': int(s['per_layer_nonzero'].shape[0]),
            }
            for s in series_list
        ],
        'outputs': {
            'png': {'file': png_path.name, 'sha256': _sha256(png_path)},
            'svg': {'file': svg_path.name, 'sha256': _sha256(svg_path)},
            'csv': {'file': csv_path.name, 'sha256': _sha256(csv_path)},
        },
    }
    prov_path.write_bytes(au._encode(provenance))
    return provenance


def build_dynamics_figure(
    arm_data: Mapping[str, Any],
    out_dir: str | Path,
    *,
    name: str = 'dynamics',
    freeze_step: int | None = 960,
) -> dict[str, Any]:
    """Render the single-arm training-dynamics figure (PNG, SVG, CSV, provenance JSON)."""
    s = _extract_series(arm_data)
    plt = car._style()
    fig, axes = plt.subplots(5, 1, figsize=(8.5, 12.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0, 1.1, 1.2, 1.0]})
    _plot_arm_column(axes, s, freeze_step=freeze_step)
    return _save_bundle(fig, plt, Path(out_dir), name=name, series_list=[s], freeze_step=freeze_step)


def build_side_by_side_dynamics_figure(
    arm_left: Mapping[str, Any],
    arm_right: Mapping[str, Any],
    out_dir: str | Path,
    *,
    name: str = 'dynamics_r1_vs_r0',
    freeze_step: int | None = 960,
) -> dict[str, Any]:
    """Render rank-1 and rank-0 training dynamics side by side on shared y axes (D21)."""
    s_left = _extract_series(arm_left)
    s_right = _extract_series(arm_right)
    plt = car._style()
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(14.5, 12.5),
        sharex='col',
        sharey='row',
        gridspec_kw={'height_ratios': [1.2, 1.0, 1.1, 1.2, 1.0]},
    )
    _plot_arm_column([axes[r, 0] for r in range(5)], s_left, freeze_step=freeze_step)
    _plot_arm_column([axes[r, 1] for r in range(5)], s_right, freeze_step=freeze_step)
    return _save_bundle(fig, plt, Path(out_dir), name=name, series_list=[s_left, s_right], freeze_step=freeze_step)

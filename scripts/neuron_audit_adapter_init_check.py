"""Post hoc Stage 0: shared initialization and drift of the rank-1 LoRA adapters (weights only, CPU).

For every arm, layer and module it compares the step-0 and audited-step adapters from the sealed inventory:
  - shared init across arms (tensor-level, not just file hash)
  - b0 norm (PEFT default: zero)
  - read drift   ||a_T - a_0|| / ||a_0||  and cos(a_T, a_0)
  - update size  s * ||a_T|| * ||b_T||  (= ||Delta W||_F, gauge-free)
  - down_proj only: Spearman(|a_0|, |a_T|) across neurons and top-64 |a| overlap step 0 vs step T,
    i.e. how much of a per-neuron |a| ranking (and so of B = |a| E|z|) is inherited from the random init
  - down_proj only: cross-arm |cos| of a_T versus of Delta a = a_T - a_0 (preview for Stage 1)

  python3 -I scripts/neuron_audit_adapter_init_check.py --artifact_root <artifacts> \
      --inventory runs/joint-neuron-adapter-audit/inventory/inventory-v2 --output_dir <out> [--top_k 64]

Outputs (resumable: an existing complete status with matching inputs is reused unless --force):
  status.json, rows.jsonl (arm x layer x module), summary.json, cross_arm_down.json, figures/*.png
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
PARENT = {m: ('self_attn' if m in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp') for m in MODULES}
ARM_COLOURS = {'gold-sft': '#2a78b5', 'online-grpo': '#d0572e', 'offline-grpo': '#e5a82b',
               'sampling-sft-k0': '#3c9a5f', 'sampling-sft-k2': '#8a5fb8', 'sampling-sft-k5': '#c45a97'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def key(layer, module, part):
    return f'base_model.model.model.layers.{layer}.{PARENT[module]}.{module}.lora_{part}.weight'


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def cos(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else float('nan')


def load_arms(artifact_root, inventory_dir, init_step, audit_step):
    inv = json.loads((inventory_dir / 'inventory.json').read_text())
    src = json.loads((inventory_dir / 'sources.json').read_text())
    arms, files = {}, {}
    for entry in src['arms']:
        if not entry['checkpoints']:
            continue
        arm = entry['arm']
        if inv['arms'][arm]['adapted_measurement'] != 'load-ready':
            raise ValueError(f'{arm} is not load-ready in the inventory')
        arms[arm] = {}
        for step in (init_step, audit_step):
            ckpt = inv['arms'][arm]['checkpoints'][str(step)]
            path = artifact_root / entry['checkpoints'][str(step)]['path'] / 'adapter_model.safetensors'
            digest = sha256(path)
            expected = ckpt['files']['adapter_model.safetensors']['sha256']  # sealed inventory record
            if digest != expected:
                raise ValueError(f'{arm} step {step}: sha256 {digest} != inventory {expected}')
            arms[arm][step] = dict(tensors=load_file(str(path)), scale=float(ckpt['config']['scale']))
            files[f'{arm}/{step}'] = dict(path=str(path.relative_to(artifact_root)), sha256=digest)
    return arms, files


def analyse(arms, init_step, audit_step, top_k):
    names = sorted(arms)
    first = arms[names[0]][init_step]['tensors']
    shared_init = {arm: max(float(np.max(np.abs(arms[arm][init_step]['tensors'][k] - first[k]))) for k in first)
                   for arm in names}
    layers = sorted({int(k.split('.layers.')[1].split('.')[0]) for k in first})
    rows = []
    for arm in names:
        t0, tT = arms[arm][init_step]['tensors'], arms[arm][audit_step]['tensors']
        s = arms[arm][audit_step]['scale']
        for layer in layers:
            for m in MODULES:
                a0, aT = t0[key(layer, m, 'A')].ravel().astype(np.float64), tT[key(layer, m, 'A')].ravel().astype(np.float64)
                b0, bT = t0[key(layer, m, 'B')].ravel().astype(np.float64), tT[key(layer, m, 'B')].ravel().astype(np.float64)
                row = dict(arm=arm, layer=layer, module=m, a_norm0=float(np.linalg.norm(a0)),
                           a_normT=float(np.linalg.norm(aT)), b_norm0=float(np.linalg.norm(b0)),
                           b_normT=float(np.linalg.norm(bT)),
                           a_drift=float(np.linalg.norm(aT - a0) / np.linalg.norm(a0)), a_cos_init=cos(aT, a0),
                           update_fro=float(s * np.linalg.norm(aT) * np.linalg.norm(bT)))
                if m == 'down_proj':
                    row['abs_a_spearman_init'] = spearman(np.abs(a0), np.abs(aT))
                    top0 = set(np.argsort(-np.abs(a0))[:top_k].tolist())
                    topT = set(np.argsort(-np.abs(aT))[:top_k].tolist())
                    row[f'abs_a_top{top_k}_overlap_init'] = len(top0 & topT) / top_k
                rows.append(row)
    cross = {}
    for x, y in itertools.combinations(names, 2):
        per = {}
        for layer in layers:
            k = key(layer, 'down_proj', 'A')
            a0 = arms[x][init_step]['tensors'][k].ravel().astype(np.float64)
            ax, ay = (arms[n][audit_step]['tensors'][k].ravel().astype(np.float64) for n in (x, y))
            per[layer] = dict(abs_cos_a=abs(cos(ax, ay)), abs_cos_delta_a=abs(cos(ax - a0, ay - a0)))
        cross[f'{x} vs {y}'] = per
    return dict(shared_init_max_abs_diff=shared_init, layers=layers), rows, cross


def summarise(rows, top_k):
    out = {}
    for arm in sorted({r['arm'] for r in rows}):
        out[arm] = {}
        for m in MODULES:
            sel = [r for r in rows if r['arm'] == arm and r['module'] == m]
            d = np.array([r['a_drift'] for r in sel])
            entry = dict(a_drift_median=float(np.median(d)), a_drift_min=float(d.min()), a_drift_max=float(d.max()),
                         a_cos_init_median=float(np.median([r['a_cos_init'] for r in sel])),
                         b_norm0_max=float(max(r['b_norm0'] for r in sel)),
                         update_fro_sum=float(sum(r['update_fro'] for r in sel)))
            if m == 'down_proj':
                entry['abs_a_spearman_init_median'] = float(np.median([r['abs_a_spearman_init'] for r in sel]))
                entry[f'abs_a_top{top_k}_overlap_init_median'] = float(
                    np.median([r[f'abs_a_top{top_k}_overlap_init'] for r in sel]))
            out[arm][m] = entry
    return out


def plot(rows, cross, top_k, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#999999', 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
    arms = sorted({r['arm'] for r in rows})
    down = [r for r in rows if r['module'] == 'down_proj']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for arm in arms:
        sel = sorted((r for r in down if r['arm'] == arm), key=lambda r: r['layer'])
        L = [r['layer'] for r in sel]
        kw = dict(color=ARM_COLOURS.get(arm), lw=2, label=arm)
        axes[0].plot(L, [r['a_drift'] for r in sel], **kw)
        axes[1].plot(L, [r['abs_a_spearman_init'] for r in sel], **kw)
        axes[2].plot(L, [r[f'abs_a_top{top_k}_overlap_init'] for r in sel], **kw)
    axes[0].set_title('How far a moved from its start\n||a64 − a0|| / ||a0||')
    axes[1].set_title('Does |a| still rank neurons like the start?\nSpearman(|a0|, |a64|) over 8,960 neurons')
    axes[2].set_title(f'Top-{top_k} |a| neurons kept from the start\n(fraction of {top_k}; random ≈ {top_k/8960:.3f})')
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.set_xlabel('layer')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('down_proj read vector a: step 0 (shared random init) vs step 64', y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S0a_down_read_drift.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for i, (field, title) in enumerate([('abs_cos_a', '|cos(a64, a64′)|: raw read vectors'),
                                         ('abs_cos_delta_a', '|cos(Δa, Δa′)|: learned part only')]):
        for pair, per in cross.items():
            L = sorted(per, key=int)
            axes[i].plot([int(l) for l in L], [per[l][field] for l in L], color='#7a7a7a', lw=1, alpha=0.6)
        axes[i].axhline(1 / np.sqrt(8960), color='#c03030', lw=1, ls='--')
        axes[i].set_title(title)
        axes[i].set_xlabel('layer')
    axes[0].set_ylabel('|cosine| between two arms (15 pairs)')
    axes[1].text(0.5, 1 / np.sqrt(8960) + 0.02, 'random directions ≈ 0.011', color='#c03030', fontsize=8)
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle('Why cross-arm read similarity must use Δa: the shared init dominates raw a', y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S0b_cross_arm_a_vs_delta_a.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    for i, m in enumerate(MODULES):
        for j, arm in enumerate(arms):
            vals = [r['a_drift'] for r in rows if r['module'] == m and r['arm'] == arm]
            ax.scatter(np.full(len(vals), i + (j - 2.5) * 0.11), vals, s=8, color=ARM_COLOURS.get(arm),
                       label=arm if i == 0 else None, alpha=0.8)
    ax.set_xticks(range(len(MODULES)), MODULES)
    ax.set_yscale('log')
    ax.set_ylabel('||a64 − a0|| / ||a0||  (28 layers each)')
    ax.set_title('Read-vector drift by module')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S0c_drift_by_module.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--inventory', required=True, help='inventory dir, relative to artifact_root or absolute')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--init_step', type=int, default=0)
    p.add_argument('--audit_step', type=int, default=64)
    p.add_argument('--top_k', type=int, default=64)
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    root = Path(a.artifact_root)
    inv_dir = Path(a.inventory) if Path(a.inventory).is_absolute() else root / a.inventory
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    config = dict(inventory=str(inv_dir), inventory_sha256=sha256(inv_dir / 'inventory.json'),
                  init_step=a.init_step, audit_step=a.audit_step, top_k=a.top_k)
    if status_path.exists() and not a.force:
        old = json.loads(status_path.read_text())
        if old.get('complete') and old.get('config') == config:
            print('STAGE0 reused', status_path)
            return 0
    atomic_json(status_path, dict(complete=False, config=config, label='post hoc, exploratory'))
    arms, files = load_arms(root, inv_dir, a.init_step, a.audit_step)
    meta, rows, cross = analyse(arms, a.init_step, a.audit_step, a.top_k)
    tmp = out / 'rows.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    os.replace(tmp, out / 'rows.jsonl')
    summary = dict(meta, per_arm=summarise(rows, a.top_k))
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'cross_arm_down.json', cross)
    plot(rows, cross, a.top_k, out / 'figures')
    atomic_json(status_path, dict(complete=True, config=config, files=files, n_rows=len(rows),
                                  label='post hoc, exploratory'))
    print('STAGE0 complete rows', len(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Post hoc Stage 1: cross-arm similarity of the rank-1 LoRA adapters, from weights only (CPU).

All arms share one random step-0 read vector a0 and start with b0 = 0 (Stage 0), so:
  - read similarity uses the learned part  da = a_T - a_0  (raw a is dominated by the shared init)
  - shared-direction control: for each pair (m, n) the mean da of the arms outside both arms' families
    (sampling-SFT k0/k2/k5, GRPO online/offline, gold SFT) is projected out of da_m and da_n before the
    cosine, so the control is fit neither on the pair nor on its near-duplicate siblings
  - write similarity uses b_T directly (b0 = 0, and a_T shares its init sign across arms, so the sign of
    cos(b) is comparable); full-update similarity cos(dW_m, dW_n) = cos(a_m, a_n) * cos(b_m, b_n) with
    dW = s b_T a_T^T (= the whole update, since b0 = 0); this product is invariant to the rank-1 sign flip
  - per arm: update size ||dW||_F = s ||a_T|| ||b_T|| and the share of a_T lying along the init direction

Every module (q, k, v, o, gate, up, down) at every layer is covered. Random-direction reference for a
cosine in d dimensions: about 1/sqrt(d).

  python3 -I scripts/neuron_audit_adapter_cross_arm.py --artifact_root <artifacts> \
      --inventory runs/joint-neuron-adapter-audit/inventory/inventory-v2 --output_dir <out>

Outputs (resumable unless --force): status.json, pairs.jsonl, arms.jsonl, summary.json, figures/*.png
"""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('stage0', Path(__file__).with_name('neuron_audit_adapter_init_check.py'))
stage0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage0)
MODULES, key, cos, sha256, atomic_json, load_arms = (stage0.MODULES, stage0.key, stage0.cos, stage0.sha256,
                                                     stage0.atomic_json, stage0.load_arms)
ARM_ORDER = ['gold-sft', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5', 'offline-grpo', 'online-grpo']
ARM_COLOURS = stage0.ARM_COLOURS
METRICS = [('cos_delta_a', 'Read change Δa'), ('cos_delta_a_resid', 'Δa, shared direction removed'),
           ('cos_b', 'Write vector b'), ('cos_dW', 'Full update ΔW')]


def family(arm):
    return 'sampling-sft' if arm.startswith('sampling-sft') else 'grpo' if 'grpo' in arm else arm


def vec(t, layer, m, part):
    return t[key(layer, m, part)].ravel().astype(np.float64)


def project_out(x, u):
    n = np.linalg.norm(u)
    return x if n == 0 else x - (x @ u) / (n * n) * u


def analyse(arms, init_step, audit_step):
    names = [a for a in ARM_ORDER if a in arms] + sorted(set(arms) - set(ARM_ORDER))
    t0 = arms[names[0]][init_step]['tensors']
    layers = sorted({int(k.split('.layers.')[1].split('.')[0]) for k in t0})
    T = {n: arms[n][audit_step]['tensors'] for n in names}
    S = {n: arms[n][audit_step]['scale'] for n in names}
    arm_rows, pair_rows = [], []
    for layer in layers:
        for m in MODULES:
            a0 = vec(t0, layer, m, 'A')
            aT = {n: vec(T[n], layer, m, 'A') for n in names}
            bT = {n: vec(T[n], layer, m, 'B') for n in names}
            da = {n: aT[n] - a0 for n in names}
            for n in names:
                arm_rows.append(dict(arm=n, layer=layer, module=m,
                                     update_fro=float(S[n] * np.linalg.norm(aT[n]) * np.linalg.norm(bT[n])),
                                     a_init_share=cos(aT[n], a0) ** 2,
                                     da_init_cos=cos(da[n], a0)))
            for x, y in itertools.combinations(names, 2):
                # shared direction from arms outside BOTH arms' families: near-duplicate arms (k0/k2/k5)
                # would otherwise remove the pair's own family direction
                others = [n for n in names if family(n) not in (family(x), family(y))]
                shared = np.mean([da[n] for n in others], axis=0)
                ca, cb = cos(aT[x], aT[y]), cos(bT[x], bT[y])
                pair_rows.append(dict(pair=f'{x}|{y}', layer=layer, module=m, dim_a=a0.size, dim_b=bT[x].size,
                                      cos_a=ca, cos_delta_a=cos(da[x], da[y]),
                                      cos_delta_a_resid=cos(project_out(da[x], shared), project_out(da[y], shared)),
                                      cos_b=cb, cos_dW=ca * cb))
    return names, layers, arm_rows, pair_rows


def summarise(names, pair_rows, arm_rows):
    out = dict(arms=names, random_reference={m: None for m in MODULES}, pairs={}, update_fro={})
    for m in MODULES:
        r = next(p for p in pair_rows if p['module'] == m)
        out['random_reference'][m] = dict(a=1 / np.sqrt(r['dim_a']), b=1 / np.sqrt(r['dim_b']))
    for p in sorted({r['pair'] for r in pair_rows}):
        out['pairs'][p] = {}
        for m in MODULES:
            sel = [r for r in pair_rows if r['pair'] == p and r['module'] == m]
            out['pairs'][p][m] = {f: float(np.median([r[f] for r in sel])) for f, _ in METRICS}
    for n in names:
        out['update_fro'][n] = {m: float(sum(r['update_fro'] for r in arm_rows if r['arm'] == n and r['module'] == m))
                                for m in MODULES}
        out['update_fro'][n]['a_init_share_median'] = float(np.median([r['a_init_share'] for r in arm_rows
                                                                       if r['arm'] == n]))
    return out


def short(n):
    return n.replace('sampling-sft-', 'samp-').replace('-grpo', ' GRPO').replace('gold-sft', 'gold SFT')


def plot(names, layers, pair_rows, arm_rows, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#999999'})
    k = len(names)

    def matrix(field, module):
        M = np.eye(k)
        for r in pair_rows:
            if r['module'] != module:
                continue
            x, y = r['pair'].split('|')
            i, j = names.index(x), names.index(y)
            M[i, j] = M[j, i] = np.nan  # filled below
        for i, j in itertools.combinations(range(k), 2):
            vals = [r[field] for r in pair_rows if r['module'] == module and r['pair'] == f'{names[i]}|{names[j]}']
            M[i, j] = M[j, i] = float(np.median(vals))
        return M

    # S1a: down_proj, four metrics as 6x6 matrices (median over 28 layers)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (field, title) in zip(axes, METRICS):
        M = matrix(field, 'down_proj')
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
        for i in range(k):
            for j in range(k):
                if i != j:
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(M[i, j]) > 0.6 else '#222222')
        ax.set_xticks(range(k), [short(n) for n in names], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(k), [short(n) for n in names] if ax is axes[0] else [], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label='cosine (median over 28 layers)')
    fig.suptitle('down_proj adapter: how alike are the six arms? (signed cosine; random ≈ 0.01–0.03)', y=1.02)
    fig.savefig(fig_dir / 'S1a_down_matrices.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S1b: every module, same four metrics, median over layers and over pairs, grouped by pair type
    groups = {'sampling-SFT vs sampling-SFT': lambda x, y: x.startswith('sampling') and y.startswith('sampling'),
              'online vs offline GRPO': lambda x, y: 'grpo' in x and 'grpo' in y,
              'across methods': lambda x, y: not (x.startswith('sampling') and y.startswith('sampling'))
              and not ('grpo' in x and 'grpo' in y)}
    gcol = {'sampling-SFT vs sampling-SFT': '#3c9a5f', 'online vs offline GRPO': '#d0572e', 'across methods': '#6b6b6b'}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (field, title) in zip(axes, METRICS):
        for gi, (g, test) in enumerate(groups.items()):
            vals = []
            for m in MODULES:
                sel = [r[field] for r in pair_rows if r['module'] == m and test(*r['pair'].split('|'))]
                vals.append(np.median(sel))
            ax.bar(np.arange(len(MODULES)) + (gi - 1) * 0.27, vals, width=0.25, color=gcol[g],
                   label=g if ax is axes[0] else None)
        ax.axhline(0, color='#999999', lw=0.8)
        ax.set_xticks(range(len(MODULES)), [m.replace('_proj', '') for m in MODULES])
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('median cosine (layers × pairs)')
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('All seven adapted modules: similarity by pair type', y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S1b_modules_by_pair_type.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S1c: down_proj by layer, pair-type medians, four metrics
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (field, title) in zip(axes, METRICS):
        for g, test in groups.items():
            ys = [np.median([r[field] for r in pair_rows if r['module'] == 'down_proj' and r['layer'] == l
                             and test(*r['pair'].split('|'))]) for l in layers]
            ax.plot(layers, ys, color=gcol[g], lw=2, label=g if ax is axes[0] else None)
        ax.axhline(0, color='#999999', lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('layer')
        ax.grid(True, color='#e6e6e6', lw=0.6)
    axes[0].set_ylabel('median cosine over pairs in group')
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('down_proj adapter similarity by layer', y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S1c_down_by_layer.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S1d: update size per layer, per arm (all modules summed, and down only)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for n in names:
        tot = [sum(r['update_fro'] for r in arm_rows if r['arm'] == n and r['layer'] == l) for l in layers]
        dn = [next(r['update_fro'] for r in arm_rows if r['arm'] == n and r['layer'] == l and r['module'] == 'down_proj')
              for l in layers]
        axes[0].plot(layers, tot, color=ARM_COLOURS.get(n), lw=2, label=n)
        axes[1].plot(layers, dn, color=ARM_COLOURS.get(n), lw=2)
    axes[0].set_title('Update size, all 7 modules summed\nΣ s·||a||·||b|| per layer')
    axes[1].set_title('Update size, down_proj only\ns·||a||·||b||')
    for ax in axes:
        ax.set_xlabel('layer')
        ax.grid(True, color='#e6e6e6', lw=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S1d_update_size.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--init_step', type=int, default=0)
    p.add_argument('--audit_step', type=int, default=64)
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    root = Path(a.artifact_root)
    inv_dir = Path(a.inventory) if Path(a.inventory).is_absolute() else root / a.inventory
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    config = dict(inventory=str(inv_dir), inventory_sha256=sha256(inv_dir / 'inventory.json'),
                  init_step=a.init_step, audit_step=a.audit_step)
    if status_path.exists() and not a.force:
        old = json.loads(status_path.read_text())
        if old.get('complete') and old.get('config') == config:
            print('STAGE1 reused', status_path)
            return 0
    atomic_json(status_path, dict(complete=False, config=config, label='post hoc, exploratory'))
    arms, files = load_arms(root, inv_dir, a.init_step, a.audit_step)
    names, layers, arm_rows, pair_rows = analyse(arms, a.init_step, a.audit_step)
    for name, rows in (('pairs.jsonl', pair_rows), ('arms.jsonl', arm_rows)):
        tmp = out / (name + '.tmp')
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        os.replace(tmp, out / name)
    atomic_json(out / 'summary.json', summarise(names, pair_rows, arm_rows))
    plot(names, layers, pair_rows, arm_rows, out / 'figures')
    atomic_json(status_path, dict(complete=True, config=config, files=files, n_pairs=len(pair_rows),
                                  n_arm_rows=len(arm_rows), label='post hoc, exploratory'))
    print('STAGE1 complete pair rows', len(pair_rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

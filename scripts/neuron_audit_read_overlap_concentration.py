"""Post hoc Stages 2-3: cross-arm overlap of what the down_proj adapters read (B), and read concentration C(K).

Objects, per arm and layer (down_proj adapter, 8,960 neurons):
  a0, aT, da = aT - a0         read vectors (shared random init a0; Stage 0)
  Z_split    = mean over math texts of the arm's own per-text mean|z_j|   (from the extract)
  B          = |aT| * Z        (= the frozen list-B score, exactly: mean|a_j z_j| with a_j fixed)
  B_learned  = |da| * Z        (read weight learned since init)
  B_init     = |a0| * Z        (what the untrained random read would give with this arm's activity)
Neurons are always ranked on DISCOVERY math texts and scored on HELD-OUT math texts.

Stage 2 (pairs of arms): top-64 overlap of the frozen B lists, and of B_learned, |da|, |aT| and Z alone,
each against a null that redraws the second list within the report's base-activity bins (bins.json);
Spearman of B and B_learned over all neurons on held-out texts.

Stage 3 (each arm): p_j = score_j / sum score, C(K) = held-out mass in the discovery top-K, and
N_eff = 1 / sum p_j^2 (effective number of neurons). Null: |aT| (or |da|) shuffled across neurons with
activity kept, so concentration beyond the null is concentration the adapter itself learned.
C(K) ignores signs, so it says nothing about cancellation in c = a^T z.

  python3 -I scripts/neuron_audit_read_overlap_concentration.py --artifact_root <artifacts> \
      --inventory runs/joint-neuron-adapter-audit/inventory/inventory-v2 --extract_dir <extract-v1> \
      --selection_dir <selection/s-v1> --output_dir <out> [--draws 200 --shuffles 200 --seed 20261112]

Outputs (resumable unless --force): status.json, validation.json, pairs.jsonl, concentration.jsonl,
summary.json, figures/*.png
"""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _here.with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage0 = _load('stage0', 'neuron_audit_adapter_init_check.py')
stage1 = _load('stage1', 'neuron_audit_adapter_cross_arm.py')
key, sha256, atomic_json, load_arms = stage0.key, stage0.sha256, stage0.atomic_json, stage0.load_arms
ARM_ORDER, ARM_COLOURS, family, short = stage1.ARM_ORDER, stage1.ARM_COLOURS, stage1.family, stage1.short
KS = [8, 16, 32, 64, 128, 256, 448, 896]
TOP = 64
VERSIONS = ['B', 'B_learned', 'B_init', 'activity', 'abs_a', 'abs_da']


def ranks(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def top(score, k=TOP):
    return np.argsort(-score, kind='stable')[:k]


def binned_null(fixed, randomized, labels, bins, draws, rng):
    """Overlap |X & Y'| / |Y| where Y' redraws Y within the base-activity bins of Y's members."""
    fixed = np.zeros(labels.size, bool) if fixed is None else fixed
    counts = {b: int(np.sum(labels[randomized] == b)) for b in bins}
    out = np.zeros(draws)
    for d in range(draws):
        hit = 0
        for b, c in counts.items():
            if c:
                hit += int(fixed[rng.choice(bins[b], c, replace=False)].sum())
        out[d] = hit / len(randomized)
    return out


def concentration(rank_score, eval_score):
    order = np.argsort(-rank_score, kind='stable')
    p = eval_score / eval_score.sum()
    cum = np.cumsum(p[order])
    return {k: float(cum[k - 1]) for k in KS}, float(1.0 / np.sum(p * p))


def analyse(arms, init_step, audit_step, extract_dir, selection_dir, draws, shuffles, seed):
    index = json.loads((extract_dir / 'index.json').read_text())
    groups, splits = np.array(index['groups']), np.array(index['splits'])
    disc = (groups == 'math') & (splits == 'discovery')
    held = (groups == 'math') & (splits == 'heldout')
    lists = json.loads((selection_dir / 'lists.json').read_text())['primary']
    bins_all = json.loads((selection_dir / 'bins.json').read_text())
    names = [a for a in ARM_ORDER if a in arms]
    layers = sorted(int(l) for l in bins_all)
    rng = np.random.Generator(np.random.PCG64(seed))
    pair_rows, conc_rows, validation = [], [], {}
    for layer in layers:
        bins = {int(b): np.array(v) for b, v in bins_all[str(layer)].items()}
        labels = np.full(8960, -1)
        for b, v in bins.items():
            labels[v] = b
        k = key(layer, 'down_proj', 'A')
        a0 = arms[names[0]][init_step]['tensors'][k].ravel().astype(np.float64)
        S = {}
        for n in names:
            aT = arms[n][audit_step]['tensors'][k].ravel().astype(np.float64)
            Z = np.load(extract_dir / n / f'mean_abs_z_layer{layer:02d}.npy').astype(np.float64)
            Zd, Zh = Z[disc].mean(0), Z[held].mean(0)
            da = aT - a0
            S[n] = dict(aT=aT, da=da,
                        disc=dict(B=np.abs(aT) * Zd, B_learned=np.abs(da) * Zd, B_init=np.abs(a0) * Zd,
                                  activity=Zd, abs_a=np.abs(aT), abs_da=np.abs(da)),
                        held=dict(B=np.abs(aT) * Zh, B_learned=np.abs(da) * Zh, B_init=np.abs(a0) * Zh,
                                  activity=Zh, abs_a=np.abs(aT), abs_da=np.abs(da)))
            frozen = lists[f'B/{n}'][str(layer)]
            recomputed = S[n]['disc']['B'][frozen['indices']]
            stored = np.asarray(frozen['scores'])
            if layer in (0, 15, 27):
                T = np.load(extract_dir / n / f'mean_abs_term_layer{layer:02d}.npy').astype(np.float64)
                term_err = float(np.max(np.abs(T[held].mean(0) - S[n]['held']['B']) / (S[n]['held']['B'] + 1e-12)))
            else:
                term_err = None
            validation[f'{n}/{layer}'] = dict(
                frozen_score_max_rel_err=float(np.max(np.abs(recomputed - stored) / stored)),
                frozen_set_reproduced=sorted(top(S[n]['disc']['B']).tolist()) == sorted(frozen['indices']),
                heldout_term_max_rel_err=term_err)
            # Stage 3: concentration with shuffled-|a| nulls (activity kept)
            for v in VERSIONS:
                C, neff = concentration(S[n]['disc'][v], S[n]['held'][v])
                row = dict(arm=n, layer=layer, version=v, C=C, n_eff=neff)
                if v in ('B', 'B_learned'):
                    w = np.abs(aT) if v == 'B' else np.abs(da)
                    nullC, nullN = {kk: [] for kk in KS}, []
                    for _ in range(shuffles):
                        ws = rng.permutation(w)
                        c, ne = concentration(ws * Zd, ws * Zh)
                        for kk in KS:
                            nullC[kk].append(c[kk])
                        nullN.append(ne)
                    row['null'] = dict(C={kk: [float(np.percentile(nullC[kk], q)) for q in (2.5, 50, 97.5)] for kk in KS},
                                       n_eff=[float(np.percentile(nullN, q)) for q in (2.5, 50, 97.5)])
                conc_rows.append(row)
        # Stage 2: pairs
        for x, y in itertools.combinations(names, 2):
            row = dict(pair=f'{x}|{y}', layer=layer)
            sets = dict(B_frozen=(np.array(lists[f'B/{x}'][str(layer)]['indices']),
                                  np.array(lists[f'B/{y}'][str(layer)]['indices'])))
            for v in ('B_learned', 'abs_da', 'abs_a', 'activity'):
                sets[v] = (top(S[x]['disc'][v]), top(S[y]['disc'][v]))
            for v, (X, Y) in sets.items():
                mask = np.zeros(8960, bool)
                mask[X] = True
                obs = len(set(X.tolist()) & set(Y.tolist())) / len(Y)
                null = binned_null(mask, Y, labels, bins, draws, rng)
                row[v] = dict(observed=obs, null_median=float(np.median(null)),
                              null_p97_5=float(np.percentile(null, 97.5)))
            row['spearman_B_heldout'] = spearman(S[x]['held']['B'], S[y]['held']['B'])
            row['spearman_B_learned_heldout'] = spearman(S[x]['held']['B_learned'], S[y]['held']['B_learned'])
            pair_rows.append(row)
        print(json.dumps(dict(kind='layer_done', layer=layer)), flush=True)
    return names, layers, pair_rows, conc_rows, validation


def summarise(names, pair_rows, conc_rows):
    out = dict(pairs={}, arms={})
    for p in sorted({r['pair'] for r in pair_rows}):
        sel = [r for r in pair_rows if r['pair'] == p]
        out['pairs'][p] = {v: dict(observed=float(np.mean([r[v]['observed'] for r in sel])),
                                   null=float(np.mean([r[v]['null_median'] for r in sel])),
                                   layers_above_p97_5=int(sum(r[v]['observed'] > r[v]['null_p97_5'] for r in sel)))
                           for v in ('B_frozen', 'B_learned', 'abs_da', 'abs_a', 'activity')}
        out['pairs'][p]['spearman_B_heldout'] = float(np.median([r['spearman_B_heldout'] for r in sel]))
        out['pairs'][p]['spearman_B_learned_heldout'] = float(np.median([r['spearman_B_learned_heldout'] for r in sel]))
    for n in names:
        out['arms'][n] = {}
        for v in VERSIONS:
            sel = [r for r in conc_rows if r['arm'] == n and r['version'] == v]
            e = dict(n_eff_median=float(np.median([r['n_eff'] for r in sel])),
                     C64_median=float(np.median([r['C'][64] for r in sel])))
            if 'null' in sel[0]:
                e['null_n_eff_median'] = float(np.median([r['null']['n_eff'][1] for r in sel]))
                e['null_C64_median'] = float(np.median([r['null']['C'][64][1] for r in sel]))
                e['layers_C64_above_null_p97_5'] = int(sum(r['C'][64] > r['null']['C'][64][2] for r in sel))
                e['layers_n_eff_below_null_p2_5'] = int(sum(r['n_eff'] < r['null']['n_eff'][0] for r in sel))
            out['arms'][n][v] = e
    return out


def plot(names, layers, pair_rows, conc_rows, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#999999'})
    k = len(names)
    groups = {'sampling-SFT vs sampling-SFT': lambda x, y: family(x) == family(y) == 'sampling-sft',
              'online vs offline GRPO': lambda x, y: family(x) == family(y) == 'grpo',
              'across methods': lambda x, y: family(x) != family(y)}
    gcol = {'sampling-SFT vs sampling-SFT': '#3c9a5f', 'online vs offline GRPO': '#d0572e', 'across methods': '#6b6b6b'}

    # S2a: 6x6 matrices of overlap / null (equal-layer means)
    panels = [('B_frozen', 'Frozen B lists\n(|a| × activity)'), ('B_learned', 'Learned read\n(|Δa| × activity)'),
              ('abs_da', '|Δa| alone'), ('activity', 'Activity alone')]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for ax, (v, title) in zip(axes, panels):
        M = np.full((k, k), np.nan)
        for i, j in itertools.combinations(range(k), 2):
            sel = [r[v] for r in pair_rows if r['pair'] == f'{names[i]}|{names[j]}']
            M[i, j] = M[j, i] = np.mean([s['observed'] for s in sel])
        im = ax.imshow(M, cmap='Greens', vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                if i != j:
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if M[i, j] > 0.6 else '#222222')
        nullv = np.mean([r[v]['null_median'] for r in pair_rows])
        ax.set_title(f'{title}\n(bin-matched null ≈ {nullv:.2f})', fontsize=10)
        ax.set_xticks(range(k), [short(n) for n in names], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(k), [short(n) for n in names] if ax is axes[0] else [], fontsize=8)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label='top-64 overlap fraction (mean over 28 layers)')
    fig.suptitle('Do two arms read the same 64 neurons? (down_proj; ranked on discovery math)', y=1.1)
    fig.savefig(fig_dir / 'S2a_overlap_matrices.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S2b: by layer, overlap minus null, pair groups
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, (v, title) in zip(axes, panels[:3]):
        for g, test in groups.items():
            ys = [np.median([r[v]['observed'] - r[v]['null_median'] for r in pair_rows
                             if r['layer'] == l and test(*r['pair'].split('|'))]) for l in layers]
            ax.plot(layers, ys, color=gcol[g], lw=2, label=g if ax is axes[0] else None)
        ax.axhline(0, color='#999999', lw=0.8)
        ax.set_title(title.replace('\n', ' '), fontsize=10)
        ax.set_xlabel('layer')
        ax.grid(True, color='#e6e6e6', lw=0.6)
    axes[0].set_ylabel('top-64 overlap minus null')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Shared read neurons beyond the activity-matched null, by layer', y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S2b_overlap_by_layer.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S3a: N_eff by layer, B and B_learned, with shuffled-|a| null band
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, v, title in [(axes[0], 'B', 'Read mass B = |a| × activity'),
                         (axes[1], 'B_learned', 'Learned read mass |Δa| × activity')]:
        for n in names:
            sel = sorted((r for r in conc_rows if r['arm'] == n and r['version'] == v), key=lambda r: r['layer'])
            ax.plot(layers, [r['n_eff'] for r in sel], color=ARM_COLOURS.get(n), lw=2, label=n)
            ax.fill_between(layers, [r['null']['n_eff'][0] for r in sel], [r['null']['n_eff'][2] for r in sel],
                            color=ARM_COLOURS.get(n), alpha=0.12, lw=0)
        act = sorted((r for r in conc_rows if r['arm'] == names[0] and r['version'] == 'activity'),
                     key=lambda r: r['layer'])
        ax.plot(layers, [r['n_eff'] for r in act], color='#222222', lw=1.2, ls='--', label='activity alone (gold SFT)')
        ax.set_yscale('log')
        ax.set_title(f'{title}\nshaded: same arm with |a| shuffled across neurons (95%)', fontsize=10)
        ax.set_xlabel('layer')
        ax.grid(True, color='#e6e6e6', lw=0.6)
    axes[0].set_ylabel('effective number of neurons  1/Σp²  (of 8,960)')
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S3a_effective_neurons.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S3b: C(K) curves at two layers, versions, for gold SFT and online GRPO
    vcol = {'B': '#2a78b5', 'B_learned': '#d0572e', 'B_init': '#8a8a8a', 'activity': '#222222'}
    vlab = {'B': 'B = |a|·activity', 'B_learned': '|Δa|·activity', 'B_init': '|a0|·activity (untrained read)',
            'activity': 'activity alone'}
    show_layers = [15, 21]
    show_arms = [a for a in ('gold-sft', 'online-grpo') if a in names]
    fig, axes = plt.subplots(len(show_arms), len(show_layers), figsize=(12, 4 * len(show_arms)), squeeze=False)
    for i, n in enumerate(show_arms):
        for j, l in enumerate(show_layers):
            ax = axes[i][j]
            for v in ('B', 'B_learned', 'B_init', 'activity'):
                r = next(r for r in conc_rows if r['arm'] == n and r['layer'] == l and r['version'] == v)
                ax.plot(KS, [r['C'][kk] for kk in KS], color=vcol[v], lw=2, marker='o', ms=4, label=vlab[v])
                if v == 'B':
                    ax.fill_between(KS, [r['null']['C'][kk][0] for kk in KS], [r['null']['C'][kk][2] for kk in KS],
                                    color=vcol[v], alpha=0.15, lw=0, label='B with |a| shuffled (95%)')
            ax.plot(KS, [kk / 8960 for kk in KS], color='#c03030', lw=1, ls=':', label='uniform (K/8,960)')
            ax.set_xscale('log')
            ax.set_xticks(KS, [str(kk) for kk in KS])
            ax.set_ylim(0, 1)
            ax.set_title(f'{n}, layer {l}', fontsize=10)
            ax.set_xlabel('K (top neurons, ranked on discovery)')
            ax.grid(True, color='#e6e6e6', lw=0.6)
            if j == 0:
                ax.set_ylabel('C(K): held-out share of read mass')
    axes[0][0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(fig_dir / 'S3b_CK_curves.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # S3c: excess C(64) over the shuffled null, by layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, v, title in [(axes[0], 'B', 'B'), (axes[1], 'B_learned', '|Δa| × activity')]:
        for n in names:
            sel = sorted((r for r in conc_rows if r['arm'] == n and r['version'] == v), key=lambda r: r['layer'])
            ax.plot(layers, [r['C'][64] - r['null']['C'][64][1] for r in sel], color=ARM_COLOURS.get(n), lw=2,
                    label=n)
        ax.axhline(0, color='#999999', lw=0.8)
        ax.set_title(f'{title}: C(64) minus shuffled-|a| null median', fontsize=10)
        ax.set_xlabel('layer')
        ax.grid(True, color='#e6e6e6', lw=0.6)
    axes[0].set_ylabel('extra share of read mass in top 64')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'S3c_excess_C64.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--extract_dir', required=True)
    p.add_argument('--selection_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--init_step', type=int, default=0)
    p.add_argument('--audit_step', type=int, default=64)
    p.add_argument('--draws', type=int, default=200)
    p.add_argument('--shuffles', type=int, default=200)
    p.add_argument('--seed', type=int, default=20261112)
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    root = Path(a.artifact_root)
    inv_dir = Path(a.inventory) if Path(a.inventory).is_absolute() else root / a.inventory
    extract_dir, selection_dir, out = Path(a.extract_dir), Path(a.selection_dir), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    config = dict(inventory_sha256=sha256(inv_dir / 'inventory.json'),
                  extract_status_sha256=sha256(extract_dir / 'status.json'),
                  lists_sha256=sha256(selection_dir / 'lists.json'), bins_sha256=sha256(selection_dir / 'bins.json'),
                  init_step=a.init_step, audit_step=a.audit_step, draws=a.draws, shuffles=a.shuffles, seed=a.seed,
                  top=TOP, ks=KS)
    if status_path.exists() and not a.force:
        old = json.loads(status_path.read_text())
        if old.get('complete') and old.get('config') == config:
            print('STAGE23 reused', status_path)
            return 0
    atomic_json(status_path, dict(complete=False, config=config, label='post hoc, exploratory'))
    arms, files = load_arms(root, inv_dir, a.init_step, a.audit_step)
    names, layers, pair_rows, conc_rows, validation = analyse(arms, a.init_step, a.audit_step, extract_dir,
                                                              selection_dir, a.draws, a.shuffles, a.seed)
    atomic_json(out / 'validation.json', validation)
    for name, rows in (('pairs.jsonl', pair_rows), ('concentration.jsonl', conc_rows)):
        tmp = out / (name + '.tmp')
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        os.replace(tmp, out / name)
    atomic_json(out / 'summary.json', summarise(names, pair_rows, conc_rows))
    plot(names, layers, pair_rows, conc_rows, out / 'figures')
    atomic_json(status_path, dict(complete=True, config=config, files=files, label='post hoc, exploratory'))
    print('STAGE23 complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

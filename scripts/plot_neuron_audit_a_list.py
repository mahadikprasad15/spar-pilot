"""List-A exploration plots from the per-text extract (pilot_eval.neuron_audit_extract) + frozen 05 selection.

Reads only: <extract_dir>/{index.json,<arm>/mean_abs_z_layerNN.npy} and <selection_dir>/{lists.json,controls.json}.
Writes PNGs + figures.json into <output_dir>. Post hoc and exploratory: nothing here changes the sealed 05/06 runs.

  python scripts/plot_neuron_audit_a_list.py --extract_dir <extract-v1> --selection_dir <s-v1> \
      --output_dir <analysis-v1/figures/a-list> [--example_layers 0,14,20,27] [--resume]
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ['untuned', 'gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5']
COLORS = dict(zip(ARMS, ['#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
MATH, GEN, NB, CTRL = '#2a78d6', '#eb6834', '#1baf7a', '#a3a29c'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11.5,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'lines.linewidth': 2, 'legend.frameon': False,
})


def short(arm):
    return arm.replace('sampling-sft-', 'sft-')


class Data:
    def __init__(self, extract_dir, selection_dir):
        self.root = Path(extract_dir)
        self.index = json.loads((self.root / 'index.json').read_text())
        groups, splits = np.array(self.index['groups']), np.array(self.index['splits'])
        self.rows = {(g, s): np.where((groups == g) & (splits == s))[0] for g in set(groups) for s in set(splits)}
        sel = Path(selection_dir)
        self.lists = json.loads((sel / 'lists.json').read_text())['primary']
        self.controls = json.loads((sel / 'controls.json').read_text())
        self.layers = sorted(int(l) for l in self.lists['A'])
        self.arms = [a for a in ARMS if (self.root / a / 'mean_abs_z_layer00.npy').exists()]
        self._cache = {}

    def z(self, arm, layer):
        key = (arm, layer)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = np.load(self.root / arm / f'mean_abs_z_layer{layer:02d}.npy').astype(np.float64)
        return self._cache[key]

    def part(self, arm, layer, group, split):
        return self.z(arm, layer)[self.rows[(group, split)]]

    def a(self, layer):
        return self.lists['A'][str(layer)]

    def a_controls(self, layer):
        return [c['indices'] for c in self.controls['A'][str(layer)] if c.get('indices')]


def pooled_sd(x, y):
    n1, n2 = len(x), len(y)
    return np.sqrt(((n1 - 1) * x.var(axis=0, ddof=1) + (n2 - 1) * y.var(axis=0, ddof=1)) / (n1 + n2 - 2))


def gap(m, g, idx, standardized=False):
    d = m[:, idx].mean(axis=0) - g[:, idx].mean(axis=0)
    if standardized:
        d = d / pooled_sd(m[:, idx], g[:, idx])
    return float(np.mean(d))


# --- H1: how busy are neurons, layer by layer -----------------------------------------------------------

def h1_layer_profile(D, out):
    ps = [50, 90, 99]
    rows = {p: [] for p in ps}
    a_med = []
    for l in D.layers:
        busy = D.part('untuned', l, 'math', 'discovery').mean(axis=0)
        for p in ps:
            rows[p].append(np.percentile(busy, p))
        a_med.append(np.median(busy[D.a(l)['indices']]))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    shades = {50: '#9ec5f4', 90: '#3987e5', 99: '#184f95'}
    for p in ps:
        ax.plot(D.layers, rows[p], color=shades[p], label=f'{p}th percentile of all 8,960 neurons')
    ax.plot(D.layers, a_med, color=GEN, ls='--', marker='o', ms=3.5, label='median of the 64 A neurons')
    ax.set_yscale('log')
    ax.set(xlabel='layer', ylabel='neuron busyness on math: mean |z| (log)', xlim=(-0.5, 27.5))
    ax.set_title('H1a. How busy neurons are, layer by layer (untuned, discovery math texts)')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def h1_histograms(D, layers, out):
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.2), sharey=True)
    for ax, l in zip(np.atleast_1d(axes), layers):
        busy = D.part('untuned', l, 'math', 'discovery').mean(axis=0)
        bins = np.logspace(np.log10(max(busy.min(), 1e-5)), np.log10(busy.max()), 60)
        ax.hist(busy, bins=bins, color='#9ec5f4', edgecolor=SURFACE, linewidth=0.5, label='all neurons')
        gate = np.sort(busy)[::-1][D.a(l)['gate_size'] - 1]
        ax.axvline(gate, color=INK2, lw=1.2, ls='--', label='10% gate (A candidates to the right)')
        for v in busy[D.a(l)['indices']]:
            ax.axvline(v, ymin=0, ymax=0.06, color=GEN, lw=1)
        ax.plot([], [], color=GEN, lw=2, label='the 64 A neurons (ticks)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'layer {l}'); ax.set_xlabel('mean |z| on math (log)')
    np.atleast_1d(axes)[0].set_ylabel('number of neurons (log)')
    np.atleast_1d(axes)[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('H1b. Distribution of neuron busyness on math, untuned (one value per neuron)', x=0.01, ha='left',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# --- H2: the ranking step ------------------------------------------------------------------------------

def scores(D, layer):
    m, g = D.part('untuned', layer, 'math', 'discovery'), D.part('untuned', layer, 'general', 'discovery')
    sd = pooled_sd(m, g)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (m.mean(axis=0) - g.mean(axis=0)) / sd
    busy = m.mean(axis=0)
    gated = np.argsort(-busy, kind='stable')[:D.a(layer)['gate_size']]
    return s, gated


def h2_scores(D, layers, out, check):
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.2), sharey=True)
    for ax, l in zip(np.atleast_1d(axes), layers):
        s, gated = scores(D, l)
        a = D.a(l)
        recomputed = sorted(gated[np.argsort(-s[gated], kind='stable')][:a['size']].tolist())
        check[str(l)] = dict(matches_frozen_A=recomputed == sorted(a['indices']))
        bins = np.linspace(np.nanpercentile(s, 0.5), np.nanmax(s), 70)
        ax.hist(s[np.isfinite(s)], bins=bins, color='#cde2fb', edgecolor=SURFACE, linewidth=0.4,
                label='all neurons')
        ax.hist(s[gated], bins=bins, color='#3987e5', edgecolor=SURFACE, linewidth=0.4, label='gated 10% (busiest on math)')
        cut = min(s[a['indices']])
        ax.axvline(cut, color=GEN, lw=1.5, label='top-64 cut')
        ax.set_yscale('log'); ax.set_title(f'layer {l}')
        ax.set_xlabel('score: (math − chat) ÷ spread')
    np.atleast_1d(axes)[0].set_ylabel('number of neurons (log)')
    np.atleast_1d(axes)[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('H2. Math-vs-chat score of every neuron (discovery texts); A = top 64 of the gated 10%',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def h2_rank_curve(D, out):
    """Sorted scores of the gated neurons per layer: is there a natural elbow near 64?"""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    cmap = plt.get_cmap('Blues')
    for l in D.layers:
        s, gated = scores(D, l)
        v = np.sort(s[gated])[::-1][:300]
        ax.plot(np.arange(1, len(v) + 1), v, color=cmap(0.3 + 0.7 * l / 27), lw=1.2)
    ax.axvline(64, color=GEN, lw=1.5, ls='--')
    ax.text(66, ax.get_ylim()[1] * 0.95, 'top-64 cut', color=GEN, va='top', fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'b', [cmap(0.3), cmap(1.0)]), norm=plt.Normalize(0, 27))
    cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label('layer'); cb.outline.set_visible(False)
    ax.set(xlabel='rank among gated neurons', ylabel='score: (math − chat) ÷ spread', xlim=(0, 300))
    ax.set_title('H2b. Score by rank, one line per layer: is 64 a natural cut-off?')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# --- H3/H4: text by text, for single neurons -------------------------------------------------------------

def pick_neurons(D, layer):
    a = D.a(layer)
    order = np.argsort(a['scores'])[::-1]
    top = [a['indices'][i] for i in order[:3]]
    busy = D.part('untuned', layer, 'math', 'discovery').mean(axis=0)
    target = np.median(busy[top])
    pool = [j for j in np.argsort(np.abs(busy - target)) if j not in set(a['indices'])]
    return top, int(pool[0])


def h3_texts(D, layer, out, near_miss=False):
    top, ref = pick_neurons(D, layer)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    m = D.part('untuned', layer, 'math', 'heldout')
    g = D.part('untuned', layer, 'general', 'heldout')
    nb = D.part('untuned', layer, 'near_miss', 'heldout') if near_miss else None
    for ax, j, lab in zip(axes, top + [ref], ['A #1', 'A #2', 'A #3', 'random, same busyness']):
        series = [(m[:, j], MATH, 'math (100)'), (g[:, j], GEN, 'chat (100)')]
        if near_miss:
            series.append((nb[:, j], NB, 'numbers, not math (30)'))
        hi = max(s.max() for s, *_ in series)
        bins = np.linspace(0, hi, 30)
        for s, c, name in series:
            ax.hist(s, bins=bins, color=c, alpha=0.45, label=name, density=True)
            ax.axvline(s.mean(), color=c, lw=1.5)
        ax.set_title(f'{lab}: neuron {j}', fontsize=10.5)
        ax.set_xlabel('mean |z| on one text')
    axes[0].set_ylabel('share of texts (density)')
    axes[0].legend(loc='upper right', fontsize=8)
    name = 'H4' if near_miss else 'H3'
    what = 'math vs chat vs number-containing chat' if near_miss else 'math vs chat'
    fig.suptitle(f'{name}. Layer {layer}, held-out texts, {what}: each histogram is one neuron, one value per text '
                 '(lines = means)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# --- per-model A lines and the four normalizations ------------------------------------------------------

def arm_gaps(D):
    out = {a: dict(raw=[], ratio=[], std=[], std_ratio=[], median=[]) for a in D.arms}
    for l in D.layers:
        idx, ctrls = D.a(l)['indices'], D.a_controls(l)
        for arm in D.arms:
            m, g = D.part(arm, l, 'math', 'heldout'), D.part(arm, l, 'general', 'heldout')
            raw = gap(m, g, idx)
            craw = [gap(m, g, c) for c in ctrls]
            std = gap(m, g, idx, standardized=True)
            cstd = [gap(m, g, c, standardized=True) for c in ctrls]
            med = np.median(D.part(arm, l, 'math', 'heldout').mean(axis=0))
            out[arm]['raw'].append(raw)
            out[arm]['ratio'].append(raw / np.mean(craw) if craw else np.nan)
            out[arm]['std'].append(std)
            out[arm]['std_ratio'].append(std / np.mean(cstd) if cstd else np.nan)
            out[arm]['median'].append(raw / med)
            out[arm].setdefault('ctrl_raw', []).append(craw)
            out[arm].setdefault('ctrl_std', []).append(cstd)
    return out


def lines(D, G, key, out, title, ylabel, log=False, ref=None):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for arm in D.arms:
        ax.plot(D.layers, G[arm][key], color=COLORS[arm], label=short(arm),
                lw=2.6 if arm == 'untuned' else 1.6, zorder=3 if arm == 'untuned' else 2)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1, ls='--')
    if log:
        ax.set_yscale('log')
    ax.set(xlabel='layer', ylabel=ylabel, xlim=(-0.5, 27.5))
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def diff_lines(D, G, key, out, title, ylabel):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    base = np.array(G['untuned'][key])
    for arm in D.arms:
        if arm == 'untuned':
            continue
        ax.plot(D.layers, np.array(G[arm][key]) - base, color=COLORS[arm], label=short(arm), lw=1.6)
    ax.axhline(0, color=INK2, lw=1, ls='--')
    ax.set(xlabel='layer', ylabel=ylabel, xlim=(-0.5, 27.5))
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def four_ways(D, G, out):
    u = G['untuned']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    panels = [
        (axes[0, 0], 'raw', 'ctrl_raw', 'Raw gap (activation units)', 'math − chat, mean |z|'),
        (axes[0, 1], 'ratio', None, '÷ random lists\' gap (same layer, same busyness)', '× random'),
        (axes[1, 0], 'median', None, '÷ layer median busyness', 'gap ÷ median mean |z|'),
        (axes[1, 1], 'std', 'ctrl_std', 'Effect size: ÷ text-to-text spread, per neuron', 'standardized gap'),
    ]
    for ax, key, ckey, title, ylab in panels:
        ax.plot(D.layers, u[key], color=MATH, marker='o', ms=3.5, label='A (64 neurons)')
        if ckey:
            for l, cs in zip(D.layers, u[ckey]):
                ax.plot([l] * len(cs), cs, 'o', ms=3.5, color=CTRL, mec=SURFACE, mew=0.8)
            ax.plot([], [], 'o', color=CTRL, label='5 random lists')
        if key == 'ratio':
            ax.axhline(1, color=INK2, lw=1, ls='--')
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('layer')
    fig.suptitle('Same held-out math-vs-chat gap of list A (untuned), four ways', x=0.01, ha='left',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extract_dir', required=True)
    p.add_argument('--selection_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--example_layers', default='0,14,20,27')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    D = Data(args.extract_dir, args.selection_dir)
    ex = [int(x) for x in args.example_layers.split(',')]
    made, check = [], {}

    def run(name, fn):
        path = out / name
        if not (args.resume and path.exists()):
            fn(path)
        made.append(name)
        print('wrote', name, flush=True)

    run('H1a_busyness_by_layer.png', lambda p: h1_layer_profile(D, p))
    run('H1b_busyness_histograms.png', lambda p: h1_histograms(D, ex, p))
    run('H2a_score_histograms.png', lambda p: h2_scores(D, ex, p, check))
    run('H2b_score_by_rank.png', lambda p: h2_rank_curve(D, p))
    for l in ex:
        run(f'H3_texts_layer{l:02d}.png', lambda p, l=l: h3_texts(D, l, p))
        run(f'H4_texts_nearmiss_layer{l:02d}.png', lambda p, l=l: h3_texts(D, l, p, near_miss=True))
    G = arm_gaps(D)
    run('A1_gap_four_ways_untuned.png', lambda p: four_ways(D, G, p))
    run('A2_gap_raw_by_model.png', lambda p: lines(D, G, 'raw', p, 'A list held-out gap, every model (raw)',
                                                    'math − chat, mean |z|', log=True))
    run('A3_gap_vs_random_by_model.png', lambda p: lines(D, G, 'ratio', p,
                                                          'A list gap ÷ random lists\' gap, every model',
                                                          '× random (same neurons, same model)', ref=1))
    run('A4_gap_std_by_model.png', lambda p: lines(D, G, 'std', p, 'A list effect size, every model',
                                                    'standardized math − chat gap'))
    run('A5_gap_change_vs_untuned.png', lambda p: diff_lines(D, G, 'std', p,
                                                              'Change in A effect size after training (arm − untuned)',
                                                              'Δ standardized gap'))
    summary = dict(arms=D.arms, layers=D.layers, example_layers=ex, recompute_check=check,
                   gaps={a: {k: v for k, v in G[a].items() if not k.startswith('ctrl')} for a in D.arms},
                   extract_index_sha256=hashlib.sha256((Path(args.extract_dir) / 'index.json').read_bytes()).hexdigest(),
                   note='post hoc, exploratory; held-out rows; A lists and controls frozen from selection s-v1')
    (out / 'figures.json').write_text(json.dumps(dict(figures=made, **summary), indent=2))
    print('FIGURES', len(made), 'recompute_check', check)


if __name__ == '__main__':
    main()

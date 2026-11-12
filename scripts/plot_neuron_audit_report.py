"""Interpretation plots for a sealed neuron-audit 06 report (reads results.json only; never rewrites it).

Usage:
  python scripts/plot_neuron_audit_report.py --results_json <report_dir>/results.json --output_dir <out> [--resume]

Writes one PNG per figure plus figures.json (manifest with the source sha256) into <out>.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_ORDER = ['gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']  # validated categorical order, slots 1-6
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#a3a29c', '#e4e3df', '#fcfcfb'
SEQ = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'lines.linewidth': 2, 'legend.frameon': False,
})


def color(arm, arms):
    return SERIES[arms.index(arm)]


def short(arm):
    return arm.replace('sampling-sft-', 'sft-')


def controls(rec):
    return [c['value'] for c in rec.get('controls', []) if c.get('value') is not None]


def end_labels(ax, xs, series, arms, *, log=False):
    """Direct labels at the right end, nudged apart so they do not collide."""
    ys = [(np.log10(v[-1]) if log else v[-1], a) for a, v in series.items()]
    ys.sort()
    lo, hi = ax.get_ylim()
    span = (np.log10(hi) - np.log10(lo)) if log else (hi - lo)
    gap, placed = span * 0.055, []
    for y, a in ys:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(short(a), xy=(xs[-1], 10 ** y if log else y), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK2)
        ax.plot([xs[-1] + 0.15], [10 ** y if log else y], 's', ms=6, color=color(a, arms), clip_on=False)


def fig_a(r, layers, out):
    """Left: raw held-out gap (math minus general mean |z|) for A vs controls. Right: A gap / mean control gap."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    v = [r['layers'][l]['A']['value'] for l in layers]
    lo = [r['layers'][l]['A']['interval']['low'] for l in layers]
    hi = [r['layers'][l]['A']['interval']['high'] for l in layers]
    x = [int(l) for l in layers]
    ratio = []
    for xi, l, vi in zip(x, layers, v):
        cs = controls(r['layers'][l]['A'])
        ax.plot([xi] * len(cs), cs, 'o', ms=4.5, color=MUTED, mec=SURFACE, mew=1, zorder=2)
        ratio.append(vi / np.mean(cs) if cs else np.nan)
    ax.fill_between(x, lo, hi, color=SERIES[0], alpha=0.18, lw=0)
    ax.plot(x, v, color=SERIES[0], zorder=3, marker='o', ms=4)
    ax.set(xlabel='layer', ylabel='math − general, avg activation size |z|', xlim=(-0.5, 27.8))
    ax.set_title('How much more A neurons fire on math (held-out)')
    ax.plot([], [], color=SERIES[0], marker='o', ms=4, label='A list (64 neurons) + 95% bootstrap band')
    ax.plot([], [], 'o', color=MUTED, ms=4.5, label='5 random lists, same activity level')
    ax.legend(loc='upper left')
    ax2.bar(x, ratio, color=SERIES[0], width=0.7, edgecolor=SURFACE, linewidth=2)
    ax2.axhline(1, color=INK2, lw=1, ls='--')
    ax2.set(xlabel='layer', ylabel='A gap ÷ average random-list gap', xlim=(-0.8, 27.8))
    ax2.set_title('Same thing, relative to random neurons')
    ax2.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_d_size(r, layers, arms, out):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = [int(l) for l in layers]
    series = {}
    for a in arms:
        y = [r['layers'][l][f'D/{a}']['mean_abs_change'] for l in layers]
        series[a] = y
        ax.plot(x, y, color=color(a, arms), label=short(a))
    ax.set_yscale('log')
    ax.set(xlabel='layer', ylabel='mean |activity change| vs untuned (log scale)', xlim=(-0.5, 27))
    ax.set_title('D: how much each arm changed its most-changed neurons, by layer')
    ax.legend(loc='upper left', ncol=3)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_d_relative(r, layers, arms, out):
    """Each arm's D size divided by the median over arms at that layer: who is above / below typical."""
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = [int(l) for l in layers]
    mat = np.array([[r['layers'][l][f'D/{a}']['mean_abs_change'] for l in layers] for a in arms])
    med = np.median(mat, axis=0)
    series = {}
    for i, a in enumerate(arms):
        y = mat[i] / med
        series[a] = list(y)
        ax.plot(x, y, color=color(a, arms), label=short(a))
    ax.axhline(1, color=INK2, lw=1, ls='--')
    ax.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_yticks(ticks, [f'{t:g}×' for t in ticks]); ax.minorticks_off()
    ax.set(xlabel='layer', ylabel='D size ÷ median of the 6 arms (log scale)', xlim=(-0.5, 27))
    ax.set_title('D, relative: gold-SFT leads early/middle layers, offline-GRPO leads late layers')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=6)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_d_vs_d(r, arms, out):
    n = len(arms)
    obs, null = np.full((n, n), np.nan), np.full((n, n), np.nan)
    for i, a in enumerate(arms):
        for j, b in enumerate(arms):
            k = f'D/{a} vs D/{b}'
            if i != j and k in r['overlaps']:
                p = r['overlaps'][k]['pooled']
                obs[i, j], null[i, j] = p['observed'], p['reference']['median']
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', SEQ[1:])
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(obs, cmap=cmap, vmin=0, vmax=0.85)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', color=MUTED)
                continue
            v = obs[i, j]
            ax.text(j, i, f'{v:.2f}\n(chance {null[i, j]:.2f})', ha='center', va='center', fontsize=8,
                    color=SURFACE if v > 0.45 else INK)
    ax.set_xticks(range(n), [short(a) for a in arms], rotation=30, ha='right')
    ax.set_yticks(range(n), [short(a) for a in arms])
    ax.set_xlabel('arm whose list is randomized (Y)'); ax.set_ylabel('arm X')
    ax.set_title('D vs D: shared most-changed neurons')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('overlap |X∩Y| / |Y|, pooled over 28 layers'); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_d_vs_d_layers(r, layers, arms, out):
    pairs = [('sampling-sft-k0', 'sampling-sft-k2'), ('offline-grpo', 'online-grpo'), ('gold-sft', 'online-grpo')]
    names = ['sft-k0 vs sft-k2 (same family)', 'offline vs online GRPO (same family)', 'gold-SFT vs online GRPO (across)']
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = [int(l) for l in layers]
    for (a, b), nm, c in zip(pairs, names, [SERIES[0], SERIES[1], SERIES[2]]):
        L = r['overlaps'][f'D/{a} vs D/{b}']['layers']
        y = [L[l]['fraction'] for l in layers]
        lo = [L[l]['reference']['p2_5'] for l in layers]
        hi = [L[l]['reference']['p97_5'] for l in layers]
        ax.fill_between(x, lo, hi, color=MUTED, alpha=0.18, lw=0)
        ax.plot(x, y, color=c, label=nm, marker='o', ms=3.5)
    ax.fill_between([], [], [], color=MUTED, alpha=0.4, label='chance ranges (2.5–97.5%) for these pairs')
    ax.set(xlabel='layer', ylabel='overlap fraction (of 64)', ylim=(0, 1), xlim=(-0.5, 27.5))
    ax.set_title('D vs D by layer: same-family arms share far more than chance')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_primary(r, arms, out):
    rows = []
    for fam in ['A vs B/{a}', 'A vs D/{a}', 'B/{a} vs D/{a}']:
        for a in arms:
            k = fam.format(a=a)
            p = r['overlaps'][k]['pooled']
            rows.append((k.replace('sampling-sft-', 'sft-'), p['observed'], p['reference']['p2_5'],
                         p['reference']['p97_5'], color(a, arms)))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(rows))[::-1]
    for yi, (k, o, lo, hi, c) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=MUTED, lw=6, solid_capstyle='round', alpha=0.7)
        ax.plot(o, yi, 'o', ms=8, color=c, mec=SURFACE, mew=2, zorder=3)
    ax.set_yticks(y, [k for k, *_ in rows])
    for yb in [5.5, 11.5]:
        ax.axhline(yb, color=GRID, lw=1.5)
    ax.set_xlabel('pooled overlap fraction (equal-layer average)')
    ax.set_title('Do the three lists pick the same neurons?')
    ax.plot([], [], 'o', color=INK2, ms=8, label='observed')
    ax.plot([], [], color=MUTED, lw=6, label='chance range (2.5–97.5%, activity-matched)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.08), ncol=2)
    ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_d_cancel(r, layers, arms, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
    x = [int(l) for l in layers]
    for ax, a in zip(axes.flat, arms):
        recs = [r['layers'][l][f'D/{a}'] for l in layers]
        ax.plot(x, [d['mean_abs_change'] for d in recs], color=color(a, arms), label='list: mean |change|')
        ax.plot(x, [abs(d['mean_change']) for d in recs], color=color(a, arms), ls='--', lw=1.5,
                label='list: |signed mean change|')
        for xi, d in zip(x, recs):
            cs = controls(d)
            ax.plot([xi] * len(cs), cs, 'o', ms=3, color=MUTED, zorder=1)
        ax.set_yscale('log'); ax.set_title(short(a), fontsize=11)
    axes[0, 0].legend(loc='upper left', fontsize=8)
    axes[0, 0].plot([], [], 'o', ms=3, color=MUTED, label='controls: |signed mean change|')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('layer')
    for ax in axes[:, 0]:
        ax.set_ylabel('activity change (log)')
    fig.suptitle('Why some D-vs-control comparisons look weak: the signed mean cancels; no |change| exists for controls',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_b(r, layers, arms, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = [int(l) for l in layers]
    s1, s2 = {}, {}
    for a in arms:
        recs = [r['layers'][l][f'B/{a}'] for l in layers]
        y1 = [d['math'] / np.mean(controls(d)) if controls(d) else np.nan for d in recs]
        y2 = [d['math'] / d['general'] for d in recs]
        s1[a], s2[a] = y1, y2
        axes[0].plot(x, y1, color=color(a, arms), label=short(a))
        axes[1].plot(x, y2, color=color(a, arms), label=short(a))
    axes[0].axhline(1, color=INK2, lw=1, ls='--'); axes[1].axhline(1, color=INK2, lw=1, ls='--')
    axes[0].set(xlabel='layer', ylabel='list ÷ mean of controls', title='B vs random neurons (held-out math writes)')
    axes[1].set(xlabel='layer', ylabel='math ÷ general', title='B on math vs general text')
    axes[0].set_ylim(bottom=0); axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='upper right', ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_split(r, arms, out):
    sh = r['split_half']
    groups = [('A', ['A']), ('A-nb', ['A-nb'])] + [(f'{g}/{short(a)}', [f'{g}/{a}']) for g in ['B', 'D', 'D-relative'] for a in arms]
    labels, vals, cols = [], [], []
    for lab, (k,) in groups:
        vs = list(sh[k].values()) if isinstance(sh[k], dict) else sh[k]
        labels.append(lab); vals.append(float(np.mean(vs)))
        cols.append(SERIES[0] if lab.startswith('A') else color(next(a for a in arms if lab.endswith(short(a))), arms))
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.7, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f'{v:.2f}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, labels); ax.set_xlim(0, 1.08); ax.grid(axis='y', visible=False)
    ax.set_xlabel('overlap of the two half-data lists (mean over 28 layers)')
    ax.set_title('Split-half stability of each list')
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


FIGURES = [
    ('01_A_selectivity.png', fig_a, 'layers'),
    ('02_D_change_size.png', fig_d_size, 'layers_arms'),
    ('03_D_change_relative.png', fig_d_relative, 'layers_arms'),
    ('04_D_vs_D_overlap.png', fig_d_vs_d, 'arms'),
    ('05_D_vs_D_by_layer.png', fig_d_vs_d_layers, 'layers_arms'),
    ('06_list_overlaps_vs_chance.png', fig_primary, 'arms'),
    ('07_B_writes.png', fig_b, 'layers_arms'),
    ('08_split_half.png', fig_split, 'arms'),
    ('09_D_controls_caveat.png', fig_d_cancel, 'layers_arms'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_json', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--resume', action='store_true', help='skip figures that already exist')
    args = ap.parse_args()
    src = Path(args.results_json)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    r = json.loads(src.read_text())
    arms = [a for a in ARM_ORDER if a in r['arms']]
    layers = sorted(r['layers'], key=int)
    made = []
    for name, fn, kind in FIGURES:
        path = out / name
        if not (args.resume and path.exists()):
            if kind == 'layers':
                fn(r, layers, path)
            elif kind == 'arms':
                fn(r, arms, path)
            else:
                fn(r, layers, arms, path)
        made.append(name)
    (out / 'figures.json').write_text(json.dumps(dict(
        source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), label=r.get('label'),
        arms=arms, layers=len(layers), figures=made), indent=2))
    print('FIGURES', out, len(made))


if __name__ == '__main__':
    main()

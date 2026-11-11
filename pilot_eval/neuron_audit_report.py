"""Ticket 06: paired stability and held-out report for the joint neuron/adapter audit.

Inputs: a sealed 05 selection (lists, controls, overlaps, split halves), the sealed block units,
the frozen sequences and the 03b qualification summary (FP64 floors, flags, partial label).
Held-out rows are read here for the first time; lists are never re-ranked on them.

Held-out estimands per layer (equal-example weighted; token-weighted shown alongside for A):
- A: mean over list neurons of (held-out math - held-out general) base mean |z|;
- B (per arm): mean over list neurons of held-out mean |a_j z_j| on math and on general background;
- D (per arm): mean over list neurons of the held-out paired change (arm - base) mean |z| on math,
  and directional agreement with each neuron's signed discovery change (zero/undefined marked).
Each is compared with the same estimand on the list's frozen matched controls. Intervals are
percentile bootstraps over held-out questions/conversations (groups resampled separately; D keeps
the arm pairing). Overlap intervals from 05 are randomization references, not confidence intervals.
Everything is descriptive and exploratory (one training seed); no causal or behavioural claim.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from pilot_eval.neuron_audit_selection import RUN_SUBDIR as SEL_SUBDIR, Discovery, SelectionRefused

RUN_SUBDIR = 'runs/joint-neuron-adapter-audit/report'


class HeldOut(Discovery):
    split = 'heldout'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write(path, json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n')


def load_selection(run_dir):
    run_dir = Path(run_dir)
    seal = json.loads((run_dir / 'complete.json').read_text())
    for name, digest in seal['files'].items():
        if _sha(run_dir / name) != digest:
            raise SelectionRefused(f'selection file {name} failed its seal')
    return {name.split('.')[0]: json.loads((run_dir / name).read_text()) for name in seal['files']}


# --- estimands and bootstrap ----------------------------------------------------------------------

def bootstrap(stat, samples, *, resamples, seed):
    """Percentile 95% interval; ``samples`` maps group -> n, each group resampled separately (stratified)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    values = []
    for _ in range(resamples):
        draw = {g: rng.integers(0, n, size=n) for g, n in samples.items()}
        values.append(stat(draw))
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(status='undefined')
    return dict(status='ok', low=float(np.percentile(values, 2.5)), high=float(np.percentile(values, 97.5)),
                resamples=int(values.size), seed=seed, method='percentile, groups resampled separately')


def a_selectivity(math, background, neurons):
    return float((math[:, neurons].mean(axis=0) - background[:, neurons].mean(axis=0)).mean()) if neurons else float('nan')


def d_change(base, arm, neurons):
    return float((arm[:, neurons] - base[:, neurons]).mean()) if neurons else float('nan')


def sign_agreement(base, arm, neurons, discovery_signed):
    """Fraction of list neurons whose held-out signed change has the discovery sign; zeros reported separately."""
    held = (arm[:, neurons] - base[:, neurons]).mean(axis=0)
    disc = np.sign(np.asarray(discovery_signed, float))
    defined = (disc != 0) & (np.sign(held) != 0)
    return dict(agree=int((np.sign(held[defined]) == disc[defined]).sum()), defined=int(defined.sum()),
                zero_or_undefined=int((~defined).sum()),
                fraction=float((np.sign(held[defined]) == disc[defined]).mean()) if defined.any() else None)


def control_values(controls, fn):
    return [dict(index=c['index'], value=fn(c['indices'])) if c['status'] == 'ok'
            else dict(index=c['index'], value=None, status=c['status']) for c in controls]


def floor_fraction(d_scores, floor, multiple=5.0):
    if floor is None:
        return dict(status='unresolved', reason='floor unavailable or zero (never treated as zero)')
    scores = np.asarray(d_scores, float)
    return dict(status='ok', floor=floor, fraction_below=float((scores < multiple * floor).mean()) if scores.size else None,
                multiple=multiple, note='exploratory flag, not a gate')


# --- report -------------------------------------------------------------------------------------

def build(plan, *, selection_dir, store_roots, sequences, qualification, output_root, run_id, layers,
          sequences_sha256=None, prototype=False):
    run_dir = Path(output_root) / RUN_SUBDIR / run_id
    if (run_dir / 'complete.json').exists():
        seal = json.loads((run_dir / 'complete.json').read_text())
        if any(_sha(run_dir / n) != h for n, h in seal['files'].items()):
            raise SelectionRefused('sealed report failed its integrity check')
        return json.loads((run_dir / 'results.json').read_text())
    sel = load_selection(selection_dir)
    summary = sel['summary']
    if summary.get('prototype') and not prototype:
        raise SelectionRefused('selection is a prototype; a production report needs a production selection')
    arms = summary['arms']
    held = HeldOut(store_roots, sequences, sequences_sha256=sequences_sha256, prototype=prototype)
    ids = {g: held.ids(g) for g in ['math', 'general', 'near_miss']}
    boot = plan['statistics']['bootstrap']
    resamples, seed = int(boot['resamples']), int(boot['seed'])
    lists, controls = sel['lists']['primary'], sel['controls']
    results = dict(layers={}, arms=arms)
    for layer in layers:
        L = str(layer)
        base_math = held.matrix('untuned', layer, 'mean_abs_z', ids['math'])
        base_gen = held.matrix('untuned', layer, 'mean_abs_z', ids['general'])
        out = {}
        a = lists['A'][L]
        if a.get('status') == 'ok':
            neurons = a['indices']
            stat = lambda d: a_selectivity(base_math[d['math']], base_gen[d['general']], neurons)  # noqa: E731
            weights = (np.asarray(held.counts('untuned', ids['math']), float),
                       np.asarray(held.counts('untuned', ids['general']), float))
            token_weighted = float((np.average(base_math[:, neurons], axis=0, weights=weights[0]) -
                                    np.average(base_gen[:, neurons], axis=0, weights=weights[1])).mean())
            out['A'] = dict(value=a_selectivity(base_math, base_gen, neurons), token_weighted=token_weighted,
                            interval=bootstrap(stat, dict(math=len(ids['math']), general=len(ids['general'])),
                                               resamples=resamples, seed=seed),
                            controls=control_values(controls.get('A', {}).get(L, []),
                                                    lambda n: a_selectivity(base_math, base_gen, n)))
        else:
            out['A'] = dict(status=a.get('status'), reason=a.get('reason'))
        for arm in arms:
            arm_math = held.matrix(arm, layer, 'mean_abs_z', ids['math'])
            terms_math = held.matrix(arm, layer, 'mean_abs_term', ids['math'])
            terms_gen = held.matrix(arm, layer, 'mean_abs_term', ids['general'])
            b = lists[f'B/{arm}'][L]
            if b.get('status') == 'ok':
                n = b['indices']
                out[f'B/{arm}'] = dict(
                    math=float(terms_math[:, n].mean()), general=float(terms_gen[:, n].mean()),
                    interval_math=bootstrap(lambda d: float(terms_math[d['math']][:, n].mean()),
                                            dict(math=len(ids['math'])), resamples=resamples, seed=seed),
                    controls=control_values(controls.get(f'B/{arm}', {}).get(L, []),
                                            lambda c: float(terms_math[:, c].mean())))
            d = lists[f'D/{arm}'][L]
            if d.get('status') == 'ok':
                n = d['indices']
                out[f'D/{arm}'] = dict(
                    mean_change=d_change(base_math, arm_math, n),
                    interval=bootstrap(lambda s: d_change(base_math[s['math']], arm_math[s['math']], n),
                                       dict(math=len(ids['math'])), resamples=resamples, seed=seed),
                    sign_agreement=sign_agreement(base_math, arm_math, n, d['signed']),
                    controls=control_values(controls.get(f'D/{arm}', {}).get(L, []),
                                            lambda c: abs(d_change(base_math, arm_math, c))),
                    controls_note='controls compare |mean held-out change|; list value compared as |mean_change|')
            rel = lists.get(f'D-relative/{arm}', {}).get(L, {})
            bd = {}
            for name, entry in [('primary', d), ('relative', rel)]:
                if b.get('status') == 'ok' and entry.get('status') == 'ok':
                    bd[name] = len(set(b['indices']) & set(entry['indices']))
                else:
                    bd[name] = None
            out[f'B_intersect_D/{arm}'] = bd
            floors = (qualification.get('fp64_floors') or {}).get(arm, {}).get('layers', {}).get(L, {})
            out[f'D_floor/{arm}'] = floor_fraction(d.get('scores', []), floors.get('p99') if floors.get('status') ==
                                                   'measured' else None)
        results['layers'][L] = out
    results['overlaps'] = {k: dict(pooled=v['pooled']) for k, v in sel['overlaps'].items()}
    results['split_half'] = sel['split_half']['stability']
    results['coverage'] = dict(
        arms=arms, partial_label=qualification.get('partial_label'),
        heldout_counts={g: len(v) for g, v in ids.items()}, discovery_counts=summary['discovery_counts'],
        unresolved_floor_layers=qualification.get('unresolved_floor_layers'),
        review_flags_raised=qualification.get('review_flags_raised'))
    results['label'] = 'PROTOTYPE / REHEARSAL — not a result' if prototype else None
    results['limitations'] = LIMITATIONS
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / 'results.json', results)
    _write(run_dir / 'report.md', render(results))
    figures = _figures(results, run_dir)
    files = ['results.json', 'report.md'] + figures
    _write_json(run_dir / 'complete.json', dict(files={n: _sha(run_dir / n) for n in files},
                                                inputs=dict(selection=_sha(Path(selection_dir) / 'complete.json'),
                                                            created_unix=time.time())))
    return results


LIMITATIONS = [
    'Descriptive forward-pass statistics on fixed texts; no causal or behavioural conclusion follows.',
    'One training seed per arm; every comparison is exploratory and the number of comparisons is listed.',
    'Math (GSM8K reference answers) and background (UltraChat first turns) differ in source, style, length and '
    'authorship; the filters do not remove these confounds.',
    'Rank-one gauge: signed reads are comparable only within a checkpoint; cross-arm comparisons use magnitudes '
    'or physical writes.',
    'Overlap intervals are bin-conditional randomization references, not confidence intervals.',
    'The FP64 floor is calibration-sample dependent; "unresolved" floors are not zero.',
    'Leakage was checked only against available lineage files; missing sources mean unknown cleanliness.',
]


def _fmt(x):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.4g}'


def render(results):
    lines = ['# Joint neuron/adapter audit — Phase 1 report', '']
    if results.get('label'):
        lines += [f"**{results['label']}**", '']
    cov = results['coverage']
    if cov.get('partial_label'):
        lines += [f"**Partial study:** {cov['partial_label']}", '']
    lines += ['## Coverage', '', f"Arms: {', '.join(cov['arms'])}", f"Discovery counts: {cov['discovery_counts']}",
              f"Held-out counts: {cov['heldout_counts']}", f"Review flags raised in 03b: {cov['review_flags_raised']}", '',
              '## Primary overlap family (equal-layer pooled; randomization reference, not a CI)', '',
              '| comparison | observed | null median | null 2.5–97.5% | layers |', '|---|---|---|---|---|']
    for name, v in sorted(results['overlaps'].items()):
        p = v['pooled']
        if p.get('status') == 'ok':
            r = p['reference']
            lines.append(f"| {name} | {_fmt(p['observed'])} | {_fmt(r['median'])} | {_fmt(r['p2_5'])}–{_fmt(r['p97_5'])} | {p['layers']} |")
        else:
            lines.append(f'| {name} | undefined | | | 0 |')
    lines += ['', '## Held-out estimates by layer (frozen lists vs frozen matched controls)', '']
    for layer, out in results['layers'].items():
        lines += [f'### Layer {layer}', '']
        a = out.get('A', {})
        if 'value' in a:
            ctrl = [c['value'] for c in a['controls'] if c['value'] is not None]
            lines.append(f"- A selectivity {_fmt(a['value'])} (token-weighted {_fmt(a['token_weighted'])}), "
                         f"95% bootstrap {_fmt(a['interval'].get('low'))}–{_fmt(a['interval'].get('high'))}; "
                         f"controls {', '.join(_fmt(c) for c in ctrl) or 'unavailable'}")
        else:
            lines.append(f"- A: {a.get('status')} ({a.get('reason')})")
        for key, v in out.items():
            if key.startswith('D/'):
                agree = v['sign_agreement']
                lines.append(f"- {key}: mean held-out change {_fmt(v['mean_change'])} "
                             f"({_fmt(v['interval'].get('low'))}–{_fmt(v['interval'].get('high'))}); sign agreement "
                             f"{agree['agree']}/{agree['defined']} (zero/undefined {agree['zero_or_undefined']})")
            elif key.startswith('B/'):
                lines.append(f"- {key}: held-out math {_fmt(v['math'])}, background {_fmt(v['general'])}")
            elif key.startswith('B_intersect_D/'):
                lines.append(f"- {key}: primary {v['primary']}, relative {v['relative']}")
            elif key.startswith('D_floor/'):
                lines.append(f"- {key}: {v['status']}" + (f", fraction below 5x floor {_fmt(v['fraction_below'])}"
                                                           if v['status'] == 'ok' else ''))
        lines.append('')
    lines += ['## Split-half stability (overlap of half-lists)', '']
    for name, per_layer in sorted(results['split_half'].items()):
        values = list(per_layer.values())
        lines.append(f"- {name}: mean {_fmt(float(np.mean(values)) if values else None)} over {len(values)} layers")
    lines += ['', '## Limitations', ''] + [f'- {x}' for x in results['limitations']]
    return '\n'.join(lines) + '\n'


def _figures(results, run_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    names = sorted(results['overlaps'])
    if not names:
        return []
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 4))
    for i, name in enumerate(names):
        p = results['overlaps'][name]['pooled']
        if p.get('status') != 'ok':
            continue
        r = p['reference']
        ax.plot([i, i], [r['p2_5'], r['p97_5']], color='0.6', linewidth=6, solid_capstyle='butt')
        ax.plot(i, p['observed'], 'o', color='C0')
    ax.set_xticks(range(len(names)), names, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('equal-layer overlap fraction')
    ax.set_title('Observed overlap (dot) vs matched-null reference (bar)')
    fig.tight_layout()
    fig.savefig(run_dir / 'overlaps.png', dpi=120)
    plt.close(fig)
    return ['overlaps.png']


def main(argv=None):
    from pilot_eval import neuron_audit_plan as plan_mod
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--selection_dir', required=True)
    parser.add_argument('--stores', nargs='+', required=True)
    parser.add_argument('--sequences_run', required=True)
    parser.add_argument('--qualification_dir', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--layers', required=True)
    parser.add_argument('--rehearsal', action='store_true')
    args = parser.parse_args(argv)
    plan = plan_mod.load(args.plan, '03b' if args.rehearsal else '04', rehearsal=args.rehearsal)
    seq_dir = Path(args.sequences_run)
    sequences = [json.loads(l) for l in (seq_dir / 'sequences.jsonl').read_text().splitlines()]
    qual_dir = Path(args.qualification_dir)
    qualification = json.loads((qual_dir / 'qualification.json').read_text())
    qualification['fp64_floors'] = json.loads((qual_dir / 'stages/fp64_floor/result.json').read_text())['floors']
    lo, hi = (int(x) for x in args.layers.split('-'))
    results = build(plan, selection_dir=args.selection_dir, store_roots=args.stores, sequences=sequences,
                    qualification=qualification, output_root=args.output_root, run_id=args.run_id,
                    layers=list(range(lo, hi + 1)),
                    sequences_sha256=json.loads((seq_dir / 'manifest.json').read_text())['sequences_sha256'],
                    prototype=args.rehearsal)
    print(json.dumps(dict(run_id=args.run_id, layers=len(results['layers']), label=results.get('label')), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

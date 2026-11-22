"""C-attribution report, figures and catalog registration (ticket 13).

``build`` assembles ``report.json`` from the sealed artifacts only (the chain's report stage calls it on the Pod);
``markdown`` renders it. ``figures`` draws the figures from ``report.json`` alone and ``register`` writes the
catalog entries; both run locally after the export is restored (the Pod needs no matplotlib and cannot commit the
catalog)::

    python -m pilot_eval.c_attribution_report figures --report report.json --out figures/
    python -m pilot_eval.c_attribution_report register --root <dir holding c-attribution/> --label <label> \\
        --location local [--prefix imports/c-attribution-<label>]

Sign convention: positive dJdλ_c means slightly strengthening c is predicted to raise J (helpful, locally);
removal_J_c = -dJdλ_c (see ``SIGN_CONVENTION``)."""
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot_eval import c_attribution as ca
from pilot_eval import c_attribution_generation as cg

SCHEMA = 'c-attribution-report-v2'
REPO = Path(__file__).resolve().parents[1]
PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
HOW_TO_READ = [
    'Attribution targets sampled J; the behavioural check measures greedy accuracy. A greedy result neither '
    'confirms nor refutes the sampled-J gradient (a direct test needs sampled interventions, deferred).',
    'Removal is not local: removing an adapter or a group of reads is a finite intervention whose parts can '
    'interact. The sum of local predicted effects is a first-order reference, reported beside the measured outcome.',
    'Helpful and harmful are local, estimated words. Positive dJdλ predicts that slightly strengthening the '
    'component near λ = 1 raises J; negative dJdλ does not prove that full removal would raise accuracy.',
    'Same cohort: the 150 questions were used for selection and earlier evaluations. Every behavioural result is '
    'exploratory and never held-out.',
    'One seed per arm: each arm is scored on its own sampled responses, so difficulty weighting differs. No '
    'objective-specific or general mechanism can be claimed.',
]
CAVEAT = ('one seed per arm; each arm is scored on its own sampled responses with a different difficulty weighting; '
          'no mechanism claim')
LABEL_LINE = '_Labels: ' + '; '.join(cg.LABELS) + '._'


class ReportError(ValueError):
    pass


def _j(path):
    return json.loads(Path(path).read_text())


def _sealed(directory):
    from pilot_eval import attribution_units as au
    au.verify_run(directory)
    return Path(directory)


# --- report.json ------------------------------------------------------------------------------------

def _attribution(sub, arm, comparison):
    counts = _j(_sealed(sub / 'rows' / arm) / 'manifest.json')['counts']
    agg = _sealed(sub / 'aggregate' / arm)
    adapters = _j(agg / 'adapters.json')
    sealed = adapters['adapters']
    boot = (comparison.get('adapter_bootstrap') or {}).get('arms', {}).get(arm)
    rows = []
    for i, (d, r) in enumerate(zip(sealed['dJdlambda'], sealed['removal_J'])):
        layer, projection = sealed['names'][i]
        row = dict(index=i, layer=int(layer), projection=projection, dJdlambda=d, removal_J=r)
        if boot:
            row.update({'p2.5': boot['dJdlambda']['p2.5'][i], 'p97.5': boot['dJdlambda']['p97.5'][i],
                        'rank_p2.5': boot['rank']['p2.5'][i], 'rank_p97.5': boot['rank']['p97.5'][i]})
        rows.append(row)
    rows.sort(key=lambda r: (-r['dJdlambda'], r['index']))
    gross = sealed['totals']['gross']
    with np.load(agg / 'neurons.npz') as npz:
        reads = np.asarray(npz['dJdlambda'], dtype=np.float64)
    layer_boot = _j(agg / 'bootstrap.json')['neurons']['per_layer']
    per_layer = []
    for li in range(reads.shape[0]):
        d = reads[li]
        per_layer.append(dict(layer=li, dJdlambda_sum=float(d.sum()), gross=float(np.abs(d).sum()),
                              positive=int((d > 0).sum()), negative=int((d < 0).sum()),
                              quantiles={f'p{q}': float(np.percentile(d, q)) for q in (1, 10, 50, 90, 99)},
                              **{k: layer_boot[k][li] for k in layer_boot}))
    identity = _j(agg / 'identity.json')
    layers = identity.get('layers') or {}
    freeze = _j(_sealed(sub / 'freeze' / arm) / f'{arm}.json')
    return dict(
        J_bar=counts['mean_reward'], questions=counts['questions'], rows=counts['rows'], groups=counts['groups'],
        informative_questions=counts['groups']['mixed'], invalid=counts['invalid'], capped=counts['capped'],
        adapters=dict(rows=rows, totals=sealed['totals'],
                      top10_share_of_gross=(sum(abs(r['dJdlambda']) for r in rows[:10]) / gross) if gross else None,
                      share_denominator='gross Σ|dJdλ|; the net total is reported separately and never a denominator',
                      interval_label=(boot and comparison['adapter_bootstrap']['label'])),
        reads_per_layer=per_layer,
        identity=dict(max_relative_to_gross=max((v['relative_to_gross'] for v in layers.values()), default=None),
                      all_passed=all(v['passed'] for v in layers.values()), layers=layers),
        harmful_adapters_reported_only=freeze['harmful_adapters_reported_only'])


def _qualification(sub, arms):
    qdir = _sealed(sub / 'qualification')
    qual = _j(qdir / 'qualification.json')
    signoff = _j(_sealed(sub / 'signoff') / 'signoff.json')
    out = dict(checks=qual['checks'], review_flags=qual.get('review_flags', []), numerics=_j(qdir / 'numerics.json'),
               signoff={k: signoff[k] for k in ('by', 'decision', 'rule')}, arms={})
    for arm in arms:
        d = qual['details'][arm]
        fd = d['finite_difference']
        informative = [i['rel_err'] for i in fd['items'] if i.get('informative')]
        memory = d['memory']
        out['arms'][arm] = dict(
            rows=len(d['rows']), capped_rows_available=d['capped_rows_available'], capped_note=d.get('capped_note'),
            cached_vs_forced=dict(per_token_abs=d['cached_vs_forced']['per_token_abs'],
                                  per_row_abs_M=d['cached_vs_forced']['per_row_abs_M'],
                                  per_row=d['cached_vs_forced'].get('per_row'),
                                  rule=d.get('cached_vs_forced_rule', 'launch-record tolerance'),
                                  note=d['cached_vs_forced'].get('note')),
            replay=d['replay'],
            finite_difference=dict(passed=fd.get('passed'), items=len(fd['items']), informative=len(informative),
                                   median_rel_err=float(np.median(informative)) if informative else None,
                                   max_rel_err=float(np.max(informative)) if informative else None,
                                   floors=fd.get('floors'), epsilon=fd.get('epsilon'), reference='FP64'),
            batch_invariance={k: d['batch_invariance'][k] for k in ('passed', 'budgets', 'max_abs', 'max_fraction')},
            identity_max_relative=max((max(np.asarray(v['error']).ravel() / np.maximum(np.asarray(v['gross']).ravel(),
                                                                                          1e-300))
                                       for v in d['identity']), default=None) if d['identity'] else None,
            noop={k: d['noop'][k] for k in ('passed', 'logits_bit_identical', 'greedy_tokens_identical', 'restored')},
            processors=dict(passed=d['processors']['passed']),
            memory=dict(memory, fraction=(memory['peak_bytes'] / memory['total_bytes']) if memory['total_bytes'] else None))
    return out


def _provenance(sub, arm, qualification):
    manifest = _j(sub / 'rows' / arm / 'manifest.json')['provenance']
    verified = ['bundle file hashes', 'base-model files against HF revision pins', 'catalog verify of the bound entries',
                'per-row adapter hashes and sealed rows']
    if qualification['arms'][arm]['processors']['passed']:
        verified.append('logits processors resolved (check f)')
    return dict(recorded=sorted(manifest.get('recorded', {})), inferred=sorted(manifest.get('inferred', {})),
                verified=verified, inferred_detail=manifest.get('inferred', {}))


def _selection(freeze):
    out = {}
    for name, g in freeze['groups'].items():
        control = g['control']
        entry = dict(kind=g['kind'], size=len(g['targets']), shortfall=g['shortfall'], prediction=g.get('prediction'),
                     distribution=g.get('layer_distribution', g.get('module_distribution')),
                     control=dict(available=control['available'],
                                  size=len(control['group']) if control.get('group') else 0,
                                  match_ratio=control.get('match_ratio'), rule=control.get('rule'),
                                  tolerance=control.get('tolerance'), seed=control.get('seed'),
                                  largest_feasible_matching=control.get('largest_feasible_matching'),
                                  deficient_strata=control.get('deficient_strata') or [],
                                  deficient_targets=len(control.get('deficient_targets') or []),
                                  prediction=control.get('prediction'),
                                  distribution=control.get('layer_distribution', control.get('module_distribution'))))
        if name == 'hot64-reads':
            entry['hot_layers'] = freeze['hot_layers']
        out[name] = entry
    return out


def _comparison(sub, arms, comparison, freezes, stability):
    a, b = arms
    groups = {arm: freezes[arm]['groups'] for arm in arms}
    width = None
    boot = comparison.get('adapter_bootstrap') or {}
    paired = (boot.get('paired') or {}).get('dJdlambda')
    adapters = {arm: _j(sub / 'aggregate' / arm / 'adapters.json')['adapters'] for arm in arms}
    paired_rows = []
    for i, (da, db) in enumerate(zip(adapters[a]['dJdlambda'], adapters[b]['dJdlambda'])):
        layer, projection = adapters[a]['names'][i]
        row = dict(index=i, layer=int(layer), projection=projection, **{a: da, b: db}, difference=da - db)
        if paired:
            row.update({'p2.5': paired['p2.5'][i], 'p97.5': paired['p97.5'][i]})
        paired_rows.append(row)
    with np.load(sub / 'aggregate' / a / 'neurons.npz') as npz:
        width = npz['dJdlambda'].shape[1]
        n_layers = npz['dJdlambda'].shape[0]

    def by_layer(targets):
        out = {}
        for t in targets:
            out.setdefault(str(int(t) // width), set()).add(int(t))
        return out
    per_layer = [by_layer(groups[arm]['layer64-reads']['targets']) for arm in arms]
    return dict(
        adapters=dict(comparison['adapters'], paired_rows=paired_rows, difference=f'{a} - {b}',
                      top10_overlap=len(set(groups[a]['top10-adapters']['targets']) &
                                        set(groups[b]['top10-adapters']['targets'])),
                      interval_label=boot.get('label')),
        reads=dict(global_top64_overlap=len(set(groups[a]['global64-reads']['targets']) &
                                            set(groups[b]['global64-reads']['targets'])),
                   per_layer_top64_overlap={str(l): len(per_layer[0].get(str(l), set()) & per_layer[1].get(str(l), set()))
                                            for l in range(n_layers)}),
        distributions={arm: {name: g.get('layer_distribution', g.get('module_distribution'))
                             for name, g in groups[arm].items()} for arm in arms},
        stability={arm: {k: stability[arm][k]['adapters'] for k in ('question_split', 'draw_split')} for arm in arms},
        caveat=CAVEAT)


def build(root, record, *, arms, ablation, run_sha256):
    """report.json body from the sealed artifacts under ``root`` (the local root holding ``c-attribution/``)."""
    sub = Path(root) / 'c-attribution'
    comparison = _j(_sealed(sub / 'aggregate' / 'comparison') / 'comparison.json')
    freezes = {arm: _j(_sealed(sub / 'freeze' / arm) / f'{arm}.json') for arm in arms}
    stability = {arm: _j(sub / 'aggregate' / arm / 'split.json')['stability'] for arm in arms}
    qualification = _qualification(sub, arms)
    selected = {}
    for arm in arms:
        boot = _j(sub / 'aggregate' / arm / 'bootstrap.json')['selected']
        intervals = boot['intervals'] or {}
        selected[arm] = dict(label=boot['label'], reads=[dict(read, arm=arm, **{k: intervals[k][i] for k in intervals})
                                                         for i, read in enumerate(boot['reads'])])
    return dict(
        schema=SCHEMA, mode=record['mode'], label=record['label'], run_id=record['run_id'], arm_order=list(arms),
        launch_record_sha256=run_sha256,
        header=dict(how_to_read=HOW_TO_READ, sign_convention=ca.SIGN_CONVENTION, labels=list(cg.LABELS),
                    group_prediction_label=ca.GROUP_PREDICTION_LABEL),
        attribution={arm: _attribution(sub, arm, comparison) for arm in arms},
        stability=stability, qualification=qualification,
        provenance={arm: _provenance(sub, arm, qualification) for arm in arms},
        selection={arm: _selection(freezes[arm]) for arm in arms},
        arms=ablation, selected_intervals=selected,
        comparison=_comparison(sub, list(arms), comparison, freezes, stability),
        labels=list(cg.LABELS))


# --- report.md ----------------------------------------------------------------------------------------

def _f(x, digits=4):
    if x is None:
        return '—'
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    if isinstance(x, float):
        return f'{x:.{digits}g}'
    return str(x)


def _table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + \
        ['| ' + ' | '.join(_f(v) for v in row) + ' |' for row in rows]


def _interval(v):
    return f"{_f(v['estimate'])} [{_f(v['p2.5'])}, {_f(v['p97.5'])}]" if v else '—'


def markdown(body):
    title = ('REHEARSAL — machinery test; numbers are not results' if body['mode'] == 'rehearsal'
             else 'C-attribution pilot report')
    lines = [f'# {title}', '', f"Run `{body['run_id']}` ({body['label']}), run hash `{body['launch_record_sha256'][:16]}`.", '',
             '## How to read the results', '', *[f'- {s}' for s in body['header']['how_to_read']], '',
             '## Sign convention', '',
             *_table(['quantity', 'meaning', 'positive means'],
                     [[r['quantity'], r['meaning'], r['positive_means']] for r in body['header']['sign_convention']]), '']
    # attribution
    lines += ['## Attribution', '', LABEL_LINE, '']
    for arm, a in body['attribution'].items():
        t = a['adapters']['totals']
        lines += [f'### {arm}', '',
                  f"J̄ = {_f(a['J_bar'])} over {a['questions']} questions ({a['rows']} rows); informative (mixed) "
                  f"questions {a['informative_questions']}; groups {a['groups']}; invalid {a['invalid']}, capped {a['capped']}.",
                  f"Gross Σ|dJdλ| = {_f(t['gross'])}; net |Σ dJdλ| = {_f(t['net'])}; top-10 share of gross = "
                  f"{_f(a['adapters']['top10_share_of_gross'])} (denominator: {a['adapters']['share_denominator']}).",
                  f"Identity: max relative to gross {_f(a['identity']['max_relative_to_gross'])}; all layers passed: "
                  f"{a['identity']['all_passed']}.", '', 'Top 10 adapters by dJdλ (exploratory pointwise intervals):', '']
        lines += _table(['layer', 'module', 'dJdλ', 'p2.5', 'p97.5', 'rank p2.5–p97.5'],
                        [[r['layer'], r['projection'], r['dJdlambda'], r.get('p2.5'), r.get('p97.5'),
                          f"{_f(r.get('rank_p2.5'))}–{_f(r.get('rank_p97.5'))}"] for r in a['adapters']['rows'][:10]])
        lines += ['', 'Harmful adapters (most negative dJdλ; reported only, never selected):', '']
        lines += _table(['layer', 'module', 'dJdλ'], [[h['layer'], h['projection'], h['dJdlambda']]
                                                       for h in a['harmful_adapters_reported_only']])
        lines += ['', 'Neuron reads per down_proj layer:', '']
        lines += _table(['layer', 'Σ dJdλ', 'p2.5', 'p97.5', 'gross', 'positive', 'negative', 'median'],
                        [[r['layer'], r['dJdlambda_sum'], r.get('p2.5'), r.get('p97.5'), r['gross'], r['positive'],
                          r['negative'], r['quantiles']['p50']] for r in a['reads_per_layer']])
        lines.append('')
    # stability
    lines += ['## Stability (descriptive; never a gate)', '', LABEL_LINE, '']
    rows = []
    for arm, s in body['stability'].items():
        for split in ('question_split', 'draw_split'):
            ad, rd = s[split]['adapters'], s[split]['reads']
            rows.append([arm, split.replace('_', ' '), ad['spearman'], ad['top10_overlap'],
                         ad['sign_agreement_top20'].get('share'), rd['global'].get('overlap'),
                         rd['hot_layer_mean'].get('overlap'), rd['per_layer_mean'].get('overlap')])
    lines += _table(['arm', 'split', 'adapter Spearman', 'top-10 overlap', 'sign agreement (top 20)',
                     'global-64 overlap', 'hot-layer mean overlap', 'per-layer mean overlap'], rows)
    lines += ['', 'The draw split is noisier: 4 draws per question per half.', '']
    # qualification
    q = body['qualification']
    lines += ['## Qualification and numerics', '', LABEL_LINE, '',
              f"Sign-off: {q['signoff']['decision']} by {q['signoff']['by']} ({q['signoff']['rule']}).", '']
    lines += _table(['check', 'passed'], [[k, v] for k, v in sorted(q['checks'].items())])
    lines.append('')
    for arm, s in q['arms'].items():
        fd, cvf = s['finite_difference'], s['cached_vs_forced']
        lines += [f'### {arm}', '']
        lines += _table(['measure', 'value'], [
            ['finite differences vs FP64: median / max relative error (informative items)',
             f"{_f(fd['median_rel_err'])} / {_f(fd['max_rel_err'])} ({fd['informative']} of {fd['items']})"],
            ['FP32 floor ulp(|M|)/ε', (fd['floors'] or {}).get('fp32_ulp_M_over_eps')],
            ['cached vs forced, per token |Δ| median / p99 / max',
             f"{_f(cvf['per_token_abs'].get('median'))} / {_f(cvf['per_token_abs'].get('p99'))} / {_f(cvf['per_token_abs'].get('max'))}"],
            ['cached vs forced, per row |ΔM| median / max',
             f"{_f(cvf['per_row_abs_M'].get('median'))} / {_f(cvf['per_row_abs_M'].get('max'))} ({cvf['rule']})"],
            ['batch invariance max fraction of max |attr|', s['batch_invariance']['max_fraction']],
            ['neuron-sum identity max relative', s['identity_max_relative']],
            ['no-op interventions', s['noop']['passed']], ['logits processors', s['processors']['passed']],
            ['peak memory fraction', s['memory']['fraction']],
            ['replay (report-only)', s['replay']['status']],
            ['capped rows available', s['capped_rows_available']]])
        if cvf.get('per_row'):
            lines += ['', f'{arm}: cached vs forced by row (M in nats; |Δ| per token is the row\'s largest)', '']
            lines += _table(['row', 'generated tokens', 'M forced', 'M cached', '|ΔM|', 'max per-token |Δ|'],
                            [[r['id'], r['generated_length'], _f(r['M_forced'], 6), _f(r['M_cached'], 6),
                              _f(r['abs_delta_M']), _f(r['max_token_abs'])] for r in cvf['per_row']])
        prov = body['provenance'][arm]
        lines += ['', f"Provenance — verified: {', '.join(prov['verified'])}; recorded: {', '.join(prov['recorded'])}; "
                      f"inferred: {', '.join(prov['inferred']) or 'none'}.", '']
    # selection
    lines += ['## Selection', '', LABEL_LINE, '']
    for arm, groups in body['selection'].items():
        lines += [f'### {arm}', '']
        lines += _table(['group', 'size', 'shortfall', 'predicted ΔJ (pp)', 'control', 'control size',
                         'control predicted ΔJ (pp)', 'deficient targets'],
                        [[name, g['size'], _shortfall(g['shortfall']), (g['prediction'] or {}).get('pp'),
                          'available' if g['control']['available'] else 'unavailable', g['control']['size'],
                          (g['control']['prediction'] or {}).get('pp'), g['control']['deficient_targets']]
                         for name, g in groups.items()])
        hot = groups.get('hot64-reads', {}).get('hot_layers') or {}
        lines += ['', f"Hot layers {hot.get('layers')}: {hot.get('measurement', '')}, {hot.get('combine')} of each arm's "
                      f"top-{hot.get('k')} at step {hot.get('step')} (source sha256 {str((hot.get('source') or {}).get('sha256'))[:16]}).", '']
    # ablation
    lines += ['## Ablation (greedy)', '', LABEL_LINE, '']
    for arm, summary in body['arms'].items():
        lines += [f'### {arm}', '']
        rows = []
        for name, c in summary['conditions'].items():
            if c.get('status') != 'run':
                rows.append([name, 'skipped: ' + str(c.get('reason')), '', '', '', '', '', '', ''])
                continue
            rows.append([name, c['accuracy'], _interval(c.get('delta_pp_vs_intact')), c.get('percent_of_intact'),
                         _interval(c.get('target_minus_control_pp')), f"{c['gains']}/{c['losses']}",
                         c['mcnemar_exact_p'], f"{c['invalid']}/{c['capped']}",
                         (c.get('predicted_removal_effect') or {}).get('pp')])
        lines += _table(['condition', 'accuracy', 'Δpp vs intact [95%]', '% of intact', 'target − control pp [95%]',
                         'gains/losses', 'McNemar p', 'invalid/capped', 'predicted ΔJ (pp)'], rows)
        floor = summary.get('nondeterminism_floor')
        lines += ['', f"Nondeterminism floor (intact repeat): {floor['questions_differing']} questions differ, "
                      f"{floor['correctness_flips']} correctness flips." if floor else
                  'Nondeterminism floor: no intact repeat in this run.']
        if summary.get('untuned_note'):
            lines.append(f"Untuned: {summary['untuned_note']}.")
        lines.append('')
    # selected intervals
    lines += ['## Selected reads and matched controls', '', LABEL_LINE, '']
    for arm, block in body['selected_intervals'].items():
        lines += [f'### {arm} ({block["label"]})', '']
        lines += _table(['group', 'role', 'layer', 'neuron', 'p2.5', 'p97.5'],
                        [[r['group'], r['role'], r['layer'], r['j'], r.get('p2.5'), r.get('p97.5')] for r in block['reads']])
        lines.append('')
    # comparison
    c = body['comparison']
    lines += ['## Online against offline', '', LABEL_LINE, '', f"Caveat: {c['caveat']}.", '']
    lines += _table(['measure', 'value'], [
        ['adapter Spearman', c['adapters'].get('spearman')], ['top-10 adapter overlap', c['adapters']['top10_overlap']],
        ['sign agreement (top 20)', c['adapters'].get('sign_agreement_top20', {}).get('share')],
        ['global top-64 read overlap', c['reads']['global_top64_overlap']],
        ['mean per-layer top-64 read overlap', float(np.mean(list(c['reads']['per_layer_top64_overlap'].values())))]])
    lines += ['', f"Largest paired adapter differences ({c['adapters']['difference']}, paired intervals):", '']
    biggest = sorted(c['adapters']['paired_rows'], key=lambda r: -abs(r['difference']))[:10]
    lines += _table(['layer', 'module', 'difference', 'p2.5', 'p97.5'],
                    [[r['layer'], r['projection'], r['difference'], r.get('p2.5'), r.get('p97.5')] for r in biggest])
    lines.append('')
    return '\n'.join(lines)


def _shortfall(s):
    if isinstance(s, dict) and 'shortfall' in s:
        return s['shortfall']
    if isinstance(s, dict):
        return sum(v.get('shortfall', 0) for v in s.values())
    return s


# --- figures (local, from report.json only) -----------------------------------------------------------

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
ARM_COLOR = {'online-grpo': '#eb6834', 'offline-grpo': '#1baf7a', 'untuned': MUTED}
TARGET_COLOR, CONTROL_COLOR = '#2a78d6', '#86b6ef'


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'], 'font.size': 10, 'axes.edgecolor': AXIS,
        'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False, 'legend.labelcolor': INK2, 'lines.linewidth': 2})
    return plt


def figures(report_path, out_dir):
    """Every figure from ``report.json`` alone; writes PNGs and ``figures.json`` (file, title, sources, sha256)."""
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    diverging = LinearSegmentedColormap.from_list('bluegrayred', ['#a32a2a', '#e66767', '#f0efec', '#3987e5', '#104281'])
    body = _j(report_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    note = ' · '.join(cg.LABELS)
    title_prefix = 'REHEARSAL (not results) — ' if body['mode'] == 'rehearsal' else ''

    def save(fig, name, title, sources):
        path = out / name
        for ax in fig.axes[:1]:  # the labels sit under the title, clear of rotated tick labels
            ax.set_title(ax.get_title(loc='left'), loc='left', pad=16)
            ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=7, color=MUTED, ha='left', va='bottom')
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        index.append(dict(file=name, title=title, sources=sources,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    arms = body.get('arm_order') or list(body['attribution'])  # JSON key order is alphabetical, not the record's
    for arm in arms:  # layer x module heatmap of dJdλ, centred at 0 (blue = helpful locally)
        rows = body['attribution'][arm]['adapters']['rows']
        layers = max(r['layer'] for r in rows) + 1
        grid = np.full((layers, len(PROJECTIONS)), np.nan)
        for r in rows:
            grid[r['layer'], PROJECTIONS.index(r['projection'])] = r['dJdlambda']
        bound = float(np.nanmax(np.abs(grid))) or 1.0
        fig, ax = plt.subplots(figsize=(5.2, 7.2))
        im = ax.imshow(grid, cmap=diverging, norm=TwoSlopeNorm(0, -bound, bound), aspect='auto')
        ax.set_xticks(range(len(PROJECTIONS)), [p.replace('_proj', '') for p in PROJECTIONS])
        ax.set_ylabel('layer')
        ax.grid(False)
        ax.set_title(f'{title_prefix}{arm}: adapter dJdλ by layer and module')
        fig.colorbar(im, ax=ax, label='dJdλ (positive: helpful locally)', shrink=0.8)
        save(fig, f'adapters-heatmap-{arm}.png', f'{arm}: adapter dJdλ heatmap', [f'attribution.{arm}.adapters.rows'])
    if len(arms) == 2:
        a, b = arms
        rows = body['comparison']['adapters']['paired_rows']
        x, y = np.array([r[a] for r in rows]), np.array([r[b] for r in rows])
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        lim = float(max(np.abs(x).max(), np.abs(y).max())) * 1.08 or 1.0
        ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1, zorder=1)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        ax.scatter(x, y, s=14, color=TARGET_COLOR, alpha=0.75, edgecolor=SURFACE, linewidth=0.5, zorder=2)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=f'{a} dJdλ', ylabel=f'{b} dJdλ')
        ax.set_title(f'{title_prefix}adapter dJdλ: {a} against {b}')
        save(fig, 'adapters-online-vs-offline.png', 'online against offline adapter dJdλ',
             ['comparison.adapters.paired_rows'])
    fig, ax = plt.subplots(figsize=(10, 4.4))  # per-layer read-score distributions: p10–p90 bar, p1–p99 whisker
    width = 0.38
    for k, arm in enumerate(arms):
        rows = body['attribution'][arm]['reads_per_layer']
        pos = np.array([r['layer'] for r in rows]) + (k - 0.5) * width
        q = {p: np.array([r['quantiles'][p] for r in rows]) for p in ('p1', 'p10', 'p50', 'p90', 'p99')}
        ax.vlines(pos, q['p1'], q['p99'], color=ARM_COLOR.get(arm, INK2), lw=1, alpha=0.6)
        ax.bar(pos, q['p90'] - q['p10'], bottom=q['p10'], width=width * 0.85, color=ARM_COLOR.get(arm, INK2),
               alpha=0.85, label=arm)
        ax.scatter(pos, q['p50'], s=8, color=SURFACE, zorder=3)
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.set(xlabel='down_proj layer', ylabel='read dJdλ')
    ax.legend(loc='upper right')
    ax.set_title(f'{title_prefix}neuron-read dJdλ per layer (bar p10–p90, line p1–p99, dot median)')
    save(fig, 'reads-per-layer.png', 'per-layer read-score distributions', [f'attribution.{a}.reads_per_layer' for a in arms])
    for arm in arms:  # ablation: paired Δpp vs intact, each target with its control beside it
        conds = body['arms'][arm]['conditions']
        names = [n for n in cg.ORDER[1:-1] if n in conds] + [n for n in ('untuned',) if n in conds]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for i, name in enumerate(names):
            c = conds[name]
            control = name.startswith('match')
            color = MUTED if name == 'untuned' else (CONTROL_COLOR if control else TARGET_COLOR)
            if c.get('status') != 'run' or not c.get('delta_pp_vs_intact'):
                ax.text(i, 0, 'n/a', ha='center', va='bottom', color=MUTED, fontsize=8)
                continue
            d = c['delta_pp_vs_intact']
            ax.bar(i, d['estimate'], width=0.62, color=color)
            ax.errorbar(i, d['estimate'], yerr=[[d['estimate'] - d['p2.5']], [d['p97.5'] - d['estimate']]],
                        color=INK2, capsize=3, lw=1)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(range(len(names)), names, rotation=30, ha='right')
        ax.set_ylabel('Δ accuracy vs intact (pp), paired 95%')
        ax.set_title(f'{title_prefix}{arm}: greedy ablation (targets dark, matched controls light)')
        save(fig, f'ablation-{arm}.png', f'{arm}: ablation Δpp with controls', [f'arms.{arm}.conditions'])
    result = dict(report=str(report_path), report_sha256=hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
                  figures=index)
    (out / 'figures.json').write_text(json.dumps(result, indent=2) + '\n')
    return json.loads((out / 'figures.json').read_text())


# --- catalog registration (local, after the export is restored) ------------------------------------------

def register(root, catalog_dir, *, location, label, prefix=''):
    """One response-set entry per (arm, condition) greedy run and the untuned run; the new prior uses of
    cohort150 (attribution, selection and freeze, greedy ablation, the exact question IDs). ``root`` holds
    ``c-attribution/``; ``prefix`` is that directory's path relative to the ``location`` root."""
    import yaml
    from pilot_eval.attribution_units import verify_run
    root, catalog_dir = Path(root), Path(catalog_dir)
    sub = root / 'c-attribution'
    report = _j(_sealed(sub / 'report') / 'report.json')
    gen = sub / 'generation'
    written = dict(responses=[])
    entries = []
    for arm_dir in sorted(p for p in gen.iterdir() if p.is_dir()):
        for cond_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            batches = sorted(p for p in cond_dir.iterdir() if p.is_dir())
            files, config = [], None
            for batch in batches:
                verify_run(batch)
                path = batch / 'responses.jsonl'
                config = config or _j(batch / 'config.json')
                files.append(dict(location=location, path=str(Path(prefix) / path.relative_to(root)),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  rows_total=sum(1 for line in path.read_text().splitlines() if line)))
            arm, condition = arm_dir.name, cond_dir.name
            untuned = arm == 'untuned'
            rid = (f'untuned.greedy.test150.c-attribution-{label}' if untuned else
                   f'{arm}.s64.greedy.test150.c-attribution-{label}-{condition}')
            entries.append((rid, dict(
                id=rid, kind='evaluation', arm=arm, checkpoint_step=None if untuned else 64,
                cohort='gsm8k-test-cohort150', cohort_coverage='subset', question_field='question_id',
                draws_per_question=1,
                policy=dict(mode='greedy', repetition_penalty=1.0, max_new_tokens=config['cap'],
                            eos_token_id=config['eos_token_id']),
                scorer=dict(name='gsm8k-flexible-v3'),
                selection=dict(correctness_filtered=False, gold_access=False, all_groups_retained=True, best_of_n=False),
                c_suitability='greedy-only', replay_of=None,
                generator=('base model with the online-grpo adapter disabled (PEFT disable_adapter)' if untuned
                           else f'{arm} step-64 adapter with condition {condition}'),
                files=files,
                evidence=dict(status='verified', checked=['sealed run directories (hash-verified)', 'file sha256',
                                                          'row counts'],
                              note=f"run {report['run_id']} ({report['mode']}); one file per fixed greedy batch, "
                                   'together covering the frozen generation question set'),
                notes=(f"C-attribution {label}: greedy condition {condition}; launch hash "
                       f"{report['launch_record_sha256'][:16]}. Exploratory; components selected on these questions."))))
    for rid, _ in entries:
        if (catalog_dir / 'responses' / f'{rid}.yaml').exists():
            raise ReportError(f'{rid} is already registered; registration never overwrites')
    for rid, entry in entries:
        (catalog_dir / 'responses' / f'{rid}.yaml').write_text(yaml.safe_dump(entry, sort_keys=False, width=110))
        written['responses'].append(rid)
    freezes = {arm: sub / 'freeze' / arm / f'{arm}.json' for arm in report['attribution']}
    question_ids = sorted({q for arm in freezes for b in _j(freezes[arm])['generation_plan'] for q in b})
    # the questions each use actually touched (rehearsal finding 2026-11-22): rows.jsonl holds every question
    attribution_ids = sorted({q for arm in freezes for unit in sorted((sub / 'attribution' / arm).glob('unit-*'))
                              for q in _j(unit / 'meta.json')['question_ids']})
    qual = _j(_sealed(sub / 'qualification') / 'qualification.json')['details']
    qualification_ids = sorted({r.split(':draw')[0] for arm in qual for r in qual[arm]['rows']})
    cohorts_path = catalog_dir / 'cohorts.yaml'
    cohorts = yaml.safe_load(cohorts_path.read_text())
    uses = cohorts['cohorts']['gsm8k-test-cohort150'].get('prior_uses') or []
    freeze_hashes = ', '.join(f"{arm} {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}" for arm, p in freezes.items())
    new = [f"C-attribution {label} ({report['mode']}): qualification numerics (log-probabilities, finite "
           f"differences with leave-one-out reward weights) on sampled rows; questions {', '.join(qualification_ids)}",
           f"C-attribution {label} ({report['mode']}): attribution over sampled rows and selection, freeze "
           f"(sha256 {freeze_hashes}); questions {', '.join(attribution_ids)}",
           f"C-attribution {label} ({report['mode']}): greedy ablation of the frozen selection; questions "
           f"{', '.join(question_ids)}"]
    new = [u for u in new if u not in uses]
    cohorts_path.write_text(_append_prior_uses(cohorts_path.read_text(), 'gsm8k-test-cohort150', new))
    if yaml.safe_load(cohorts_path.read_text())['cohorts']['gsm8k-test-cohort150']['prior_uses'] != uses + new:
        raise ReportError('cohorts.yaml did not round-trip after appending prior uses')
    written['prior_uses'] = new
    return written


def _append_prior_uses(text, cohort, items):
    """Insert ``items`` at the end of ``cohort``'s ``prior_uses`` list, leaving every other line as written."""
    import yaml
    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.rstrip() == f'  {cohort}:')
    key = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith('prior_uses:'))
    indent = len(lines[key]) - len(lines[key].lstrip())
    end = key + 1
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent
                                or lines[end].lstrip().startswith('- ')):
        end += 1
    while lines[end - 1].strip() == '':
        end -= 1
    item_indent = next((len(l) - len(l.lstrip()) for l in lines[key + 1:end] if l.lstrip().startswith('- ')), indent + 2)
    block = yaml.safe_dump(list(items), width=110, allow_unicode=True)
    added = ''.join(' ' * item_indent + l + '\n' for l in block.splitlines())
    return ''.join(lines[:end]) + added + ''.join(lines[end:])


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='C-attribution report figures and catalog registration')
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('figures')
    f.add_argument('--report', required=True)
    f.add_argument('--out', required=True)
    r = sub.add_parser('register')
    r.add_argument('--root', required=True, help='directory holding c-attribution/ (the restored export)')
    r.add_argument('--label', required=True)
    r.add_argument('--location', default='local')
    r.add_argument('--prefix', default='', help="root's path relative to the location root")
    r.add_argument('--catalog', default='catalog')
    args = parser.parse_args(argv)
    if args.command == 'figures':
        print(json.dumps([x['file'] for x in figures(args.report, args.out)['figures']]))
        return 0
    print(json.dumps(register(args.root, args.catalog, location=args.location, label=args.label,
                              prefix=args.prefix), indent=2))
    print('next: python -m pilot_eval.catalog verify --root <location>=<path>, and commit only when it reports OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

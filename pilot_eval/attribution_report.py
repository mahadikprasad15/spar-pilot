"""Ticket 10: adapter-attribution report (spec §Statistics), pure numpy on sealed outputs or fixtures.

Unit: question. One set of seeded bootstrap indices per cohort is shared by every arm and condition (paired).
Intervals are percentile 95%; two-sided bootstrap p = 2·min(P(draw ≤ 0), P(draw ≥ 0)). Rows with an empty
stop mask are NaN and excluded per resample (never 0).

- **Validity gate** per arm × metric: sign agreement for the 5 top singles (≥ ``singles_min``) and the top-10
  group (prediction = sum of members, first-order); Spearman, sign agreement, median |error| and slope over
  top and matched singles with paired intervals. A limited check on selected removals.
- **Primary 1** (gold-sft): D_S, D_L, D_A and r = |D_L|/|D_A| from the same resampled questions; supported if
  D_S < 0 (interval), the reference D_A interval excludes 0 and the 95th percentile of r < margin;
  contradicted if the first two hold and the 5th percentile of r ≥ margin; otherwise inconclusive.
- **Primary 2**: Spearman (average ranks) on signed mean-attribution vectors over discovery; paired bootstrap
  of the four correlation differences; arms with an unresolved module excluded.
- **Primary 3**: R = (M_sparse − M_removed)/(M_full − M_removed) per order; unavailable when the interval of
  M_full − M_removed includes 0 or is non-finite; tests are R(attr) − R(size) and R(attr) − mean R(random);
  classification by interval bounds only and descriptive.
- **Holm** over each metric's family (primary 1's decision, primary 2's four differences, primary 3's two
  contrasts per arm). Concordance, concentration and cancellation are exploratory/descriptive.
"""
import json

import numpy as np

METRIC_KEY = {'trace': 'delta_trace', 'stop': 'delta_stop'}
GATE_LABEL = ('limited check on selected removals: a pass does not establish the ordering or magnitudes of all '
              '196 causal effects')
SCOPE = ("Scope: questions the untuned model solves under flexible v3 and pilot decoding, read as that model's "
         'own generated tokens; not all GSM8K questions and not newly acquired ability.')
INDEPENDENCE = ('Confirmation rows are partially independent: the joint audit computed descriptive activation '
                'statistics on these held-out rows and the user has seen that report; no quantity of this study '
                'was computed on them before the freeze.')


# --- bootstrap primitives ------------------------------------------------------------------------

def bootstrap_indices(n, *, resamples, seed):
    if not isinstance(resamples, int) or not isinstance(seed, int) or n < 1:
        raise ValueError('resamples and seed must be fixed integers (launch record); n >= 1')
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def _nanmean_rows(values, indices):
    v = np.asarray(values, dtype=np.float64)[indices]
    count = np.sum(~np.isnan(v), axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, np.nansum(v, axis=-1) / np.maximum(count, 1), np.nan)


def summarize(point, draws):
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if point is None or not np.isfinite(point) or not len(draws):
        return dict(estimate=None, interval_95=None, p_two_sided=None, reason='non-finite')
    p = min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
    return dict(estimate=float(point), interval_95=np.percentile(draws, [2.5, 97.5]).tolist(), p_two_sided=float(p),
                draws=len(draws))


def mean_interval(values, indices):
    v = np.asarray(values, dtype=np.float64)
    defined = ~np.isnan(v)
    if not defined.any():
        return dict(estimate=None, interval_95=None, p_two_sided=None, reason='no-defined-values',
                    n_defined=0, n_excluded=int(len(v)))
    out = summarize(float(v[defined].mean()), _nanmean_rows(v, indices))
    return dict(out, n_defined=int(defined.sum()), n_excluded=int((~defined).sum()))


def holm(pvalues, *, alpha):
    items = sorted((p, k) for k, p in pvalues.items() if p is not None)
    m, running, out = len(items), 0.0, {}
    for rank, (p, key) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = dict(p=p, adjusted=running, reject=running <= alpha)
    for key, p in pvalues.items():
        if p is None:
            out[key] = dict(p=None, adjusted=None, reject=False, reason='unavailable')
    return out


def _ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _slope(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


# --- validity gate ---------------------------------------------------------------------------------

def _predicted(freeze, index):
    for key in ('predicted', 'control_predicted'):
        if str(index) in freeze.get(key, {}):
            return float(freeze[key][str(index)])
    if 'predicted_all' in freeze:
        return float(freeze['predicted_all'][index])
    raise KeyError(index)


def validity_gate(stage, freeze, metric, indices, *, singles_min):
    key = METRIC_KEY[metric]
    points = []  # (name, predicted, values)
    agree = 0
    for r, target in enumerate(freeze['top5'], 1):
        values = stage[f'top{r}-{metric}'][key]
        pred, real = _predicted(freeze, target), mean_interval(values, indices)['estimate']
        agree += int(real is not None and np.sign(real) == np.sign(pred) and pred != 0)
        points.append((f'top{r}', pred, values))
    for r, target in enumerate(freeze['top5'], 1):
        control = freeze['controls']['per_target'].get(str(target))
        name = f'match{r}-{metric}'
        if control is not None and name in stage:
            points.append((f'match{r}', _predicted(freeze, control), stage[name][key]))
    group_pred = float(sum(_predicted(freeze, i) for i in freeze['top10']))
    group_real = mean_interval(stage[f'top10-{metric}'][key], indices)
    group_agree = group_real['estimate'] is not None and np.sign(group_real['estimate']) == np.sign(group_pred)
    preds = np.array([p for _, p, _ in points])
    real_point = np.array([np.nanmean(v) for _, _, v in points])
    real_draws = np.stack([_nanmean_rows(v, indices) for _, _, v in points], axis=1)

    def stats(real):
        rho = spearman(preds, real)
        return dict(spearman=np.nan if rho is None else rho, sign_agreement=float(np.mean(np.sign(real) == np.sign(preds))),
                    median_abs_error=float(np.median(np.abs(real - preds))), slope=_slope(preds, real))

    point = stats(real_point)
    draws = [stats(row) for row in real_draws]
    diagnostics = {k: dict(estimate=point[k], interval_95=np.nanpercentile([d[k] for d in draws], [2.5, 97.5]).tolist())
                   for k in point}
    passed = bool(agree >= singles_min and group_agree)
    return dict(metric=metric, top_singles_sign_agree=agree, singles_min=singles_min, group_sign_agree=bool(group_agree),
                group_prediction=group_pred, group_prediction_label='first-order (sum of members)',
                group_real=group_real, diagnostics=diagnostics, points=[n for n, _, _ in points], label=GATE_LABEL,
                **{'pass': passed})


def gate_consequences(gates, primary2_result, metric):
    def ok(arm):
        return gates.get((arm, metric), {}).get('pass') is True
    p2 = {k: ('confirmatory' if all(ok(a) for a in c.get('arms', [])) else 'unvalidated-exploratory')
          for k, c in primary2_result['contrasts'].items()}
    gold_ok = all(gates.get(('gold-sft', m), {}).get('pass') is True for m in METRIC_KEY
                  if ('gold-sft', m) in gates)
    return dict(primary2=p2,
                primary1='gate passed on selected removals' if gold_ok else 'rests only on directly measured contrasts',
                primary3='rests only on the measured 64-term intervention; attribution concentration is described as '
                         'attribution, not causal sparsity')


# --- primary 1 -------------------------------------------------------------------------------------

def primary1(stage, indices, *, margin):
    needed = ['top10-stop', 'match10-stop', 'top10-trace', 'match10-trace']
    missing = [n for n in needed if n not in stage]
    if missing:
        return dict(outcome='inconclusive', reasons=[f'conditions unavailable: {missing}'], p=None)

    def diff(cond, key):
        a, b = stage[cond][key], stage[cond.replace('top10', 'match10')][key]
        return float(np.nanmean(a) - np.nanmean(b)), _nanmean_rows(a, indices) - _nanmean_rows(b, indices)

    (ds, ds_d), (dl, dl_d), (da, da_d) = diff('top10-stop', 'delta_stop'), diff('top10-stop', 'delta_trace'), \
        diff('top10-trace', 'delta_trace')
    with np.errstate(divide='ignore', invalid='ignore'):
        r_point = abs(dl) / abs(da) if da != 0 else np.inf
        r_draws = np.abs(dl_d) / np.abs(da_d)
    s_ds, s_dl, s_da = summarize(ds, ds_d), summarize(dl, dl_d), summarize(da, da_d)
    finite = r_draws[np.isfinite(r_draws)]
    r_upper = float(np.percentile(finite, 95)) if len(finite) else np.inf
    r_lower = float(np.percentile(finite, 5)) if len(finite) else np.inf
    reference = s_da['interval_95'] is not None and (s_da['interval_95'][0] > 0 or s_da['interval_95'][1] < 0)
    stop_effect = s_ds['interval_95'] is not None and s_ds['interval_95'][1] < 0
    reasons = []
    if not reference:
        reasons.append('reference D_A interval includes 0 (or non-finite)')
    if not stop_effect:
        reasons.append('D_S interval does not exclude 0 below')
    if reference and stop_effect and r_upper < margin:
        outcome = 'supported'
    elif reference and stop_effect and r_lower >= margin:
        outcome = 'contradicted'
    else:
        outcome = 'inconclusive'
        if reference and stop_effect:
            reasons.append(f'r interval [{r_lower:.3g}, {r_upper:.3g}] straddles the margin {margin}')
    p_equiv = float(np.mean(r_draws >= margin)) if len(r_draws) else None
    components = [s_ds['p_two_sided'], s_da['p_two_sided'], p_equiv]
    p = None if any(c is None for c in components) else max(components)
    return dict(outcome=outcome, reasons=reasons, D_S=s_ds, D_L=s_dl, D_A=s_da, r_point=float(r_point),
                r_upper_95=r_upper, r_lower_95=r_lower, margin=margin, p=p,
                p_rule='larger of the component p-values (D_S, D_A two-sided; P(r >= margin))',
                note='Stage D can confirm or fail to confirm, not rescue, this teacher-forced result; '
                     'rho(attr_L, attr_S) and top-10 overlap are descriptive')


# --- primary 2 ---------------------------------------------------------------------------------------

CONTRASTS = dict(a=(('online-grpo', 'offline-grpo'), ('gold-sft', 'online-grpo')),
                 b=(('online-grpo', 'offline-grpo'), ('gold-sft', 'offline-grpo')),
                 c=(('sampling-sft-k0', 'gold-sft'), ('sampling-sft-k2', 'gold-sft')),
                 d=(('sampling-sft-k2', 'gold-sft'), ('sampling-sft-k5', 'gold-sft')))


def primary2(discovery, indices, *, excluded_arms):
    """``discovery``: arm -> per-question attribution matrix [n_questions, adapters] (same question order)."""
    means = {arm: np.nanmean(np.asarray(m, dtype=np.float64), axis=0) for arm, m in discovery.items()}
    contrasts, correlations = {}, {}
    for key, (first, second) in CONTRASTS.items():
        arms = sorted(set(first) | set(second))
        if any(a in excluded_arms or a not in discovery for a in arms):
            contrasts[key] = dict(estimate=None, interval_95=None, p_two_sided=None, arms=arms,
                                  reason='arm excluded (unresolved module) or missing')
            continue
        point = (spearman(means[first[0]], means[first[1]]) or np.nan) - \
            (spearman(means[second[0]], means[second[1]]) or np.nan)
        draws = []
        for row in indices:
            m = {a: np.nanmean(np.asarray(discovery[a], dtype=np.float64)[row], axis=0) for a in arms}
            r1, r2 = spearman(m[first[0]], m[first[1]]), spearman(m[second[0]], m[second[1]])
            draws.append(np.nan if r1 is None or r2 is None else r1 - r2)
        contrasts[key] = dict(summarize(point, draws), arms=arms, definition=f'rho{first} - rho{second}')
    names = sorted(means)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            correlations[f'{a}|{b}'] = spearman(means[a], means[b])
    support = {k: v.get('estimate') is not None and v['estimate'] > 0 for k, v in contrasts.items()}
    return dict(contrasts=contrasts, correlations=correlations, excluded_arms=sorted(excluded_arms),
                grpo_sharing_points_positive=support['a'] and support['b'],
                k_ordering_points_positive=support['c'] and support['d'],
                vectors='signed mean attribution over discovery; Spearman with average ranks')


# --- primary 3 -------------------------------------------------------------------------------------

ORDERS = ['attr', 'size', 'rand1', 'rand2', 'rand3']


def _removed_condition(stage, down, metric, top5):
    name = f'down-full-{metric}'
    if name in stage:
        return name
    if top5 is not None and down['index'] in top5:
        return f'top{top5.index(down["index"]) + 1}-{metric}'
    return None


def primary3(stage, down, metric, indices, *, sparse_lower, dense_upper, top5=None):
    key = METRIC_KEY[metric]
    removed = _removed_condition(stage, down, metric, top5)
    unavailable = dict(estimate=None, interval_95=None, p_two_sided=None)
    if removed is None or any(f'read64-{o}-{metric}' not in stage for o in ORDERS):
        reason = 'conditions-missing'
        return dict(R={o: dict(unavailable, reason=reason) for o in ORDERS}, classification='unavailable',
                    tests={t: dict(unavailable, reason=reason) for t in ('attr_minus_size', 'attr_minus_mean_random')},
                    classification_is_descriptive=True)
    rem = stage[removed][key]
    denominator_point = -float(np.nanmean(rem))
    denominator_draws = -_nanmean_rows(rem, indices)
    den = summarize(denominator_point, denominator_draws)
    available = den['interval_95'] is not None and (den['interval_95'][0] > 0 or den['interval_95'][1] < 0)
    if not available:
        reason = 'denominator-interval-includes-0'
        return dict(R={o: dict(unavailable, reason=reason) for o in ORDERS}, classification='unavailable',
                    denominator=den, removed_condition=removed,
                    tests={t: dict(unavailable, reason=reason) for t in ('attr_minus_size', 'attr_minus_mean_random')},
                    classification_is_descriptive=True)
    R_point, R_draws = {}, {}
    for order in ORDERS:
        values = stage[f'read64-{order}-{metric}'][key]
        R_point[order] = (float(np.nanmean(values)) - float(np.nanmean(rem))) / denominator_point
        R_draws[order] = (_nanmean_rows(values, indices) - _nanmean_rows(rem, indices)) / denominator_draws
    R = {o: summarize(R_point[o], R_draws[o]) for o in ORDERS}
    random_point = np.mean([R_point[o] for o in ORDERS[2:]])
    random_draws = np.mean([R_draws[o] for o in ORDERS[2:]], axis=0)
    tests = dict(attr_minus_size=summarize(R_point['attr'] - R_point['size'], R_draws['attr'] - R_draws['size']),
                 attr_minus_mean_random=summarize(R_point['attr'] - random_point, R_draws['attr'] - random_draws))
    low, high = R['attr']['interval_95']
    classification = 'sparse' if low >= sparse_lower else ('dense-candidate' if high < dense_upper else 'inconclusive')
    return dict(R=R, tests=tests, denominator=den, removed_condition=removed, classification=classification,
                classification_is_descriptive=True,
                classification_rule=f'sparse: lower 95% bound of R(attr) >= {sparse_lower}; dense: upper bound < '
                                    f'{dense_upper} in every arm (see dense_everywhere); never point estimates')


def dense_everywhere(results):
    return bool(results) and all(r['classification'] in ('dense-candidate',) for r in results.values())


# --- concordance and descriptive -----------------------------------------------------------------

def concordance(stage_c, stage_d, conf_indices, cohort_indices):
    from pilot_eval.stage0_findings_math import mcnemar
    rows, disagreements = {}, []
    for name, (full, cond) in stage_d.items():
        by = {r['id']: r for r in cond}
        ids = [r['id'] for r in full]
        if set(ids) != set(by):
            raise ValueError(f'{name}: generation items differ between full and condition')
        acc = np.array([float(by[i]['correct']) - float(r['correct']) for i, r in zip(ids, full)])
        length = np.array([float(by[i]['tokens']) - float(r['tokens']) for i, r in zip(ids, full)])
        dl = mean_interval(stage_c[name]['delta_trace'], conf_indices)
        ds = mean_interval(stage_c[name].get('delta_stop', np.full(1, np.nan)), conf_indices) \
            if 'delta_stop' in stage_c[name] else dict(estimate=None)
        da, dlen = mean_interval(acc, cohort_indices), mean_interval(length, cohort_indices)
        gains = int(sum(by[i]['correct'] and not r['correct'] for i, r in zip(ids, full)))
        losses = int(sum(r['correct'] and not by[i]['correct'] for i, r in zip(ids, full)))

        def agree(a, b, sign=1):
            if a['estimate'] is None or b['estimate'] is None:
                return None
            return bool(np.sign(a['estimate']) != 0 and np.sign(a['estimate']) == sign * np.sign(b['estimate']))

        row = dict(delta_trace=dl, delta_stop=ds, delta_accuracy=da, delta_length=dlen,
                   mcnemar_p=mcnemar(gains, losses), gains=gains, losses=losses,
                   trace_vs_accuracy=dict(agree=agree(dl, da), expected='same sign'),
                   stop_vs_length=dict(agree=agree(ds, dlen, -1), expected='higher stop pull, shorter responses'))
        rows[name] = row
        for check in ('trace_vs_accuracy', 'stop_vs_length'):
            if row[check]['agree'] is False:
                disagreements.append(dict(condition=name, check=check))
    return dict(conditions=rows, disagreements=disagreements,
                label='exploratory (predeclared concordance); behavioural claims only for generated arms and conditions')


def concentration(attr_j, *, null_orders, null_seed, resolution_floor):
    attr = np.asarray(attr_j, dtype=np.float64)
    total, signed = float(np.abs(attr).sum()), float(attr.sum())
    if not isinstance(null_seed, int) or not isinstance(resolution_floor, float):
        raise ValueError('null seed and resolution floor must come from the launch record / 06b rule')
    if total < resolution_floor:
        return dict(available=False, reason='total |attr| below the resolution floor', neurons_to_50=None,
                    neurons_to_90=None, cancellation=None, null=None)
    cumulative = np.cumsum(np.sort(np.abs(attr))[::-1]) / total
    first = lambda f: int(np.argmax(cumulative >= f - 1e-12) + 1)  # noqa: E731
    rng = np.random.default_rng(null_seed)
    null = np.mean([np.cumsum(np.abs(attr)[rng.permutation(len(attr))]) / total for _ in range(null_orders)], axis=0)
    return dict(available=True, neurons_to_50=first(0.5), neurons_to_90=first(0.9),
                cancellation=None if abs(signed) < resolution_floor else total / abs(signed),
                null=dict(orders=null_orders, seed=null_seed, mean_fraction=null.tolist(),
                          definition='expected fraction of total |attr| reached by m neurons in a uniformly random order'),
                label='attribution, not causal sparsity')


# --- assembly and rendering ------------------------------------------------------------------------

def _n(stage):
    for conditions in stage.values():
        for cond in conditions.values():
            for v in cond.values():
                return len(v)
    raise ValueError('no Stage C results')


def build(inputs):
    s = inputs['settings']
    stage_c = inputs['stage_c']
    first_arm = next(iter(stage_c.values()))
    conf = bootstrap_indices(_n(first_arm), resamples=s['resamples'], seed=s['seed'])
    report = dict(scope=SCOPE, independence=INDEPENDENCE, provenance=inputs['provenance'],
                  provenance_statement='no overlap with available sources', settings=s, gates={}, primary3={},
                  holm={}, consequences={})
    gates = {}
    for arm, metrics in stage_c.items():
        for metric, stage in metrics.items():
            gates[(arm, metric)] = validity_gate(stage, inputs['freeze'][arm][metric], metric, conf,
                                                 singles_min=s['gate_singles_min'])
    report['gates'] = {f'{a}|{m}': g for (a, m), g in gates.items()}
    gold = stage_c.get('gold-sft', {})
    merged = {**gold.get('trace', {}), **gold.get('stop', {})}
    report['primary1'] = primary1(merged, conf, margin=s['margin']) if gold else \
        dict(outcome='inconclusive', reasons=['gold-sft not measured'], p=None)
    report['primary2'] = {}
    for metric in METRIC_KEY:
        disc = inputs['discovery'][metric]
        n = len(next(iter(disc.values())))
        report['primary2'][metric] = primary2(disc, bootstrap_indices(n, resamples=s['resamples'], seed=s['seed']),
                                              excluded_arms=inputs.get('unresolved_arms', []))
        report['primary3'][metric] = {
            arm: primary3(stage_c[arm][metric], inputs['freeze'][arm][metric]['down_proj'], metric, conf,
                          sparse_lower=s['sparse_lower'], dense_upper=s['dense_upper'],
                          top5=inputs['freeze'][arm][metric].get('top5'))
            for arm in stage_c if metric in stage_c[arm]}
        report['primary3'][metric + '_dense_everywhere'] = dense_everywhere(report['primary3'][metric])
        family = {'primary1': report['primary1']['p']}
        family.update({f'primary2.{k}': v['p_two_sided'] for k, v in report['primary2'][metric]['contrasts'].items()})
        for arm, r in report['primary3'][metric].items():
            for t, v in r['tests'].items():
                family[f'primary3.{arm}.{t}'] = v['p_two_sided']
        report['holm'][metric] = holm(family, alpha=s['alpha'])
        report['consequences'][metric] = gate_consequences(gates, report['primary2'][metric], metric)
    if inputs.get('stage_d'):
        cohort_n = len(next(iter(inputs['stage_d'].values()))[0])
        report['concordance'] = concordance(
            {k: v for arm in stage_c.values() for m in arm.values() for k, v in m.items()}, inputs['stage_d'], conf,
            bootstrap_indices(cohort_n, resamples=s['resamples'], seed=s['seed']))
    report['descriptive'] = dict(neuron=inputs.get('neuron') or {}, a_vs_a0=None if inputs.get('a_vs_a0') is None else
                                 dict(inputs['a_vs_a0'], label='raw saved-factor cosine and drift; gauge-dependent, '
                                                               'historical description only'))
    report['compute'] = inputs.get('compute') or {}
    return report


def _fmt(x):
    if x is None:
        return 'unavailable'
    if isinstance(x, float):
        return f'{x:.4g}'
    return str(x)


def render(report):
    lines = ['# Adapter and neuron attribution — report', '', report['scope'], '',
             f"Provenance: {report['provenance_statement']}; per arm: "
             + '; '.join(f"{a}: {v.get('status')}" for a, v in sorted(report['provenance'].items())) + '.', '',
             report['independence'], '', '## Validity gates', '', f'Each gate is a {GATE_LABEL}.', '',
             '| arm | metric | top singles agreeing | group agrees | pass |', '| --- | --- | --- | --- | --- |']
    for key, g in report['gates'].items():
        arm, metric = key.split('|')
        lines.append(f"| {arm} | {metric} | {g['top_singles_sign_agree']}/5 | {g['group_sign_agree']} | {g['pass']} |")
    p1 = report['primary1']
    lines += ['', '## Primary 1 (gold-sft separability)', '', f"Outcome: **{p1['outcome']}**. "
              f"r point {_fmt(p1.get('r_point'))}, one-sided 95% bounds [{_fmt(p1.get('r_lower_95'))}, "
              f"{_fmt(p1.get('r_upper_95'))}], margin {_fmt(p1.get('margin'))}. " + ' '.join(p1['reasons'])]
    for metric in METRIC_KEY:
        p2 = report['primary2'][metric]
        lines += ['', f'## Primary 2 ({metric})', '']
        for k, c in p2['contrasts'].items():
            label = report['consequences'][metric]['primary2'][k]
            lines.append(f"- ({k}) {c.get('definition', '')}: {_fmt(c['estimate'])} {c['interval_95']} [{label}]")
        lines += ['', f'## Primary 3 ({metric})', '', report['consequences'][metric]['primary3'], '']
        for arm, r in report['primary3'][metric].items():
            lines.append(f"- {arm}: R(attr) {_fmt(r['R']['attr']['estimate'])} {r['R']['attr']['interval_95']}; "
                         f"classification {r['classification']} (descriptive)")
        lines += ['', f'Holm ({metric}): ' + ', '.join(f"{k} adj {_fmt(v['adjusted'])} reject {v['reject']}"
                                                         for k, v in report['holm'][metric].items())]
    if 'concordance' in report:
        lines += ['', '## Concordance (exploratory)', '', report['concordance']['label'],
                  'Disagreements: ' + (json.dumps(report['concordance']['disagreements']) or 'none')]
    lines += ['', '## Compute appendix', '', '```', json.dumps(report['compute'], indent=2, default=str), '```', '']
    return '\n'.join(lines)


# --- wiring for 06b-09 (descriptive) ---------------------------------------------------------------

def resolution_floor(rule, measured):
    """Evaluate the record's approved rule '<k>x ...' against the qualification's measured discrepancy."""
    import re
    match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*x\b', str(rule))
    if not match:
        raise ValueError(f'resolution-floor rule must read "<factor>x ..." (got {rule!r})')
    return float(match.group(1)) * float(measured)


def concentration_table(freeze, neuron, *, null_orders, null_seed, resolution_floor):
    """Concentration and cancellation at each arm x metric's frozen top down_proj adapter (discovery means)."""
    out = {}
    for arm, metrics in freeze.items():
        out[arm] = {}
        for metric, sel in metrics.items():
            layer = sel['down_proj']['layer']
            values = neuron.get(arm, {}).get(metric, {}).get(layer)
            if values is None:
                out[arm][metric] = dict(available=False, reason='no neuron attributions', layer=layer)
                continue
            out[arm][metric] = dict(concentration(values, null_orders=null_orders, null_seed=null_seed,
                                                  resolution_floor=float(resolution_floor)), layer=layer)
    return out


def a_vs_a0(rows_path, freeze):
    """Raw cos(a_T, a_0) and drift from the post hoc init check, for each frozen top down_proj; no sign alignment."""
    if rows_path is None:
        return None
    rows = [json.loads(line) for line in open(rows_path) if line.strip()]
    index = {(r['arm'], r['layer'], r['module']): r for r in rows}
    out = {}
    for arm, metrics in freeze.items():
        out[arm] = {}
        for metric, sel in metrics.items():
            row = index.get((arm, sel['down_proj']['layer'], 'down_proj'))
            out[arm][metric] = None if row is None else dict(layer=row['layer'], cos_a=row['cos_a'],
                                                             a_drift=row['a_drift'])
    return dict(out, label='raw saved-factor cosine and drift; gauge-dependent, historical description only')


def early_stop_threshold(untuned_items, *, rule):
    """Threshold from this study's untuned run under the record's rule (kind: quantile-of-untuned-eos-tokens)."""
    if not isinstance(rule, dict) or rule.get('kind') != 'quantile-of-untuned-eos-tokens' or \
            not isinstance(rule.get('quantile'), float) or not 0 < rule['quantile'] < 1:
        raise ValueError('early-stop rule must be {kind: quantile-of-untuned-eos-tokens, quantile: 0<q<1} '
                         'from the launch record')
    tokens = [r['tokens'] for r in untuned_items if r['stop_reason'] == 'eos']
    if not tokens:
        raise ValueError('untuned run has no EOS-terminated responses')
    return float(np.quantile(tokens, rule['quantile']))


def early_stop_rate(items, threshold):
    """Fraction of responses that stopped with EOS below the threshold; capped responses never count."""
    return float(np.mean([r['stop_reason'] == 'eos' and r['tokens'] < threshold for r in items])) if items else None


def stage_d_outcomes(items, *, early_threshold):
    n = len(items)
    capped = sum(r['stop_reason'] != 'eos' for r in items)
    invalid = sum(r.get('status') != 'valid' for r in items)
    return dict(n=n, accuracy=float(np.mean([bool(r['correct']) for r in items])),
                mean_tokens=float(np.mean([r['tokens'] for r in items])), capped=capped, invalid=invalid,
                capped_rate=capped / n, early_stop_rate=early_stop_rate(items, early_threshold))


def _correct(row):
    if 'correct' in row:
        return bool(row['correct'])
    if isinstance(row.get('score'), dict) and 'correct' in row['score']:
        return bool(row['score']['correct'])
    raise ValueError(f"pilot row {row.get('id')} has no correct flag")


def pilot_comparison(study, pilot):
    """Item-level agreement of this study's full-arm run with the pilot's saved results on the same items."""
    a = {r['id']: _correct(r) for r in study}
    b = {r['id']: _correct(r) for r in pilot}
    if set(a) != set(b):
        raise ValueError('study and pilot items differ')
    flips = [dict(id=i, pilot=b[i], study=a[i]) for i in sorted(a) if a[i] != b[i]]
    return dict(n=len(a), agree=len(a) - len(flips), flips=flips, study_accuracy=float(np.mean(list(a.values()))),
                pilot_accuracy=float(np.mean(list(b.values()))))

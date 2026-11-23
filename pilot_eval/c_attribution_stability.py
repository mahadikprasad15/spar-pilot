"""C-attribution uncertainty and stability (ticket 04). Descriptive only: nothing here is a gate.

Inputs are the sealed per-question sums from ``c_attribution.QuestionSums`` (FP64):
    adapters S [Q, A]; reads S [Q, L, W]; draw-half sums [2, A] and [2, L, W].
No gradients are recomputed.

Rankings follow the selection rule: "top k" means the k largest dJdλ (most helpful first).
Sign agreement over "each half's top k" uses each half's k largest |dJdλ| and reports the share of the
union whose signs agree between halves.
"""
import resource
import sys
import time
import tracemalloc

import numpy as np

from pilot_eval.c_attribution import SIGN_CONVENTION, aggregate, totals

SPLIT_SEED = 20261118
BOOTSTRAP_SEED = 1234
RESAMPLES = 10_000
INTERVAL_LABEL = 'exploratory, pointwise; selected on this cohort'
DRAW_SPLIT_LABEL = 'noisier: 4 draws per question per half'
PERCENTILES = (2.5, 97.5)


# --- splits -------------------------------------------------------------------------------------

def question_split(question_ids, seed=SPLIT_SEED):
    """Two halves of the question IDs (75/75 for 150), seeded; all draws of a question stay together."""
    ids = list(question_ids)
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('need at least 2 unique question ids')
    order = np.random.default_rng(seed).permutation(len(ids))
    half = len(ids) // 2
    a = sorted(ids[i] for i in order[:half])
    b = sorted(ids[i] for i in order[half:])
    return {'seed': seed, 'A': a, 'B': b}


def split_dJdlambda(S, question_ids, split):
    """dJdλ for each half from per-question S (each half averaged over its own questions)."""
    index = {q: i for i, q in enumerate(question_ids)}
    out = []
    for half in ('A', 'B'):
        rows = [index[q] for q in split[half]]
        out.append(aggregate(np.asarray(S)[rows])['dJdlambda'])
    return out


def draw_split_dJdlambda(half_sums, Q):
    """dJdλ for draws 0–3 and 4–7 from the accumulated half sums (sum over questions; divide by Q)."""
    half_sums = np.asarray(half_sums, dtype=np.float64)
    return [-half_sums[0] / Q, -half_sums[1] / Q]


# --- comparison summaries -------------------------------------------------------------------------

def _ranks(x):
    """Average ranks (1-based) with ties averaged, like scipy.stats.rankdata."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else None


def top(d, k):
    """Indices of the k largest values, ties broken by index (stable)."""
    d = np.asarray(d, dtype=np.float64).ravel()
    return np.argsort(-d, kind='mergesort')[:k]


def overlap(a, b, k):
    return int(len(set(top(a, k).tolist()) & set(top(b, k).tolist())))


def sign_agreement(a, b, k):
    """Share of the union of each half's top k by signed dJdλ (the selection order) whose signs agree."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    union = sorted(set(top(a, k).tolist()) | set(top(b, k).tolist()))
    agree = np.sign(a[union]) == np.sign(b[union])
    return {'share': float(agree.mean()), 'n': len(union)}


def compare_adapters(a, b, *, k_top=10, k_sign=20):
    return {'spearman': spearman(a, b), f'top{k_top}_overlap': overlap(a, b, k_top),
            f'sign_agreement_top{k_sign}': sign_agreement(a, b, k_sign)}


def compare_reads(a, b, layers, hot_layers, *, k=64):
    """Global top-k, then per layer and per hot layer: overlap and sign agreement, plus equal-layer means."""
    a, b = np.asarray(a), np.asarray(b)  # [L, W]
    width = a.shape[1]
    layer_of = lambda flat: [int(layers[i // width]) for i in flat]  # noqa: E731
    out = {'global': {f'top{k}_overlap': overlap(a, b, k), f'sign_agreement_top{k}': sign_agreement(a, b, k),
                      'layer_distribution': {h: _counts(layer_of(top(x, k))) for h, x in (('A', a), ('B', b))}}}
    per = {}
    for li, layer in enumerate(layers):
        per[int(layer)] = {'overlap': overlap(a[li], b[li], k), 'sign_agreement': sign_agreement(a[li], b[li], k)}
    out['per_layer'] = per
    out['per_layer_mean'] = _mean(per)
    hot = {int(h): per[int(h)] for h in hot_layers}
    out['hot_layers'] = hot
    out['hot_layer_mean'] = _mean(hot)
    return out


def _counts(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


def _mean(per):
    if not per:
        return None
    return {'overlap': float(np.mean([v['overlap'] for v in per.values()])),
            'sign_agreement': float(np.mean([v['sign_agreement']['share'] for v in per.values()]))}


def stability(sums, *, hot_layers, split=None, layers=None):
    """Question-split and draw-split summaries for one arm's completed QuestionSums."""
    split = split or question_split(sums.order)
    Q = len(sums.order)
    layers = list(layers if layers is not None else sums.layers)
    out = {'label': 'descriptive; never a gate', 'sign_convention': SIGN_CONVENTION,
           'question_split': {'seed': split['seed'], 'A': split['A'], 'B': split['B']}}
    qa, qb = split_dJdlambda(sums.adapters, sums.order, split)
    da, db = draw_split_dJdlambda(sums.adapter_halves, Q)
    out['question_split']['adapters'] = compare_adapters(qa, qb)
    out['draw_split'] = {'label': DRAW_SPLIT_LABEL, 'halves': ['draws 0-3', 'draws 4-7'],
                         'adapters': compare_adapters(da, db)}
    if sums.reads is not None:
        ra, rb = split_dJdlambda(sums.reads, sums.order, split)
        out['question_split']['reads'] = compare_reads(ra, rb, layers, hot_layers)
        ha, hb = draw_split_dJdlambda(sums.read_halves, Q)
        out['draw_split']['reads'] = compare_reads(ha, hb, layers, hot_layers)
    return out


# --- bootstrap ------------------------------------------------------------------------------------

def resample_counts(Q, B=RESAMPLES, seed=BOOTSTRAP_SEED):
    """[B, Q] counts of each question in each resample of Q whole questions. Shared by both arms."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((B, Q), dtype=np.float64)
    draws = rng.integers(0, Q, size=(B, Q))
    for b in range(B):
        counts[b] = np.bincount(draws[b], minlength=Q)
    return counts


def bootstrap_dJdlambda(S, counts, *, chunk=1024, percentiles=PERCENTILES, center=None):
    """Percentile intervals of dJdλ = -(counts @ S)/Q, streamed over column chunks; no resamples kept.

    ``S`` is [Q, ...]; returns arrays shaped like S[0] for each percentile. ``center`` optionally
    subtracts a second arm's S (paired difference, same counts).
    """
    S = np.asarray(S, dtype=np.float64)
    Q = S.shape[0]
    flat = S.reshape(Q, -1)
    other = None if center is None else np.asarray(center, dtype=np.float64).reshape(Q, -1)
    out = {p: np.empty(flat.shape[1]) for p in percentiles}
    for start in range(0, flat.shape[1], chunk):
        block = flat[:, start:start + chunk]
        if other is not None:
            block = block - other[:, start:start + chunk]
        values = -(counts @ block) / Q
        for p, v in zip(percentiles, np.percentile(values, percentiles, axis=0)):
            out[p][start:start + chunk] = v
    shape = S.shape[1:]
    return {f'p{p}': out[p].reshape(shape) for p in percentiles}


def bootstrap_ranks(S, counts, *, percentiles=PERCENTILES):
    """Rank intervals for adapters (1 = largest dJdλ) across resamples."""
    S = np.asarray(S, dtype=np.float64)
    values = -(counts @ S) / S.shape[0]
    ranks = np.argsort(np.argsort(-values, axis=1, kind='mergesort'), axis=1) + 1
    return {f'p{p}': v for p, v in zip(percentiles, np.percentile(ranks, percentiles, axis=0))}


def adapter_bootstrap(arms, *, B=RESAMPLES, seed=BOOTSTRAP_SEED):
    """``arms`` = {name: (question_ids, S [Q, A])}; identical question order is required for pairing."""
    names = list(arms)
    order = arms[names[0]][0]
    if any(list(arms[n][0]) != list(order) for n in names):
        raise ValueError('arms must share the question order to pair resamples')
    counts = resample_counts(len(order), B, seed)
    out = {'label': INTERVAL_LABEL, 'resamples': B, 'seed': seed, 'percentiles': list(PERCENTILES), 'arms': {}}
    for n in names:
        S = arms[n][1]
        out['arms'][n] = {'dJdlambda': bootstrap_dJdlambda(S, counts), 'rank': bootstrap_ranks(S, counts)}
    if len(names) == 2:
        a, b = names
        out['paired'] = {'difference': f'{a} - {b}',
                         'dJdlambda': bootstrap_dJdlambda(arms[a][1], counts, center=arms[b][1])}
    return out


def neuron_bootstrap(S, counts, *, selected=None, chunk=1024, tiers=('selected', 'per_layer'), benchmark=False):
    """Read intervals in priority order. ``S`` is [Q, L, W]; ``selected`` = list of (layer_index, j).

    Tiers: 'selected' (selected reads and controls), 'per_layer' (per-layer sums), 'all' (every read).
    With ``benchmark``, each tier records wall seconds, tracemalloc peak bytes and process max RSS.
    """
    S = np.asarray(S, dtype=np.float64)
    out = {'label': INTERVAL_LABEL, 'resamples': int(counts.shape[0])}
    jobs = {
        'selected': lambda: (bootstrap_dJdlambda(np.stack([S[:, li, j] for li, j in selected], 1), counts, chunk=chunk)
                             if selected else None),
        'per_layer': lambda: bootstrap_dJdlambda(S.sum(2), counts, chunk=chunk),
        'all': lambda: bootstrap_dJdlambda(S, counts, chunk=chunk),
    }
    bench = {}
    for tier in tiers:
        if benchmark:
            tracemalloc.start()
            t0 = time.perf_counter()
        out[tier] = jobs[tier]()
        if benchmark:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            bench[tier] = {'seconds': time.perf_counter() - t0, 'tracemalloc_peak_bytes': int(peak),
                           'max_rss_bytes': _max_rss()}
    if benchmark:
        out['benchmark'] = {'shape': list(S.shape), 'resamples': int(counts.shape[0]), 'chunk': chunk, 'tiers': bench}
    return out


def _max_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == 'darwin' else rss * 1024)


# --- also reported ---------------------------------------------------------------------------------

def descriptives(rewards, adapter_dJdlambda, *, k=10):
    """J-bar, informative (mixed) question count, and the top-k adapter share of gross |dJdλ|."""
    values = [r for rs in rewards.values() for r in rs]
    mixed = sum(1 for rs in rewards.values() if 0 < sum(rs) < len(rs))
    d = np.asarray(adapter_dJdlambda, dtype=np.float64)
    t = totals(d)
    share = float(np.abs(d[top(d, k)]).sum() / t['gross']) if t['gross'] > 0 else None
    return {'J_bar': float(np.mean(values)), 'informative_questions': mixed, 'questions': len(rewards),
            f'top{k}_adapter_share_of_gross': share, 'gross': t['gross'], 'net': t['net']}

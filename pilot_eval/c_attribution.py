"""C-attribution: leave-one-out weights, per-question aggregation, totals and signs (ticket 03).

Pure numpy; no torch import. The instrument (``attribution_instrument``) supplies per-response adapter
attributions attr_c(q, k) = -dM_qk/dlambda_c and, for neuron reads, per-question sums it reduces itself
from coefficients this module computes (``reduction_request``).

Estimand: J(lambda) = (1/Q) sum_q E_{y ~ pi_lambda(.|q)} [R(q, y)], R in {0, 1}.

    b_qk = (sum_k' R_qk' - R_qk) / (n_q - 1)          leave-one-out baseline
    coef_qk = (R_qk - b_qk) / n_q                       per-question coefficient (detached)
    w_qk = coef_qk / Q                                  weight
    S_q,c = sum_k coef_qk * attr_c(q, k)                per-question pre-aggregate
    removal_J_c = (1/Q) sum_q S_q,c = sum_qk w_qk attr_c(q, k)
    dJdlambda_c = -removal_J_c

Sign convention (also written into every output as ``sign_convention``):

| Quantity | Meaning | Positive means |
|---|---|---|
| `attr_c(e)` (existing) | first-order Δ metric when c is removed (1 → 0) | removing raises the metric |
| `removal_J_c` | first-order ΔJ when c is removed | removal is predicted to **raise** J locally |
| `dJdλ_c` | ∂J/∂λ_c at λ = 1 | slightly strengthening c is predicted to raise J (**helpful**, locally) |

Concentration shares always use the gross total sum_c |dJdlambda_c|; the net total |sum_c dJdlambda_c| is
reported beside it and is never a denominator.
"""
import json
from pathlib import Path

import numpy as np

SIGN_CONVENTION = [
    {'quantity': 'attr_c(e)', 'meaning': 'first-order Δ metric when c is removed (1 → 0)',
     'positive_means': 'removing raises the metric'},
    {'quantity': 'removal_J_c', 'meaning': 'first-order ΔJ when c is removed',
     'positive_means': 'removal is predicted to raise J locally'},
    {'quantity': 'dJdλ_c', 'meaning': '∂J/∂λ_c at λ = 1',
     'positive_means': 'slightly strengthening c is predicted to raise J (helpful, locally)'},
]
GROUP_PREDICTION_LABEL = 'first-order sampled-J predicted removal effect'
HALVES = ((0, 1, 2, 3), (4, 5, 6, 7))
IDENTITY_RTOL = 1e-4  # production-1: |sum_j attr_j - attr_i| <= 1e-4 * sum_j |attr_j|


class AggregationError(ValueError):
    pass


# --- weights ------------------------------------------------------------------------------------

def loo_coefficients(rewards):
    """(R_k - b_k) / n for one question's draws; b_k is the mean of the other draws. n < 2 is refused."""
    r = np.asarray(rewards, dtype=np.float64)
    n = r.size
    if r.ndim != 1 or n < 2:
        raise AggregationError(f'leave-one-out needs at least 2 draws, got {n}')
    if not np.isfinite(r).all():
        raise AggregationError('rewards must be finite')
    baseline = (r.sum() - r) / (n - 1)
    coef = (r - baseline) / n
    if np.all(r == r[0]):
        coef = np.zeros(n)  # exactly 0 for all-same groups, not merely rounding-small
    return coef


def loo_weights(groups):
    """Weights w_qk = (R_qk - b_qk) / (Q n_q) for ``groups`` = {question_id: [R_q0, R_q1, ...]}.

    Returns dict(Q, weights={qid: array}, coefficients={qid: array}) where coefficients omit 1/Q.
    """
    if not groups:
        raise AggregationError('no questions')
    q = len(groups)
    coefficients = {qid: loo_coefficients(r) for qid, r in groups.items()}
    return {'Q': q, 'coefficients': coefficients, 'weights': {qid: c / q for qid, c in coefficients.items()}}


def half_coefficients(rewards, draw_indices, halves=HALVES):
    """Per-row [n, 2] coefficients for the draw split: leave-one-out within each half (n = 4)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    draw_indices = list(draw_indices)
    out = np.zeros((len(rewards), len(halves)))
    for h, members in enumerate(halves):
        rows = [k for k, d in enumerate(draw_indices) if d in members]
        if sorted(draw_indices[k] for k in rows) != sorted(members):
            raise AggregationError(f'draw half {h} needs draws {list(members)}, got {sorted(draw_indices[k] for k in rows)}')
        out[rows, h] = loo_coefficients(rewards[rows])
    return out


def reduction_request(rows, rewards):
    """The instrument's ``neuron_reduction`` for one batch of rows (each with question_id, draw_index).

    ``rewards`` = {question_id: [R for draw 0, 1, ...]} for the whole question, so a micro-batch may hold
    part of a question. Each row's stored reward must equal its entry there.
    """
    coef, halves = [], []
    for row in rows:
        r = rewards[row['question_id']]
        d = row['draw_index']
        if 'reward' in row and row['reward'] != r[d]:
            raise AggregationError(f"{row['question_id']} draw {d}: reward {row['reward']} != {r[d]}")
        coef.append(float(loo_coefficients(r)[d]))
        halves.append([float(v) for v in half_coefficients(r, range(len(r)))[d]])
    return {'question_ids': [r['question_id'] for r in rows], 'coefficients': coef, 'half_coefficients': halves}


# --- accumulation across units --------------------------------------------------------------------

class QuestionSums:
    """Accumulate per-question S_q (adapters and down_proj reads) and draw-half sums, in FP64.

    ``order`` fixes the question index. Adapter S_q is computed here from per-response attributions;
    read S_q comes pre-reduced from the instrument's ``by_question`` output.
    """

    def __init__(self, order, *, n_adapters, layers=(), width=0):
        self.order = list(order)
        self.index = {q: i for i, q in enumerate(self.order)}
        if len(self.index) != len(self.order):
            raise AggregationError('duplicate question ids')
        self.layers = list(layers)
        q = len(self.order)
        self.adapters = np.zeros((q, n_adapters))
        self.adapter_halves = np.zeros((2, n_adapters))
        self.reads = np.zeros((q, len(self.layers), width)) if self.layers else None
        self.read_halves = np.zeros((2, len(self.layers), width)) if self.layers else None
        self.rows_seen = {}

    def add_adapters(self, question_ids, draw_indices, attr, coefficients, halves):
        attr = np.asarray(attr, dtype=np.float64)
        if not np.isfinite(attr).all():
            raise AggregationError('adapter attributions must be finite')
        coefficients = np.asarray(coefficients, dtype=np.float64)
        halves = np.asarray(halves, dtype=np.float64)
        for k, (qid, d) in enumerate(zip(question_ids, draw_indices)):
            key = (qid, d)
            if key in self.rows_seen:
                raise AggregationError(f'row {key} added twice')
            self.rows_seen[key] = True
            self.adapters[self.index[qid]] += coefficients[k] * attr[k]
        self.adapter_halves += halves.T @ attr

    def add_reads(self, neuron_out, question_ids):
        """``neuron_out`` = instrument output ``neuron['seqlogp']``; ``question_ids`` = its neuron_question_ids."""
        for li, layer in enumerate(self.layers):
            block = neuron_out[layer]
            for qi, qid in enumerate(question_ids):
                self.reads[self.index[qid], li] += block['by_question'][qi]
            self.read_halves[:, li] += block['half_sums']

    def complete(self, expected_rows):
        if len(self.rows_seen) != expected_rows:
            raise AggregationError(f'{len(self.rows_seen)} rows accumulated, expected {expected_rows}')


# --- aggregation, totals, predictions -------------------------------------------------------------

def aggregate(S, Q=None):
    """removal_J = (1/Q) sum_q S_q and dJdlambda = -removal_J, over the leading (question) axis."""
    S = np.asarray(S, dtype=np.float64)
    Q = S.shape[0] if Q is None else Q
    if Q < 1 or S.shape[0] != Q:
        raise AggregationError(f'S has {S.shape[0]} questions, Q = {Q}')
    removal = S.sum(0) / Q
    return {'removal_J': removal, 'dJdlambda': -removal}


def totals(dJdlambda):
    """Gross sum |dJdlambda|, net |sum dJdlambda|, their ratio, and gross shares (never net shares)."""
    d = np.asarray(dJdlambda, dtype=np.float64)
    gross = float(np.abs(d).sum())
    net = float(abs(d.sum()))
    shares = np.abs(d) / gross if gross > 0 else np.zeros_like(d)
    return {'gross': gross, 'net': net, 'net_over_gross': net / gross if gross > 0 else None,
            'shares_of_gross': shares, 'denominator': 'gross'}


def share(dJdlambda, selection):
    t = totals(dJdlambda)
    flat = np.asarray(dJdlambda, dtype=np.float64)
    return {'share_of_gross': float(np.abs(flat[selection]).sum() / t['gross']) if t['gross'] > 0 else None,
            'gross': t['gross'], 'net': t['net']}


def group_prediction(removal_J, selection):
    """Sum of removal_J over the selected components, in J units and percentage points."""
    value = float(np.asarray(removal_J, dtype=np.float64)[selection].sum())
    return {'label': GROUP_PREDICTION_LABEL, 'J': value, 'pp': 100 * value}


def per_component_gross_net(S):
    """Per component: |sum_q S_q| / sum_q |S_q| (1 = no cancellation across questions)."""
    S = np.asarray(S, dtype=np.float64)
    gross = np.abs(S).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(gross > 0, np.abs(S.sum(0)) / gross, np.nan)


def neuron_identity(read_dJdlambda, adapter_dJdlambda, down_index, *, rtol=IDENTITY_RTOL):
    """Aggregated identity per down_proj layer: sum_j dJdlambda_j = dJdlambda_i (gross denominator)."""
    out, ok = {}, True
    for li, (layer, i) in enumerate(sorted(down_index.items())):
        reads = np.asarray(read_dJdlambda[li], dtype=np.float64)
        net = float(reads.sum())
        gross = float(np.abs(reads).sum())
        error = net - float(adapter_dJdlambda[i])
        passed = abs(error) <= rtol * gross
        ok &= passed
        out[str(layer)] = {'adapter_index': int(i), 'reads_net': net, 'reads_gross': gross,
                           'adapter': float(adapter_dJdlambda[i]), 'error': error,
                           'relative_to_gross': abs(error) / gross if gross > 0 else None, 'passed': bool(passed)}
    return {'rule': f'|sum_j dJdλ_j - dJdλ_i| <= {rtol} * sum_j |dJdλ_j|', 'layers': out, 'passed': bool(ok),
            'sign_convention': SIGN_CONVENTION}


def summary(sums, *, adapters, down_index):
    """Arm-level aggregate from a completed ``QuestionSums``: adapters, reads, totals, identity."""
    a = aggregate(sums.adapters)
    out = {'sign_convention': SIGN_CONVENTION, 'Q': len(sums.order), 'question_ids': sums.order,
           'adapters': {'names': list(adapters), **a, 'totals': _plain(totals(a['dJdlambda']))}}
    if sums.reads is not None:
        r = aggregate(sums.reads)
        out['reads'] = {'layers': sums.layers, **r,
                        'totals': {'all': _plain(totals(r['dJdlambda'])),
                                   'per_layer': [_plain(totals(x)) for x in r['dJdlambda']]}}
        out['identity'] = neuron_identity(r['dJdlambda'], a['dJdlambda'], down_index)
    return out


def _plain(t):
    return {k: v for k, v in t.items() if k != 'shares_of_gross'}


def write_identity(path, identity):
    Path(path).write_text(json.dumps(identity, indent=2, ensure_ascii=False) + '\n')

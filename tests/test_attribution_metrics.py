"""Ticket 02: proxy metrics (trace log-likelihood, stop pull, KL) on hand-built logits (CPU)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip('torch', exc_type=ImportError)

from pilot_eval import attribution_metrics as am  # noqa: E402

V = 6


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def fixture():
    """Prompt 0-1, header 2, content 3-4, <|im_end|> 5; distinct logits at every position."""
    ids = [1, 2, 3, 4, 0, 5]
    logits = np.array([[0.1 * (p + 1) * (v + 1) + 0.37 * ((p * 7 + v * 3) % 5) for v in range(V)]
                       for p in range(len(ids))])
    return ids, logits, dict(trace_targets=[3, 4], stop_targets=[3, 4])


def batch(rows, width=None):
    """rows: (ids, logits); right-padded with garbage logits."""
    width = width or max(len(i) for i, _ in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    logits = torch.full((len(rows), width, V), 0.0, dtype=torch.float32)
    for r, (i, l) in enumerate(rows):
        ids[r, :len(i)] = torch.tensor(i)
        logits[r, :len(i)] = torch.tensor(l, dtype=torch.float32)
        logits[r, len(i):] = 1e3 * torch.arange(V, dtype=torch.float32)  # padding: must be inert
    return ids, logits


def test_trace_loglik_hand_computed_includes_first_content_excludes_eos():
    ids, logits, masks = fixture()
    lp = log_softmax(logits)
    expected = (lp[2, ids[3]] + lp[3, ids[4]]) / 2  # logits read at t-1
    tid, tl = batch([(ids, logits)])
    value = am.trace_loglik(tl, tid, [masks])
    assert value.dtype == torch.float64 and value.shape == (1,)
    assert value.item() == pytest.approx(expected, rel=1e-6)
    # first content token's prediction (logits at header position 2) is read
    changed = logits.copy()
    changed[2, ids[3]] += 1.0
    assert am.trace_loglik(*batch([(ids, changed)])[::-1], [masks]).item() != pytest.approx(expected, rel=1e-6)
    # <|im_end|>'s prediction (logits at position 4) is not read; nor are prompt or EOS positions
    for p in (0, 1, 4, 5):
        changed = logits.copy()
        changed[p] += np.arange(V)
        assert am.trace_loglik(*batch([(ids, changed)])[::-1], [masks]).item() == pytest.approx(expected, rel=1e-6)


def test_padding_is_inert_and_examples_are_independent():
    ids, logits, masks = fixture()
    short_ids, short_logits = ids[:5], logits[:5]
    short_masks = dict(trace_targets=[3], stop_targets=[3])
    alone = am.trace_loglik(*batch([(short_ids, short_logits)])[::-1], [short_masks])
    padded = am.trace_loglik(*batch([(short_ids, short_logits), (ids, logits)], width=9)[::-1],
                             [short_masks, masks])
    assert padded[0].item() == pytest.approx(alone[0].item(), rel=1e-7)


def test_target_positions_validated():
    ids, logits, _ = fixture()
    tid, tl = batch([(ids, logits)])
    for bad in ([0], [6], [], [3, 3]):
        with pytest.raises(ValueError):
            am.trace_loglik(tl, tid, [dict(trace_targets=bad, stop_targets=[3])])


def test_logit_positions_is_a_one_step_shift_without_audit_view_names():
    assert am.logit_positions([3, 4, 7], length=8) == [2, 3, 6]
    with pytest.raises(ValueError):
        am.logit_positions([0], length=8)


def test_stop_pull_hand_computed_single_and_two_ids():
    ids, logits, masks = fixture()
    lp = log_softmax(logits)
    tid, tl = batch([(ids, logits)])
    values, reasons = am.stop_pull(tl, [masks], stop_ids=[5])
    assert values[0].item() == pytest.approx((lp[2, 5] + lp[3, 5]) / 2, rel=1e-6) and reasons == [None]
    values, _ = am.stop_pull(tl, [masks], stop_ids=[5, 0])
    two = [math.log(math.exp(lp[p, 5]) + math.exp(lp[p, 0])) for p in (2, 3)]
    assert values[0].item() == pytest.approx(sum(two) / 2, rel=1e-6)
    with pytest.raises(ValueError):
        am.stop_pull(tl, [masks], stop_ids=[])
    with pytest.raises(ValueError):
        am.stop_pull(tl, [masks], stop_ids=[5, 5])


def test_stop_pull_empty_mask_is_nan_with_reason_never_zero():
    ids, logits, masks = fixture()
    tid, tl = batch([(ids, logits), (ids, logits)])
    values, reasons = am.stop_pull(tl, [masks, dict(trace_targets=[3, 4], stop_targets=[])], stop_ids=[5])
    assert math.isfinite(values[0].item()) and math.isnan(values[1].item())
    assert reasons[0] is None and reasons[1] == 'empty-stop-pull-mask'
    summary = am.equal_example_mean(values)
    assert summary['examples'] == 1 and summary['excluded'] == 1 and summary['mean'] == pytest.approx(values[0].item())


def test_equal_example_weighting_is_not_token_weighting():
    values = torch.tensor([1.0, 4.0], dtype=torch.float64)
    assert am.equal_example_mean(values)['mean'] == pytest.approx(2.5)
    empty = am.equal_example_mean(torch.tensor([float('nan')], dtype=torch.float64))
    assert empty['mean'] is None and empty['examples'] == 0


def test_kl_direction_full_against_condition_three_distributions():
    ids, full, masks = fixture()
    rng = np.random.default_rng(0)
    cut, untuned = full + rng.normal(0, 1, full.shape), full + rng.normal(0, 2, full.shape)

    def manual(p, q):
        lp, lq = log_softmax(p), log_softmax(q)
        return np.mean([(np.exp(lp[t]) * (lp[t] - lq[t])).sum() for t in (2, 3)])

    _, tf = batch([(ids, full)])
    _, tc = batch([(ids, cut)])
    _, tu = batch([(ids, untuned)])
    forward = am.kl_full_vs_condition(tf, tc, [masks])[0].item()
    assert forward == pytest.approx(manual(full, cut), rel=1e-6)
    assert forward != pytest.approx(manual(cut, full), rel=1e-3)
    assert forward != pytest.approx(manual(cut, untuned), rel=1e-3)
    assert am.kl_full_vs_condition(tf, tf, [masks])[0].item() == pytest.approx(0.0, abs=1e-12)


def test_metrics_are_differentiable_with_finite_gradients():
    ids, logits, masks = fixture()
    tid, tl = batch([(ids, logits)])
    tl.requires_grad_(True)
    loss = am.trace_loglik(tl, tid, [masks]).sum() + am.stop_pull(tl, [masks], stop_ids=[5])[0].sum()
    (grad,) = torch.autograd.grad(loss, [tl])
    assert torch.isfinite(grad).all() and grad[0, 2].abs().sum() > 0 and grad[0, 5].abs().sum() == 0


def test_requires_fp32_finite_logits():
    ids, logits, masks = fixture()
    tid, tl = batch([(ids, logits)])
    with pytest.raises(ValueError):
        am.trace_loglik(tl.double(), tid, [masks])
    tl[0, 2, 0] = float('nan')
    with pytest.raises(ValueError):
        am.trace_loglik(tl, tid, [masks])

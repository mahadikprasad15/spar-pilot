"""Ticket 02: teacher-forced proxy metrics for adapter attribution (spec §Metrics), differentiable.

Masks are *target* positions in the teacher-forced sequence (prompt IDs + generated IDs):

- ``trace_targets`` T_e: every response-content token, including the first; never ``<|im_end|>``,
  the trailing newline, prompt, header or padding.
- ``stop_targets``: target positions x_{t+1} whose context x_<=t is a stop-pull prefix position
  (next target before the start of the last non-empty content line; built in ticket 01).

Both are read from the logits one position earlier (``logit_positions``). This adapts the shift of
``token_measurement_math.prediction_positions`` without its audit view names.

    L_e = mean_{t in T_e} log p(x_t | x_<t)
    S_e = mean_{t in P_e} log sum_{s in S} p(s | x_<=t)       (NaN with a reason when P_e is empty)
    KL_e = mean_{t in T_e} KL(full arm || condition) at the logits that predict x_t

log-softmax runs on the FP32 logits; per-position terms are accumulated in FP64. Every example has
equal weight (``equal_example_mean``).
"""
import torch

EMPTY_STOP = 'empty-stop-pull-mask'


def logit_positions(targets, *, length):
    """Target t is predicted by the logits at t - 1; refuse t = 0, out-of-range and duplicate targets."""
    targets = [int(t) for t in targets]
    if len(set(targets)) != len(targets) or any(not 1 <= t < length for t in targets):
        raise ValueError(f'target positions must be unique and in [1, {length})')
    return [t - 1 for t in targets]


def _check(logits):
    if logits.dtype != torch.float32 or logits.ndim != 3:
        raise ValueError('FP32 [batch, seq, vocab] logits required')
    if not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError('nonfinite logits')


def _length(row, width):
    return int(row.get('length', width))


def _selected(logits, masks, key, *, allow_empty):
    out = []
    for r, row in enumerate(masks):
        positions = logit_positions(row[key], length=_length(row, logits.shape[1]))
        if not positions and not allow_empty:
            raise ValueError(f'empty {key} mask')
        out.append(positions)
    return out


def trace_loglik(logits, input_ids, masks):
    """Per-example trace log-likelihood [batch] (FP64)."""
    _check(logits)
    values = []
    for r, positions in enumerate(_selected(logits, masks, 'trace_targets', allow_empty=False)):
        index = torch.tensor(positions, device=logits.device)
        targets = input_ids[r, index + 1]
        lp = torch.log_softmax(logits[r, index], dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        values.append(lp.double().sum() / len(positions))
    return torch.stack(values)


def stop_pull(logits, masks, *, stop_ids):
    """Per-example stop pull [batch] (FP64) and reasons; an empty mask is NaN, never 0."""
    _check(logits)
    stop_ids = [int(s) for s in stop_ids]
    if not stop_ids or len(set(stop_ids)) != len(stop_ids):
        raise ValueError('stop IDs must be a nonempty set')
    stops = torch.tensor(stop_ids, device=logits.device)
    values, reasons = [], []
    for r, positions in enumerate(_selected(logits, masks, 'stop_targets', allow_empty=True)):
        if not positions:
            values.append(torch.tensor(float('nan'), dtype=torch.float64, device=logits.device))
            reasons.append(EMPTY_STOP)
            continue
        lp = torch.log_softmax(logits[r, torch.tensor(positions, device=logits.device)], dim=-1)
        values.append(torch.logsumexp(lp[:, stops], dim=-1).double().sum() / len(positions))
        reasons.append(None)
    return torch.stack(values), reasons


def kl_full_vs_condition(full_logits, condition_logits, masks):
    """Per-example mean KL(full arm || condition) [batch] over the trace-target logit positions (FP64 sums)."""
    _check(full_logits)
    _check(condition_logits)
    if full_logits.shape != condition_logits.shape:
        raise ValueError('paired logits must have the same shape')
    values = []
    for r, positions in enumerate(_selected(full_logits, masks, 'trace_targets', allow_empty=False)):
        index = torch.tensor(positions, device=full_logits.device)
        p = torch.log_softmax(full_logits[r, index], dim=-1)
        q = torch.log_softmax(condition_logits[r, index], dim=-1)
        values.append((p.exp() * (p - q)).double().sum(-1).sum() / len(positions))
    return torch.stack(values)


def equal_example_mean(values):
    """Equal weight per example over defined (non-NaN) values; empty -> None, never 0."""
    v = values.detach().double().cpu()
    defined = ~torch.isnan(v)
    count = int(defined.sum())
    return dict(mean=float(v[defined].mean()) if count else None, examples=count,
                excluded=int((~defined).sum()), weighting='equal-example')

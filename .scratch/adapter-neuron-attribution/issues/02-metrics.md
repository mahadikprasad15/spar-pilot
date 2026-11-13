# 02: Proxy metrics

**Status:** ready-for-agent — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** none

**What to build:** `pilot_eval/attribution_metrics.py`: per-example trace log-likelihood, stop pull and KL from logits and frozen masks; differentiable.

- [ ] Trace log-likelihood: mean over target positions of log p(x_t | x_<t), logits read at t−1.
- [ ] Stop pull: mean over prefix positions of logsumexp of the stop IDs' log-probs; rows with no positions return NaN with a reason, never 0.
- [ ] KL(full ‖ condition) per example over trace logit positions; FP64 accumulation.
- [ ] Tests: hand-computed sequence with distinct per-position values (first content token included, EOS excluded); padding/template inert; equal-example weighting; finite gradients.

## Comments

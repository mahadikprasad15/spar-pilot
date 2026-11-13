# 02: Proxy metrics

**Status:** resolved (2026-11-13) — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** none

**What to build:** `pilot_eval/attribution_metrics.py`: per-example trace log-likelihood, stop pull and KL from logits and frozen masks; differentiable.

- [ ] Trace log-likelihood: mean over target positions of log p(x_t | x_<t), logits read at t−1.
- [ ] Stop pull: mean over prefix positions of logsumexp of the stop IDs' log-probs; rows with no positions return NaN with a reason, never 0.
- [ ] KL(full ‖ condition) per example over trace logit positions; FP64 accumulation.
- [ ] Tests: hand-computed sequence with distinct per-position values (first content token included, EOS excluded); padding/template inert; equal-example weighting; finite gradients.

## Comments

### 2026-11-13 — review round 3 (Codex), adopted

- `token_measurement_math.prediction_positions` expects full-length `question`/`solution`/`user` masks and raises on this study's masks: adapt or parameterize its target-shift logic; do not call it unchanged. Test masks with distinct prompt and content positions.
- KL direction is KL(full arm ‖ condition) on FP32 forwards with an explicitly constructed same-arm unablated reference — not `DriftEngine`'s variant ‖ untuned with an FP64 head. Test a three-distribution fixture where KL(full ‖ cut), its reverse and KL(cut ‖ untuned) all differ.

## Answer

2026-11-13 — `pilot_eval/attribution_metrics.py`, tests `tests/test_attribution_metrics.py` (10, CPU).

- Masks are **target** positions: `trace_targets` (T_e) and `stop_targets` (x_{t+1} for t ∈ P_e). Both are read from the logits one position earlier via `logit_positions`, which adapts the shift in `token_measurement_math.prediction_positions` without its `question`/`solution`/`user` view names. It refuses target 0, out-of-range and duplicate targets.
- `trace_loglik`: mean log p(x_t | x_<t) per example (FP32 log-softmax, FP64 sums). The hand-computed fixture shows the first content token is read (header logits) and `<|im_end|>`, the prompt and the EOS position are not. Padding is inert.
- `stop_pull(stop_ids=…)`: mean logsumexp of the stop IDs' log-probs. An empty mask gives NaN with reason `empty-stop-pull-mask`, never 0. The stop set itself is sealed and checked in ticket 01.
- `kl_full_vs_condition`: KL(full arm ‖ condition) at the trace-target logit positions, vocabulary sum in FP64. A three-distribution fixture separates it from the reverse direction and from KL(cut ‖ untuned).
- `equal_example_mean`: equal weight per example over defined values; all empty gives `None`.
- Logits must be FP32 and finite. Gradients are finite and reach only the read positions.

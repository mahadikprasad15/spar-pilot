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

### 2026-11-13 — review round 3 (Codex), adopted

- `token_measurement_math.prediction_positions` expects full-length `question`/`solution`/`user` masks and raises on this study's masks: adapt or parameterize its target-shift logic; do not call it unchanged. Test masks with distinct prompt and content positions.
- KL direction is KL(full arm ‖ condition) on FP32 forwards with an explicitly constructed same-arm unablated reference — not `DriftEngine`'s variant ‖ untuned with an FP64 head. Test a three-distribution fixture where KL(full ‖ cut), its reverse and KL(cut ‖ untuned) all differ.

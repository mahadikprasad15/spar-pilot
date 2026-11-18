# 03: Leave-one-out weights, aggregation and sign convention

**Status:** needs-triage
**Type:** task
**Blocked by:** 01, 02

**What to build:** `pilot_eval/c_attribution.py` (pure numpy; no torch import at module level).

- [ ] `loo_weights(groups)`: w_qk = (R_qk − b_qk)/(Q·n_q), b_qk = (Σ R − R_qk)/(n_q − 1); refuse n_q < 2; all-same groups → exactly 0; Σ_k w_qk = 0.
- [ ] `aggregate(attr, weights)` over sealed per-response units: `removal_J` = Σ w·attr and `dJdλ` = −`removal_J`, for 196 adapters and for each `down_proj` layer's 8,960 neuron reads.
- [ ] The sign-convention table from the spec appears verbatim in the module docstring and in every output's `sign_convention` field.
- [ ] Aggregated neuron identity: Σ_j dJdλ_j = dJdλ_i per `down_proj` layer at `identity_rtol`, written to `identity.json`.
- [ ] Tests: hand-computed weights; exact-enumeration fixture (2-token vocabulary, length-2 responses) where the exact ∇J equals the estimator's expectation over all responses; a synthetic helpful adapter gives dJdλ > 0 and removal_J < 0; question equal-weighting with unequal group sizes refused or handled as specified; mutations fail (sign flip, GRPO std-normalized advantages, token-mean metric).

## Comments

## Answer

# 02: Per-example teacher-forced metrics

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** `pilot_eval/attribution_metrics.py`: M_acc, M_len, KL_full and Drift from logits and frozen masks, per example, equal example weighting, differentiable (used as the backward target in 03).

**Blocked by:** none (fixture-testable).

- [ ] M_acc(e) = mean log p(next token) over `response_content`.
- [ ] M_len(e) = mean log(p(EOS) + p(`####` first token)) over the M_len mask; numerically stable (logsumexp of the two log-probs).
- [ ] KL_full and Drift as mean per-token KL over the stated masks; FP64 accumulation.
- [ ] Return a vector over examples; `sum()` is the backward target.
- [ ] Tests: hand-computed values on a 3-token vocabulary fixture; padding and template positions contribute nothing; equal-example weighting differs from token weighting on a fixture where it should; gradients finite.

## Comments

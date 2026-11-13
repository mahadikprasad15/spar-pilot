# 02: Proxy metrics

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** none

**What to build:** `pilot_eval/attribution_metrics.py`: per-example M_acc, M_len and KL from logits and frozen masks; differentiable; a vector over examples whose sum is the backward target.

- [ ] M_acc: mean log p(next token) over `response_content`.
- [ ] M_len: mean logsumexp of the two log-probs (EOS, `####` first token) over the M_len mask.
- [ ] KL(full ‖ ablated) per example over `response_content`; FP64 accumulation.
- [ ] Tests: hand-computed values on a tiny-vocabulary fixture; padding/template positions contribute nothing; equal-example weighting; finite gradients.

## Comments

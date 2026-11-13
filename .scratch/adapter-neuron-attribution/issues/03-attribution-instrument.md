# 03: Attribution instrument

**Status:** needs-triage — draft v2
**Type:** task
**Blocked by:** 02

**What to build:** `pilot_eval/attribution_instrument.py`.

- [ ] Gate: forward hook on each `lora_B` returning g · out (peft 0.18.0); proven equal to an explicit scaled-branch gate with s ≠ 1.
- [ ] Pass 1: forward + backward of Σ trace log-likelihood + backward of Σ stop pull (graph retained once); attr_i(e) over all unpadded positions; per-token ∂M/∂c_t for every `down_proj` adapter, per metric.
- [ ] Pass 2: forward-only recompute of z per `down_proj` layer, one layer at a time; attr_j(e) = −a_j Σ_t z_{j,t} g_t; Σ_j attr_j = attr_i enforced per example.
- [ ] GPU-side reductions; one async host copy per batch; background writes; token-budget packing by length.
- [ ] Routing and frozen parameters verified after every `set_adapter`.
- [ ] Tests (tiny random Qwen2 + PEFT 0.18.0): g = 1 logits identical; untuned and zero-b → 0; finite differences (5 adapters, 3 neurons) independent of the hook path; earlier-prompt-token fixture; pass-2 recompute equals stored-z reference; batch 1 vs 8; gauge; mutations (wrong layer, wrong branch, wrong sign, mean-not-sum, response-only positions) each fail.

## Comments

# 03: Attribution instrument

**Status:** ready-for-agent — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 02

**What to build:** `pilot_eval/attribution_instrument.py`.

- [ ] Capture only the scalar read c_t (the `lora_A` output, rank 1) of all 196 adapters; attr_i(e) = −Σ_t c_t ∂M/∂c_t. Never request full-width branch outputs or their gradients.
- [ ] Oracle on fixtures: a gate on the explicitly built scaled branch, −Σ_t ⟨∂M/∂Δy, Δy⟩, agrees for every module type, s ≠ 1 and both metrics.
- [ ] Differentiable path: `requires_grad_(True)` on the embedding output; all weights frozen; each metric via its own `torch.autograd.grad` call over the list of c_t tensors, returning fresh tensors (never `.grad` accumulation).
- [ ] Pass 1: forward + autograd.grad of Σ trace log-likelihood (retain_graph) + autograd.grad of Σ stop pull; attr_i(e) over all unpadded positions from the scalar reads; per-token ∂M/∂c_t for every `down_proj` adapter, per metric; plus per example each adapter's Σ_t |c_t| and token count and each `down_proj` adapter's signed Σ_t c_t (for matching and c̄).
- [ ] Pass 2: forward-only recompute of z per `down_proj` layer, one layer at a time; attr_j(e) = −a_j Σ_t z_{j,t} g_t; Σ_j attr_j = attr_i enforced per example.
- [ ] Bounded output head: logits computed in position chunks with checkpointing, using `stage0_push_backend.collect` as a pattern only (it requires zero B and trainable adapter weights; this study keeps all weights frozen) (full-vocabulary FP32 logits are ~10 GB at 16k tokens).
- [ ] GPU-side reductions; one async host copy per batch; background writes; token-budget packing by length.
- [ ] Routing and frozen parameters verified after every `set_adapter`.
- [ ] Tests (tiny random Qwen2 + PEFT 0.18.0): frozen model without the embedding-output graph has no grad_fn (documents the failure); each metric's attributions match its own finite differences; metric order does not change results; g = 1 logits identical; untuned and zero-b → 0; finite differences (5 adapters, 3 neurons) independent of the hook path; earlier-prompt-token fixture; pass-2 recompute equals stored-z reference; batch 1 vs 8; gauge; mutations (wrong layer, wrong branch, wrong sign, mean-not-sum, response-only positions) each fail.

## Comments

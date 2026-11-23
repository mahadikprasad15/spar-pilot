# 03: Attribution instrument

**Status:** resolved (2026-11-13) — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
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

## Answer

2026-11-13 — `pilot_eval/attribution_instrument.py`, tests `tests/test_attribution_instrument.py` (22, tiny random Qwen2 + PEFT 0.18.0, CPU, LoRA scale 2).

- **Scalar reads only.** `Instrument` captures each adapter's `lora_A` output c_t ([B, T, 1]) for all adapters, ordered by layer then q, k, v, o, gate, up, down. It computes attr_i = −Σ_t c_t ∂M/∂c_t over every unpadded position. No full-width branch output or gradient is requested.
- **Graph.** A forward hook calls `requires_grad_(True)` on the embedding output; every weight stays frozen (checked, refused otherwise). Each metric is a separate `torch.autograd.grad` over the list of c_t, with `retain_graph` only on the first call. A test documents that the frozen model without this hook has logits with no `grad_fn`.
- **Bounded head.** Logits are computed only at the read positions, in checkpointed chunks (`head_chunk`), through `attribution_metrics.position_terms`, the same formulas as ticket 02. Values match the full-logit path.
- **Pass 1 outputs** per example: attr for both metrics (NaN rows where the stop mask is empty), Σ_t |c_t| and the token count per adapter (write matching), signed Σ_t c_t per `down_proj` (c̄), and per-token g for every `down_proj` adapter.
- **Pass 2** is forward only. The `down_proj` pre-hook contracts attr_j = −a_j Σ_t z_{j,t} g_t in FP64 and drops z, so at most one layer's z is held. Σ_j attr_j = attr_i is enforced per example at `identity_rtol`, which must be supplied (launch record) and is relative to Σ_j |attr_j|.
- **Independent oracle and finite differences.** `branch_gates` and `neuron_gate` rebuild the scaled branch explicitly with a gate. Checks against them:
  - The branch-output oracle matches for all 7 module types, both metrics, at s = 2.
  - Central finite differences match for 5 adapters and 3 neurons, both metrics.
  - A prompt-position-only adapter is attributed correctly.
  - Pass 2 equals a stored-z reference; batch 1 equals the batched run; gauge k ∈ {−2, 0.5} leaves results unchanged; metric order makes no difference; g = 1 hooks leave logits bit-identical.
  - Untuned and zero-b give exactly 0.
  - Each mutation fails: wrong layer, wrong branch, wrong sign, mean instead of sum, response-only positions.
- **Routing.** Checked before every pass: active adapter set, not merged or disabled, rank 1, dropout 0, FP32, frozen. A `set_adapter` switch is re-checked, and the instrument refuses the trainable flags PEFT sets on the new adapter.
- **Found and fixed.** Leaving `disable_adapter()` in PEFT 0.18.0 re-marks adapter weights trainable. `frozen_disable_adapter` restores the flags; recorded in `docs/agents/experiment-lessons.md`.
- `unit_blocks` maps one batch result to BlockStore blocks, storing FP32 copies of FP64 results. The identity is checked in memory on FP64 before storage. Stop-pull arrays sit in a `stop` group whose count is 0 for empty masks, so those rows are NaN, never 0.
- **Not in this module.** Token-budget packing uses `neuron_audit_qualify.pack` in the chain runner, and background unit writes belong to ticket 11. Host copies are per output array, not one async copy per batch; that is a performance item to measure in 06a and 06b, not a validity issue.
- Addendum (with ticket 04): pass 2 also saves the term size |a_j| Σ_t |z_{j,t}| over unpadded positions per `down_proj` layer (`term_size`), which the term-size neuron order needs; no extra run.

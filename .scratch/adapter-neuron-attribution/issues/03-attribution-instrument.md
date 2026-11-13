# 03: Merged attribution instrument

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 02

**What to build:** `pilot_eval/attribution_instrument.py`: one forward and two backwards (M_acc, M_len) per batch giving per-example attribution for all 196 adapters and for all 8,960 neurons of all 28 `down_proj` adapters.

- [ ] Hook each live LoRA branch output s·B(Ax) (resolve from the installed PEFT; fail if fused or missing); `retain_grad()`; no parameter requires grad; `requires_grad_(True)` on the embedding output.
- [ ] attr_i(e) = −Σ_t ⟨grad, Δy⟩ keeping the example axis.
- [ ] `down_proj`: z via the `NeuronCollector` pre-hook convention, c_t via `lora_A` output, a from saved factors; attr_j(e) = −a_j Σ_t z_{j,t} ∂M/∂c_t, reduced in the backward hook so z is held for one batch only.
- [ ] Enforce Σ_j attr_j = attr_i per example; fail the unit otherwise.
- [ ] FP64 accumulation, FP32 storage; neuron results to `BlockStore`.
- [ ] GPU-side reductions: no `.item()`/`.cpu()` inside hooks; one async copy to pinned host memory per batch; block writes on a background thread.
- [ ] Length-bucketed packing to a token budget given by the plan (set in 05b), not a fixed batch size.
- [ ] Tests (tiny random Qwen2 + PEFT, rank 1 on all seven projections): logits unchanged with hooks; untuned → 0; finite differences (5 adapters, 3 neurons); batch 1 vs 8; gauge k ∈ {−2, 0.5}; mutation tests (wrong layer, wrong sign, mean-not-sum) each fail.

## Comments

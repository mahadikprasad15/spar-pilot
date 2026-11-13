# 03: Adapter and neuron-to-adapter attribution instrument (CPU)

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** `pilot_eval/attribution_instrument.py`: hooks on live PEFT LoRA branches and on the `down_proj` input / `lora_A` output; per-example adapter attribution (Stage A) and neuron-to-adapter attribution (Stage B) from one backward of Σ_e M(e).

**Blocked by:** 02.

- [ ] Hook the branch output s·B(A x) of each live LoRA module (resolve the module path from the installed PEFT; fail if the branch is fused or missing); `retain_grad()`; no parameter gets `requires_grad`.
- [ ] attr_i(e) = −Σ_t ⟨grad, Δy⟩, keeping the example axis; FP64 accumulation; store FP32.
- [ ] Stage B at given layers: z_t via the `NeuronCollector` pre-hook convention, c_t via `lora_A` output; attr_j(e) = −a_j Σ_t z_{j,t} ∂M/∂c_t, with a read from the saved factors.
- [ ] Enforce Σ_j attr_j = attr_i per example within the plan tolerance; fail the unit otherwise.
- [ ] Tests (tiny random Qwen2 + PEFT, rank 1 on all seven projections): logits unchanged with hooks; untuned → 0; finite difference for 5 adapters and 3 neurons; batch size 1 vs 8 identical; gauge k ∈ {−2, 0.5}; mutation tests (wrong layer, wrong sign, mean-not-sum backward) each fail a test.

## Comments

# 04: Ablation engine and condition builder

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 02

**What to build:** `pilot_eval/attribution_ablation.py`: in-memory removal hooks, the 64-neuron read, the constant read, the write-matched sampler, edited checkpoints, and a builder that reads only a sealed selection.

- [ ] Remove any adapter set (s·b → 0).
- [ ] Rebuild c_t from a given neuron subset, rescaled to the discovery mean |c_t|; constant read c_t → c̄.
- [ ] Write-matched sampler: same module type, ±tolerance, seeded; unavailable rather than loosened.
- [ ] Edited checkpoint (chosen b zeroed) with hash; logits equal the in-memory hook.
- [ ] Builder refuses unsealed selections or any confirmation-derived input.
- [ ] Conditions switch hooks on one loaded model (all six adapters via PEFT multi-adapter); no reload per condition.
- [ ] Generation path: length-sorted batches at the plan's generation batch size, left padding, per-sequence stop at EOS.
- [ ] Tests: all-neuron read == full; empty removal == full; constant read writes s·b·c̄; rescale factor; checkpoint == hook; sampler determinism and unavailable path.

## Comments

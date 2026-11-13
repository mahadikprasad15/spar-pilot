# 04: Ablation engine and condition builder (CPU)

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** `pilot_eval/attribution_ablation.py`: in-memory ablation hooks and a condition builder that reads only a sealed selection.

**Blocked by:** 02.

- [ ] Adapter removal (s·b → 0) for any set; band necessity and band-only sufficiency on the untuned model.
- [ ] Masked read at a `down_proj` adapter: drop-top-m, keep-top-m with rescaling to the discovery mean |c_t|, constant c̄.
- [ ] Write-matched random control sampler (same module type, ±tolerance, seeded); returns unavailable rather than loosening.
- [ ] Edited-checkpoint writer (chosen b zeroed) with hash; check its logits equal the in-memory hook.
- [ ] Condition builder refuses unsealed selections and any confirmation-derived input.
- [ ] Tests: keep-all == full, keep-none == removed, constant-c̄ write == s b c̄, rescale factor correct, checkpoint == hook, sampler determinism and unavailable path.

## Comments

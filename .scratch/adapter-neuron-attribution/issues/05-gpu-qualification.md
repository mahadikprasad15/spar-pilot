# 05: Real-GPU qualification and budget

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** A qualification run on the Pod before production, reusing the joint audit's preflight, export and Pod chain.

**Blocked by:** 01, 03, 04.

- [ ] FP32, TF32 off, eager attention, backend recorded; all six arms load against inventory-v2 (reuse `check_live_adapter`).
- [ ] Finite-difference check on the real model: 5 adapters, 3 neurons per selected-layer candidate, one arm per family (SFT, GRPO). Record relative error; tolerance proposed in the plan, approved before production.
- [ ] Batch invariance: per-example attributions at two token budgets on 20 examples; record the maximum discrepancy.
- [ ] Time one arm end-to-end for Stage A, B and a 5-condition Stage C; project totals; peak memory.
- [ ] Verified partial export of the qualification outputs.
- [ ] User sets the hard wall-time cap after reading this record.

## Comments

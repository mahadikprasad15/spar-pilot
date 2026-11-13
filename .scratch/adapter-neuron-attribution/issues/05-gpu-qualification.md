# 05: GPU qualification

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 01, 03, 04

**What to build:** A short run on the Pod before production, using the joint audit's preflight, Pod chain and export.

- [ ] FP32, TF32 off, eager; all six arms load against inventory-v2 (`check_live_adapter`).
- [ ] Finite differences on the real model: 5 adapters and 3 neurons, for `gold-sft` and `online-grpo`; record relative errors.
- [ ] Batch invariance: per-example attributions at two token budgets on 20 rows; record the maximum discrepancy.
- [ ] Time one arm through A+B, 5 Stage C conditions and 10 generations; peak memory; projected totals.
- [ ] Propose tolerances from these measurements; user approves them and sets the wall-time cap before 06.
- [ ] Verified partial export.

## Comments

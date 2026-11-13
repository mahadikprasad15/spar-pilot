# 05b: Pod qualification, utilization targets and budget

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 05a

**What to build:** Qualification on the production Runpod A100 80 GB with the joint audit's preflight, Pod chain and export, then a small Pod rehearsal with an interrupted resume.

- [ ] Preflight: container disk ≥ 60 GB free, persistent volume identity, offline model load in the worker, online export write/readback.
- [ ] All six arms load against inventory-v2 (`check_live_adapter`); FP32, TF32 off, eager; runtime recorded.
- [ ] Finite differences on the real model (5 adapters, 3 neurons; `gold-sft`, `online-grpo`); batch invariance at two token budgets on 20 rows. Propose tolerances; user approves.
- [ ] Token-budget sweep (16k–256k, forward-only and forward+backward) with a longest-sequence stress batch; choose the largest with peak memory ≤ 80% and no OOM.
- [ ] Generation batch sweep (16/32/64/150); choose the largest that fits; record greedy-output differences across batch settings on 20 items.
- [ ] Utilization check against the plan targets (median busy ≥ 85% for A+B and C, peak memory 50–80%, padding ≤ 15%). A miss is investigated with the profiler trace before production.
- [ ] Optional precision comparison: attribution and ΔM on 20 rows under FP32 vs TF32, for the user's TF32 decision.
- [ ] Pod rehearsal: 2 arms, a few units, interrupted after one unit, local dir deleted, resumed from the volume, clean-root restore verified. Labelled `rehearsal-pod`.
- [ ] Time projection for 06–08 from measured throughput; the user sets the hard wall-time cap.

## Comments

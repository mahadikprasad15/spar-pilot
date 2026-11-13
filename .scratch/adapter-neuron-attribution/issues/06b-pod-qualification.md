# 06b: Pod qualification and preparation

**Status:** needs-triage — draft v3
**Type:** task
**Blocked by:** 06a
**Authorization:** an approved 06b budget. Pod stop needs separate authorization and a verified backup.

- [ ] Preflight: container disk ≥ 60 GB free, persistent volume identity, offline model load in the worker, online export write/readback.
- [ ] All six arms load against inventory-v2; FP32, TF32 off, eager; runtime recorded.
- [ ] On calibration rows: finite differences (5 adapters, 3 neurons; gold-sft, online-grpo); batch invariance at two budgets; propose tolerances and the resolution floor.
- [ ] Token-budget sweep 2k–64k on the real path; record failures; stop if none fits ≤ 80% memory.
- [ ] Generation batch qualification on the 20 rehearsal items (batch 1 vs 8/16/32/64, full cap, longest first, KV peak); freeze the batch manifest.
- [ ] Blind base-trace generation for discovery and confirmation at the qualified batch; seal; check minimums. Confirmation outputs other than the traces are not computed.
- [ ] Resume rehearsal: interrupt, delete local state, restore from the volume, clean-root restore verified.
- [ ] Utilization summary and time projection; user approves the production cap.

## Comments

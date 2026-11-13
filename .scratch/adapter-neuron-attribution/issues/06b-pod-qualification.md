# 06b: Overnight Pod chain (execution)

Run by the chain runner (ticket 11) together with 07–10, unattended, from an approved launch record. This ticket defines the qualification and preparation stages of that chain.

**Status:** needs-triage — draft v3
**Type:** task
**Blocked by:** 06a
**Authorization:** the user's approval of the launch record (spec §Overnight unattended chain); an attended Pod rehearsal of the chain first unless waived. The Pod is stopped at the end of every run.

- [ ] Preflight: container disk ≥ 60 GB free, persistent volume identity, offline model load in the worker, online export write/readback.
- [ ] All six arms load against inventory-v2; FP32, TF32 off, eager; runtime recorded.
- [ ] On calibration rows: finite differences (5 adapters, 3 neurons; gold-sft, online-grpo) and batch invariance at two budgets, checked against the launch record's fixed tolerances; automatic sign-off or stop.
- [ ] Token-budget sweep 2k–64k on the real path; record failures; stop if none fits ≤ 80% memory.
- [ ] Generation batch qualification on the 20 rehearsal items (batch 1 vs 8/16/32/64, full cap, longest first, KV peak); freeze the batch manifest.
- [ ] Blind base-trace generation for discovery and confirmation at the qualified batch; seal; check minimums. Confirmation outputs other than the traces are not computed.
- [ ] Resume rehearsal: interrupt, delete local state, restore from the volume, clean-root restore verified.
- [ ] Utilization summary and time projection feed the overrun rule; production continues automatically after sign-off.

## Comments

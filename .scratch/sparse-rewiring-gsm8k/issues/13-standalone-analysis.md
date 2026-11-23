# 13: Standalone analysis of each rank

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03, 04

Requested by the researcher on 2026-11-23 (D22). CPU analysis of each arm on its own, built on fixtures now and run on real outputs after ticket 09. Full table in `spec.md` §Standalone analysis.

- [ ] Selection: survivor counts and per-layer distribution; gate dynamics and the settle step; stability of survivors across checkpoints.
- [ ] What survivors learned: gauge-invariant change size, gate vs up; rank 1, shared `a` per layer and its absolute cosine with gold LoRA's gate and up `lora_A`; rank 0, effective rank of survivor rows; activation shift vs base on validation (the forward pass runs in ticket 09).
- [ ] Behaviour: validation CE and accuracy; cohort150 at 64 and final with invalid, capped and length; per-question flips vs gold-sft and untuned with paired bootstrap.
- [ ] Red/green: spec test 16.
- [ ] Definitions marked [proposed] (settle step, effective rank) need the researcher's OK before the report is final.

## Comments

## Answer

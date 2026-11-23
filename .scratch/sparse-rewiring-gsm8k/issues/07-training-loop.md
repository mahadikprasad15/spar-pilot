# 07: Training loop

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03, 05, 06
**Group:** D

Adapt the explicit loop in `sampling_sft_training_backend.train`: two AdamW groups, warmup + cosine to 1,024, read-only clipping, L0 term, freeze at 960, shared-mask groups, dynamics logging. Keep every check the existing loop makes.

- [ ] Red/green: spec tests 7, 9, 10 and 19.
- [ ] Validation CE every 16 steps through `completion_loss` (hook supplied by ticket 09; a stub here).

## Comments

## Answer

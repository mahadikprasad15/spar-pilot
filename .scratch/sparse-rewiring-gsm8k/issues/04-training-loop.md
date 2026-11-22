# 04: Training loop

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 02, 03

Explicit training loop adapted from `sampling_sft_training_backend.train`, with two AdamW groups, shared `LambdaLR`, read-only clipping, L0 term, fixed-mask phase, frozen order and full recovery state. Every numeric value comes from the plan; none is a code default.

- [ ] Red/green: spec tests 7–11.
- [ ] Values stay placeholders in fixtures until D1–D7, D10, D14, D15 are decided; the plan schema refuses a missing value.

## Comments

## Answer

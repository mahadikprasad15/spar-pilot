# 04: Training loop

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 02, 03

Explicit training loop adapted from `sampling_sft_training_backend.train`, with two AdamW groups, shared `LambdaLR`, read-only clipping, L0 term, fixed-mask phase, frozen order and full recovery state. Every numeric value comes from the plan; none is a code default.

- [ ] Red/green: spec tests 7–11.
- [ ] Mask-noise groups (D14, proposed): one cached `u` per layer per 4 consecutive examples, reused under checkpoint recompute; spec test 10a.
- [ ] Write the dynamics logs from `spec.md` §Training dynamics: per-step gate totals and gauge-invariant read-change norms, and every 8 steps (D20) the validation CE, per-layer counts and an FP16 `logα` snapshot.
- [ ] Values stay placeholders in fixtures until D1–D7, D10, D14, D15 are decided; the plan schema refuses a missing value.

## Comments

## Answer

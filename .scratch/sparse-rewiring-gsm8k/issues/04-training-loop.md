# 04: Training loop

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 02, 03

Explicit training loop adapted from `sampling_sft_training_backend.train`, with two AdamW groups, shared `LambdaLR`, read-only clipping, L0 term, fixed-mask phase, frozen order and full recovery state. Every numeric value comes from the plan; none is a code default.

- [ ] Red/green: spec tests 7–11.
- [ ] Mask-noise groups (D14, proposed): one cached `u` per layer per 4 consecutive examples, reused under checkpoint recompute; spec test 10a.
- [ ] Write the dynamics logs from `spec.md` §Training dynamics: every step (D20, decided) the gate totals, per-layer counts and gauge-invariant read-change norms; every 16 steps the validation CE and an FP16 `logα` snapshot.
- [ ] Adaptive stop (D5): cosine for a 1,024-step maximum; stop checks every 64 steps from 256 on the hard set and expected-active, written to `logs/stop.jsonl`; freeze at the settling check and stop 64 steps later, or freeze at 960 and stop at 1,024; record the stop step and the LR at the stop. Thresholds come from the plan. Spec test 18.
- [ ] Values stay placeholders in fixtures until D1–D7, D10, D14, D15 are decided; the plan schema refuses a missing value.

## Comments

## Answer

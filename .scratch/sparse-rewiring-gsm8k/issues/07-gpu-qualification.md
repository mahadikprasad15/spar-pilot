# 07: Gpu qualification

**Status:** ready-for-human
**Type:** task
**Blocked by:** 04, 06

**Needs the researcher's approval and one A100-80GB (D18, decided).** Qualification only, not the scientific run.

- [ ] Real longest training example for **both rank 1 and rank 0**: forward/backward, peak memory (rank 0 adds about 12 GB), IEEE precision readback.
- [ ] Zero-update logits identity vs untuned on the real model; base hash.
- [ ] A disposable few-step run under a separate name; save/restore and resume check; throughput per step, validation pass and greedy answer.
- [ ] Time one validation CE pass, one greedy validation pass and one single-survivor revert (for D23's estimate).
- [ ] Replace the arithmetic estimate in `decision-log.md` with measured values.

## Comments

## Answer

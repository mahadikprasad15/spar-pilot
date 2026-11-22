# 07: Gpu qualification

**Status:** ready-for-human
**Type:** task
**Blocked by:** 04, 06

**Needs the researcher's approval and a GPU (D18).** Qualification only, not the scientific run.

- [ ] Real longest training example: forward/backward, peak memory, IEEE precision readback.
- [ ] Zero-update logits identity vs untuned on the real model; base hash.
- [ ] A disposable few-step run under a separate name; save/restore and resume check; throughput per step, validation pass and greedy answer.
- [ ] Replace the arithmetic estimate in `decision-log.md` with measured values.

## Comments

## Answer

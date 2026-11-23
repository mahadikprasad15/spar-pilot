# 15: GPU qualification

**Status:** ready-for-human
**Type:** task
**Blocked by:** 14, 20
**Group:** G

**Needs the researcher's approval and one A100-80GB.** Qualification only, not the scientific run, and validation-only (no cohort150).

- [ ] Longest real example forward/backward for both ranks; peak memory; IEEE readback; base hash.
- [ ] Zero-update logits identity vs untuned on the real model.
- [ ] A disposable few-step run under a separate name: save/restore, resume on a fresh directory, throughput, one validation CE pass, one greedy validation pass, one single-survivor revert timing.
- [ ] Replace the arithmetic estimates in `decision-log.md` with measured values. An OOM stops, with no automatic fallback (ADR 0004).

## Comments

## Answer

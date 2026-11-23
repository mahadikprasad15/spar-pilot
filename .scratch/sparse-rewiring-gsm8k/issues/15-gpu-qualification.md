# 15: GPU qualification

**Status:** ready-for-human
**Type:** task
**Blocked by:** 14, 20
**Group:** G

**Needs the researcher's approval and one A100-80GB.** Qualification only, not the scientific run, and validation-only (no cohort150).

- [ ] Reuse the existing harness (`spec.md` §FP32 and qualification): `Stages.preflight`, `TransformersBackend.preflight`, `noop_check`, `processor_check`, `cached_vs_forced`, the extended `HFTrainingEngine.preflight` and the memory rule. New checks only for the shared-mask cache and the D24 disk total.
- [ ] Longest real example forward/backward for both ranks; peak memory; IEEE readback; base hash.
- [ ] Zero-update logits identity vs untuned on the real model.
- [ ] A disposable few-step run under a separate name: checkpoint save and reload, a restart in a fresh directory that reproduces the first steps, throughput, one validation CE pass, one greedy validation pass, one single-survivor revert timing.
- [ ] Replace the arithmetic estimates in `decision-log.md` with measured values. An OOM stops, with no automatic fallback (ADR 0004).

## Comments

## Answer

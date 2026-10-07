# 02: Save the untuned training-cohort sampling baseline

**What to build:** Sample eight completions per problem on a frozen 128-problem
subset of the training cohort and expose auditable reward, length and dead-group
evidence for resolving the full-run length cap.

**Blocked by:** 01 — Verify source evidence and prepare the GRPO matching audit.

**Status:** ready-for-agent

- [ ] Require the explicit source/precision/scorer/subset/pilot-generation
      settings necessary for this stage; reject unresolved inputs rather than
      blocking independent preparation or choosing defaults.
- [ ] Batch the 1,024 generations with frozen prompt/group/draw IDs, order,
      seeds and effective sampling settings; save token IDs, text, stop/cap
      status, strict/flexible extraction and reward for every completion.
- [ ] Import the selected repository scorer and verify strict-correct implies
      flexible-correct; never copy a last-number rule into the reward function.
- [ ] Produce length percentiles, capped/censored counts, per-group correct
      counts, dead-group fraction and strict/flexible reward summaries.
- [ ] Refuse to justify a final cap from a percentile censored at the pilot
      limit; allow an explicitly named, preserved sampling extension.
- [ ] Recover verified partial sampling without duplicated draw IDs or
      regeneration of completed groups; display exit-aware live progress.
- [ ] Exercise the workflow offline using controlled generation, interruption,
      capped/ambiguous answers, mismatched scorer and immutable artifact tests.
- [ ] Provide notebook guidance distinguishing this training-cohort baseline
      from held-out evaluation and from the deferred full-cohort X1 baseline.

## Comments

- October 7, 2026: approved as ticket 2. Final cap selection remains an explicit
  evidence-based freeze decision; this ticket does not hardcode that value.

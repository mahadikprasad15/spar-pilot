# 02: Save the untuned training-cohort sampling baseline

**What to build:** Sample eight completions per problem on a frozen 128-problem
subset of the training cohort and expose auditable reward, length and dead-group
evidence for resolving the full-run length cap.

**Blocked by:** 01 — Verify source evidence and prepare the GRPO matching audit.

**Status:** resolved

- [x] Require the explicit source/precision/scorer/subset/pilot-generation
      settings necessary for this stage; reject unresolved inputs rather than
      blocking independent preparation or choosing defaults.
- [x] Batch the 1,024 generations with frozen prompt/group/draw IDs, order,
      seeds and effective sampling settings; save token IDs, text, stop/cap
      status, strict/flexible extraction and reward for every completion.
- [x] Import the selected repository scorer and verify strict-correct implies
      flexible-correct; never copy a last-number rule into the reward function.
- [x] Apply the explicit reward-zero rule for every capped completion, even
      with a numeric answer, and display its potential shorter-answer incentive
      alongside cap fractions and censored percentiles.
- [x] Produce length percentiles, capped/censored counts, per-group correct
      counts, dead-group fraction and strict/flexible reward summaries.
- [x] Refuse to justify a final cap from a percentile censored at the pilot
      limit; allow an explicitly named, preserved sampling extension.
- [x] Recover verified partial sampling without duplicated draw IDs or
      regeneration of completed groups; display exit-aware live progress.
- [x] Exercise the workflow offline using controlled generation, interruption,
      capped/ambiguous answers, mismatched scorer and immutable artifact tests.
- [x] Provide notebook guidance distinguishing this training-cohort baseline
      from held-out evaluation and from the deferred full-cohort X1 baseline.

## Comments

- October 7, 2026: approved as ticket 2. Final cap selection remains an explicit
  evidence-based freeze decision; this ticket does not hardcode that value.

## Answer

Implemented `grpo-baseline` with an explicit settings file, 128 × 8 frozen
training-cohort draws, one model load, batched HF sampling and repository strict
plus flexible-v3 scores. Every draw saves IDs, prompt/gold, tokens, stop/cap
status and reward; cap-zero is explicit. Summary reports lengths, censoring,
invalids, correct counts and dead groups without selecting a final cap.

Verified completed batches are reused; an interrupted unsealed batch is
regenerated with its saved seed. Frozen source, settings, code/scorer and
runtime evidence prevent mixed reuse. No new training resume infrastructure
is introduced. Extensions use separate preserved names.

The guided notebook adds pinned dependency installation, explicit settings,
sampling/resume and evidence interpretation. Scientific settings are not
silently approved or filled into the prepared full-run placeholders.

Red/green checks covered missing public workflow, scorer implementation binding,
inconsistent cap metadata, cached runtime integrity and notebook handoff.
Interruption/corruption/scorer/censored-answer tests passed. Final affected
suite: 35 passed in 36.53 seconds (CPU controlled generation, no downloads).
Evidence: artifacts/runs/diagnostics/pilot4-ticket02-v1/.
Actual GPU sampling/speed/memory remain unverified. Commits are local.

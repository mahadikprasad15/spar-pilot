# 01: Verify source evidence and prepare the GRPO matching audit

**What to build:** Prepare a Pilot 4 plan from verified SFT and activation
source artifacts, exposing actual matches and unresolved decisions before any
scientific model run. The command and guided notebook show the same audit.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Verify the selected source configs, completion/checkpoint manifests,
      cohort identities and ordered 512 training/150 evaluation IDs without
      downloading a model; preserve every source artifact.
- [x] Reuse existing model/tokenizer/dataset pins, actual prompt template,
      optimizer and adapter settings, and fixed measurement-sequence identities.
- [x] Save a versioned matching audit and preparation manifest under the chosen
      artifact root; repeated preparation verifies/reuses them.
- [x] Record the approved FP32 choice for both arms and the superseded BF16
      source text, verifying actual base/adapter dtypes. Display the historical-
      versus-measured baseline distinction and unresolved reward version.
- [x] Represent pending settings explicitly and reject attempts to freeze/run
      an unresolved plan. Preserve source-derived known values and rationales.
- [x] Add offline public-workflow tests for coherent preparation, source
      corruption, wrong cohorts, missing evidence and incompatible reuse.
- [x] Supply a notebook stage explaining the audit and concrete next action.

## Comments

- October 7, 2026: approved as ticket 1 of the nine-ticket plan. Parent: Pilot 4
  rank-1 GRPO specification. No old pilot configuration is modified.
- October 7 amendment: the precision decision is closed by user approval of
  FP32; source/runtime verification remains required, not another precision vote.
- October 7 implementation: claimed; public preparation/audit commands are
  tested against real source artifacts with controlled offline fixtures.

## Answer

Implemented CPU-only grpo-prepare, grpo-audit and grpo-check-ready through
the public workflow. Sources, saved optimizer prompt order, fixed measurement
tokens/masks and checkpoint manifests are verified without model access. The
guided Pilot 4 notebook includes the preparation and audit stages; later
scientific stages are explicitly not implemented yet.

Red/green evidence includes preparation, failure-state persistence, source
order preservation, execution-freeze rejection, notebook command execution
and exact full-name projection compatibility (rejecting lm_head). SFT memory
microbatches are retained as source evidence, not copied into GRPO's completion
batch settings; the plan states 8 prompts and 64 completions per update.
The final affected offline suite passed 27 tests in 26.58 seconds.
This is fixture evidence, not validation of the user's Drive or target GPU.

Diagnostic results live under the canonical ignored artifact tree; source
artifact bytes are preserved and incompatible prepared plans are rejected.

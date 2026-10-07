# 01: Verify source evidence and prepare the GRPO matching audit

**What to build:** Prepare a Pilot 4 plan from verified SFT and activation
source artifacts, exposing actual matches and unresolved decisions before any
scientific model run. The command and guided notebook show the same audit.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Verify the selected source configs, completion/checkpoint manifests,
      cohort identities and ordered 512 training/150 evaluation IDs without
      downloading a model; preserve every source artifact.
- [ ] Reuse existing model/tokenizer/dataset pins, actual prompt template,
      optimizer and adapter settings, and fixed measurement-sequence identities.
- [ ] Save a versioned matching audit and preparation manifest under the chosen
      artifact root; repeated preparation verifies/reuses them.
- [ ] Record the approved FP32 choice for both arms and the superseded BF16
      source text, verifying actual base/adapter dtypes. Display the historical-
      versus-measured baseline distinction and unresolved reward version.
- [ ] Represent pending settings explicitly and reject attempts to freeze/run
      an unresolved plan. Preserve source-derived known values and rationales.
- [ ] Add offline public-workflow tests for coherent preparation, source
      corruption, wrong cohorts, missing evidence and incompatible reuse.
- [ ] Supply a notebook stage explaining the audit and concrete next action.

## Comments

- October 7, 2026: approved as ticket 1 of the nine-ticket plan. Parent: Pilot 4
  rank-1 GRPO specification. No old pilot configuration is modified.
- October 7 amendment: the precision decision is closed by user approval of
  FP32; source/runtime verification remains required, not another precision vote.

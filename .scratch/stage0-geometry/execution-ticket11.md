# Ticket 11: audited untuned rollout completion and cohorts

## Implemented

- Public `stage0-rollouts-{audit,prepare,runtime,profile,profile-verify,freeze,run,
  status,verify,recover-writer}` commands. Only Ticket 01 source readiness is a
  prerequisite; primary/conditional intervention execution is not required.
- Used-file historical audit: original source/settings hashes, all raw eight-draw
  groups/seals/runtime, common 512 membership, prompt/gold consistency and explicit
  old ordering/seeds. Missing raw files are reported without treating aggregate
  reports as response evidence or loading a model.
- Local-only HF FP32 untuned backend. Reuses the historical HFSampler sampling
  method, matching explicit settings and input IDs. Actual base hashing maps
  pure-model parameter names to the audited PEFT naming convention; parameter
  guards and full final integrity protect each attempt. No adapter is loaded.
- User amendment persisted in design/spec on 2026-10-27: exact replay of two saved
  groups × eight draws, with source/settings identities/hashes and available
  historical GPU/software identities matched. Exact success permits reuse128 +
  missing384. Any mismatch or unrecoverable seed/order selects fresh512 in one
  recorded session, with original128 separately archived. Never use reward rates
  or a newly fitted numeric envelope to justify a join.
- Bounded measured profile precedes a separately reviewed production freeze.
  Model-loading/guard time and cap-extrapolated group cost are recorded. Profile
  groups with matching new production IDs/seeds can be reused. Forecasts are not
  completion guarantees. All allocations share the existing cumulative 20-unit
  ledger; another job's active reservation cannot fund R0.
- Group-level atomic seals, explicit source/runtime/input/code identity, resumed
  missing work, cumulative attempt timing and owner-token recovery. Failed full
  integrity invalidates already sealed groups from that attempt. Re-audit of a
  prepared/completed product returns saved verified status rather than overwriting
  completion with an earlier source-audit lifecycle state.
- Raw full JSONL plus flat Parquet; all three scorers, cap/EOS metadata, old/new
  origin and historical training rewards retained. Cohort memberships retain
  zero-success questions, observed k/8 and separate whole-answer negative-log(k/8).
  One correct draw is selected uniformly with dedicated NumPy PCG64 seed42,
  saved question order and sorted eligible draw indices.
- CPU notebook 29–31 and GPU notebook 20–25, with visible source paths, exact
  profile scope, measured-budget reviews, route choice and reconciliation.
  Parquet support is checked on CPU before sampling; optional dependency extra
  `stage0-rollouts` supplies PyArrow. NLL/entropy/gradient stages are not run here.

## Test evidence

- Startup affected baseline: 20 passed in 33.27s.
- Initial math test red before cohort/draw module; source audit test red before
  public CLI wiring. First green slice: 2 passed in 2.14s.
- Public profile/production tests initially red before their orchestration existed.
  A missing profile reservation context was exposed and fixed without changing
  budget or acceptance rules. Both end-to-end routes then passed (4 focused tests
  in 22.95s).
- Runtime command and notebook handoff tests red before implementation, then green.
- Resume tests exposed a re-audit status downgrade and a collection-phase lifecycle
  left as running after global completion; both were corrected and asserted.
- Expanded focused suite: 8 passed in 24.66s before the final recovery/state checks.
- First affected regression: **52 passed in 79.18s**, covering rollouts, original
  baseline, shared source/budget workflow, qualification and paired collection.
  This is not a full legacy-suite run.
- Tests include exact/mismatched replay, missing seed fallback, timing-group reuse,
  reversed historical IDs (to catch source-index seed substitution), group resume,
  old-source corruption, unknown/exhausted budget, changed session before loading,
  failed final integrity, explicit writer recovery, scorer/stop/cohort semantics,
  Parquet reconstruction, parameter-name bridge and clean notebook handoff.

## Persisted evidence and actual execution status

`artifacts/validation/stage0-ticket11/` contains labelled synthetic reuse and
uniform-fallback snapshots, their 4096-row tables/cohort memberships/reports,
regression output and structured validation. Deliberate source corruptions in
retained positive snapshots were restored from the original sealed fixture.

No real model was downloaded or loaded, no GPU/Colab run occurred, no credits
were spent, and no Drive upload or GitHub push was performed. The local historical
report snapshot lacks the raw source run; CPU section29 must audit actual durable
inputs before scientific execution. Numerical/source procedure rules were not
relaxed to make a test pass. The only scientific amendment is the user's explicit
exact-replay/fresh512 fallback decision, preserved in design/spec.

## Real-schema check before release

Read the retained real `grpo.training.json` and `sft.config.json` snapshots. GRPO
stores the cohort as 64 lists of eight IDs in `optimizer_groups`, rather than
SFT's `training_indices`. Added an independent schema test (red before the
validator existed; green after implementation) and corrected real-input checking
to validate both representations, their source hashes and ordered membership.
This prevents valid real GRPO metadata from being rejected by a synthetic-only
assumption. Final regression and release pin are recorded below. Earlier
synthetic snapshots are retained; the final verified snapshots use v2 names.

## Final release validation

- Final affected regression after the real-schema correction: **53 passed in
  84.56s**. Final R0 release check after adding model-load/hash progress messages:
  **13 passed in 26.53s**. No full legacy-suite claim.
- Final verified snapshots: `replay-reuse-synthetic-v2` and
  `uniform-fallback-synthetic-v2`, each with exactly 4096 target draws and
  reconstructed JSONL/Parquet/memberships. Earlier v1 fixtures are retained as
  pre-schema-fix development evidence; v2 is the release validation source.
- Both notebooks parse and contain no saved outputs; diff checks pass.

- Verified scientific implementation/notebook pin: `ccf026eea7baa25b7a08963b7051c642ef928dfc`.

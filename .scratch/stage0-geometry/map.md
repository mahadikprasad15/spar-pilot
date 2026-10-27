# Stage 0 design review

## Notes

- Original proposal: `sources/stage0-design-supplied-2026-10-26.md`.
- Working design and decision tree: `design.md`.
- User requested grill-with-docs before implementation/specification.
- User subsequently invoked to-spec and to-tickets. Synthesis is in
  `spec.md`; the approved slice breakdown is in `ticket-proposal.md`.
- The user approved the public saved-workflow test boundary, all 15 slices and
  their dependency edges on 2026-10-26. Spec and individual `ready-for-agent`
  issues are published in this local tracker. Tickets 01-13 are implemented;
  real source/weight execution remains separately pending.

## Decisions so far

- E0.4 intervention is the main result. Supporting experiments run if the GPU
  session has room and do not block it. Necessary edit construction, validation,
  controls and outcome measurements remain prerequisites. Conditional V6/V6c
  requires E0.3 classification, not completion of the whole supporting chain.
- Preserve the supplied proposal separately from review findings.
- No new Stage 0 GPU execution has started.
- User amended E0.6a: NLL primary, descriptive excess surprise secondary,
  separate whole-answer negative-log correctness probability; retained +/-0.02
  own_all acceptance threshold.
- User amended GPU policy: available suitable GPU, FP32/TF32-off, within-session
  pairing, new write measurements alongside new KL. Common-input compatibility
  replaces the disjoint-cohort reward-rate gate; absent an applicable independent
  envelope, historical and new results are reported separately.
- Always measure subspace rotation/intruders; add update-to-gap ratio. Intruder
  overlap uses m = max(k50, 40), with k30/k70 sensitivities also floored at 40;
  norm threshold 0.5 primary, 0.3-0.7 sweep. Mark thresholds at/below sqrt(m/d_out)
  undecidable and report that reference next to overlaps.
- Save raw gradients; push is their negative, with saved A0 factor orientation.
- Save all energy components and cross term; headline fraction excludes cross term.
- Conditional V6 is a merged-weight edit, with an explicit merged-weight loader.
- V6 halves all primary-flagged directions for either arm with eligible flags.
  Edited modules disable their adapters; other modules keep theirs. V6c matches
  counts and nearby singular values using non-flagged directions in the same modules.
- V6c control candidates are new top-40; detection stays top-20. Minimum total
  absolute singular-value difference with index tie-break; log pair ratios and
  repeat matched edits with only ratios in [0.8, 1.25]. Short-control modules
  excluded from both matched edits; full V6 remains separately reported.
- Merged loader: bounded 16-sequence logit/greedy diagnostic, double-adapter
  negative control with >=1000x error, hard stop above 1e-3; propose allowance
  3x observed maximum for user approval. Near-tie exceptions require common-prefix
  margin evidence and do not constitute exact token agreement.

- All six final proposals approved: Pilot 4 evaluation/targeted audit; lower
  95% bound > -5pp for non-inferiority; per-module V1 and V2 size controls;
  common-input historical checks; measured budgets with a TOTAL 20-unit ceiling;
  numerical/reporting conventions. See design.md final approval section.

## Implementation frontier and remaining evidence

- Tickets 01-03 are implemented; see their execution records. Ticket 04 software
  is implemented and locally tested; real Colab qualification and its evidence
  reviews remain pending. Ticket 05 implements paired greedy collection and CPU
  raw-result verification; real collection remains pending. Ticket 06 implements fixed-context KL and effective-write collection and CPU
  verification; real collection remains pending. Ticket 07 implements paired
  primary findings, targeted intended-answer review and immutable report
  snapshots. The primary software path is complete; real execution is pending. Primary path:
  01 -> 02 -> 03 -> 04 -> (05,06) -> 07.
- Conditional intruder extension: 02 -> 08; (03,08) -> 09; (07,09) -> 10.
- Optional branches: 01 -> 11 -> 12; (02,11) -> 13; (02,04) -> 14;
  (02,06) -> 15. None gates the primary report.
- Verify source tensors, cohort membership, runtime metadata and KL completion.
- Implement approved contracts through red/green CPU tests and incrementally
  runnable shared notebooks; write/freeze the complete preregistration before
  relevant result inspection.
- Numerical qualification and measured job budgets retain their explicit approval
  stop points. Design approval is not evidence that those gates have passed.

- Ticket 07 execution/validation record: `execution-ticket07.md`; 73 affected
  tests passed on 2026-10-27. Ticket 08 implements rotation/intruder geometry;
  its real CPU execution is pending. Conditional Ticket 09 is now implemented; Ticket 10 is also implemented; optional Ticket 14 is the next implementation
  frontier. Neither conditional nor supporting work is required to run the primary path.

- Ticket 08: CPU rotation/intruder geometry implemented; `execution-ticket08.md`.
  No small-update skip, top-20 detection/top-40 matching pool, floored subspaces,
  chance-aware eligibility and explicit degeneracies. Ticket 09 consumes this
  verified output for conditional edit construction; no V6 run is implied.
  Final affected regression: 44 tests passed on 2026-10-27.

- Ticket 09: conditional full/matched/ratio merged matrices and the mixed-loader
  interface are implemented; see `execution-ticket09.md`. Real construction,
  mixed-loader numerical qualification and edited-model execution remain pending.
  CPU notebook sections 26–28 expose preparation, bounded resume and verification.

- Ticket 10 connects conditional catalogue qualification to the existing paired
  evaluation, fixed-context KL/effective-write, targeted review and report paths.
  The extension has separate names/families and preserves the single cumulative
  20-unit ledger. Each distinct merged mask uses the same frozen 16 prompts;
  identical layouts are reused. Real qualification/measurement remain pending.
  See `execution-ticket10.md` for software validation and execution boundaries.

- Ticket 11: audited R0 sampling, exact two-group replay, user-approved fresh512
  fallback, bounded budgets/resume, JSONL/Parquet and verified derived cohorts are
  implemented. CPU 29–31 / GPU 20–25. `execution-ticket11.md` records 53 affected
  passing tests; real source recovery and GPU execution remain pending. Ticket 12
  consumes verified memberships for optional per-answer NLL/entropy measurements.

- Ticket 12: unique-answer teacher-forced NLL, entropy/rank/excess-surprise
  collection and CPU reports are implemented, with SFT/end-token mapping,
  separately frozen measured budgets, same-sampled-runtime enforcement, saved
  problem bootstraps, positional summaries and unchanged ±0.02 gate. CPU32–33 /
  GPU26–31. `execution-ticket12.md` separates software validation from pending
  real GPU scoring and scientific gate evaluation. Ticket13 is implemented; Ticket14 is next.

- Ticket13: matched initial CE/GRPO gradients, exact A₀ audit, independent
  reconstruction, measured budgets, resume, CPU summaries and prediction-bound
  endpoint reveal are implemented. CPU34–36 / GPU32–37. See
  `execution-ticket13.md`; real collection remains pending.

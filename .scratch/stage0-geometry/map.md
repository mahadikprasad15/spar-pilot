# Stage 0 design review

## Notes

- Original proposal: `sources/stage0-design-supplied-2026-10-26.md`.
- Working design and decision tree: `design.md`.
- User requested grill-with-docs before implementation/specification.
- User subsequently invoked to-spec and to-tickets. Synthesis is in
  `spec.md`; the approved slice breakdown is in `ticket-proposal.md`.
- The user approved the public saved-workflow test boundary, all 15 slices and
  their dependency edges on 2026-10-26. Spec and individual `ready-for-agent`
  issues are published in this local tracker. Tickets 01-03 are implemented;
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
  reviews remain pending. Primary path:
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

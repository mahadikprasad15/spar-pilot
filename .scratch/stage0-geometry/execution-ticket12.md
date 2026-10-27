# Ticket 12: teacher-forced NLL, entropy and excess surprise

## Implemented software slice

- Public `stage0-distance-{prepare,runtime,run,forecast,freeze,status,
  verify-measurement,report,verify,recover-writer}` workflow. Verified R0 sources
  and four frozen memberships are required; all parent identities remain bound.
- Exact prompt/answer-token deduplication across gold and sampled datasets. Each
  unique sequence gets one body forward; memberships retain repeated-draw weight.
- Gold masks use saved SFT labels and observed supervised EOS. Sampled target IDs
  include actual generated EOS; caps get no synthetic EOS. Prediction token t
  uses context t−1. No response generation or adapter/optimizer operation occurs.
- Local-only FP32 untuned model, TF32/autocast off. Output-head projection is
  chunk/workspace bounded; vocabulary surprise/entropy reductions are FP64.
  Per-token surprise, entropy, integer rank, token/context mapping and unit hashes
  are persisted without a full sequence-by-vocabulary production array.
- One bounded unique-answer measurement precedes production. Forecast uses worst
  observed seconds per squared total sequence length plus measured attempt
  overhead; it is explicitly a heuristic, not a completion guarantee. A distinct
  review freezes measured production and reserves from the existing cumulative
  20-unit budget. Profile reservation is settled before production. New allocation
  handoff updates the selected reservation rather than reusing a settled ID.
- Resume skips sealed arrays; changed sources/session/code/memberships fail before
  model loading. Failed full integrity invalidates earlier units in that attempt.
  Exact owner-token recovery retains budget and sealed evidence.
- CPU token-weighted NLL/entropy/excess-surprise estimates, 10,000 saved
  whole-problem bootstrap draws, seed42, pointwise 95% intervals. Primary
  ci_low/high are NLL; excess-surprise intervals are independently named.
- Whole-answer negative-log(k/8) is an equal-problem summary only for correct
  cohorts, not compared with per-token NLL. Empty eligible cohorts are explicit.
- Summary Parquet/CSV, per-problem CSV, top50 surprising gold token occurrences,
  structured results and inspectable HTML. The own_all ±0.02 gate is unchanged;
  failure preserves results and marks failed-gate without changing tolerances.
  A finite stopped-sequence ratio is not advertised as an exact zero identity.
- CPU sections32–33 / GPU26–31 expose preparation, bounded measurement, separate
  production review, collection, actual-usage reconciliation and CPU report.

## Reporting conventions

Initially implemented explicit defaults while the optional clarification was
unanswered. The user subsequently approved all three conventions, with additions:
full logit-vector rank/NLL vocabulary including padded rows, dataset short-answer
and window counts, exact Pilot2 SFT/generated-token/end-turn treatment, and
answer-weighted sensitivity with whole-problem bootstrap. Those additions are
implemented and separately validated below; no longer a pending/default choice.

## Test evidence so far

- Session baseline: 13 Ticket11 tests passed in23.71s.
- Independent probability and deduplication math tests red before implementation,
  then 2 passed. Full initial workflow: 4 passed in79.35s.
- Synthetic gold EOS mismatch was rejected by the SFT mask/end-turn contract;
  corrected the fixture to use the same sampled EOS. No gate was weakened.
- Measured-production test red before freeze enforcement. Fixed a reservation
  selection bug exposed after settling the bounded profile.
- Notebook handoff test red before cells were added, then green.
- Scoring/runtime/budget/full-integrity workflow tests: 8 passed in107.02s.
- Independent uneven-length oracle: NLL2.5 (not equal-problem mean2), entropy.875;
  whole-problem NLL interval[1,3], separate whole-answer correctness mean.5.
- Final affected regression recorded below after completion.

## Actual execution

CPU synthetic fixtures only. No model download, CUDA execution, training, Colab
unit usage, Drive upload or GitHub push. Real R0 and scoring remain pending. Raw
per-token outputs permit later positional summaries without extra GPU forwards.

## Validation closeout in progress

- Affected regression before positional-window addition: **27 passed in161.92s**
  (distance, R0 and qualification/budget suites).
- Positional-window test red before implementation, then four independent math
  checks passed in0.16s, including short-answer nonoverlap and token-weighted
  whole-problem interval oracle. Final scoring suite recorded below.

## Final closeout, 27 October2026

- Final scoring suite after positional summaries: **10 passed in140.74s**.
  Together with the affected regression, all **28 unique affected tests** passed
  across the recorded runs. No full legacy-suite claim.
- Labelled passing and failed-own_all-gate synthetic snapshots, raw per-token
  arrays, summary Parquet/CSVs, bootstrap draws, reports and validation logs are
  retained at `artifacts/validation/stage0-ticket12/`. Both report variants
  independently reconstruct; the failed gate is preserved with threshold.02.
- Both notebooks parse, contain no saved outputs, and pass their public-workflow
  handoff test. Diff whitespace checks pass.
- No real inference, numerical acceptance change, Colab use, Drive upload or push.

- Verified scientific implementation/notebook release pin: `97b21a8bf5a3e0c5fe30454cc84abb138e997a01`.

## Approved-convention extension, 27 October2026

- Actual scoring vocabulary receipt records full output-head row count, tokenizer
  size and padded rows. Rank uses the same promoted full vector as NLL softmax.
  Every scoring unit binds the receipt; ranks/targets must be within its range.
- Dataset summary records answers under64 scored tokens, first32/middle/last32
  token counts including zero windows, and answers lacking observed end-of-turn.
  Gold tokens preserve the exact saved loss-bearing SFT sequence. Generated
  EOS is included; capped sequences receive no artificial EOS.
- Answer-weighted NLL/entropy/excess-surprise sensitivity and separate95% bounds
  reuse whole-problem bootstrap indices, retaining all samples/draw multiplicity.
  Primary weighting and ±0.02 gate remain unchanged.
- Baseline four math checks passed in0.11s. Uneven-length test red on missing
  sensitivity field, then green. Two-problem/three-answer oracle confirms
  sensitivity13/3 vs primary19/5, with problem-cluster sensitivity interval[2,9].
  Both focused tests passed in0.22s. Final suite recorded below.
- Previous validation snapshots/pins remain historical software evidence; the
  amended implementation requires a newly named scoring plan when its identity
  differs. No real scoring artifacts exist to redo.

- Approved-convention final scoring suite: **11 passed in138.28s**. Both amended
  synthetic reports independently verified, including padded vocabulary receipts,
  answer-weighted sensitivities and short-answer/window counts. Evidence at
  `artifacts/validation/stage0-ticket12/approved-conventions-*`. No GPU run.

- Approved-convention scientific source/notebook pin: `efe0352bb96262ea054ee18af913b1213438d096`.

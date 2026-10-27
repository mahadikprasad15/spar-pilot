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

An optional clarification was sent for rank ties and positional windows. After
allowing time for a reply, the agent explicitly stated the implementation defaults:
one plus strictly-greater logits (ties share rank), and disjoint first32/middle/
last32 windows with first32 precedence on short answers. These are recorded as
implementation conventions, not as user approval. Per-token evidence permits
changing either on CPU without repeating model inference. Window bootstrap
indices and denominators are separately saved and verified.

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

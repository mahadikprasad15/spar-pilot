# 07: Produce the descriptive gate report and explicit training selection

**What to build:** CPU commands restore eligible data and HF score evidence to produce gold/k-arm plain-NLL, length, format, leakage, acceptance and coverage views. The common-cohort paired bootstrap uses approved estimands, reports absent baseline/offset honestly, and creates a source-bound request for training k0/k5. No automatic merge or training launch occurs.

**Blocked by:** 06.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Restore verified gold/k0/k2/k5 eligible products and emit per-arm plus three-arm-common-cohort descriptive tables/plots with source identities.
- [x] Collect/reuse verified HF FP32 gate score evidence under the appropriate budgeted model-work route; keep subsequent statistical/report commands CPU-only.
- [x] Compute stop-inclusive answer NLL then equal-problem means; length median/p90, format, coverage/retry/pass/unfinished rates and acceptance by step/cut decile retain explicit denominators.
- [x] Use10000 whole-problem paired bootstrap resamples with seed42 and percentile95% intervals: difference of arm medians recomputed per resample for length, difference of means for NLL.
- [x] Report literal calculator-marker rate and response word4-gram overlap with cleaned gold, with predeclared word normalization and undefined status for fewer than4 words.
- [x] Omit offset/baseline columns and state relative sharpening not measured. No automatic merge, equivalence or causal mediation claim.
- [x] Create a source-bound explicit k0/k5 training-selection/approval product with measured budget and coverage flags. k2 stays unselected without separate approval.
- [x] Hand-computable public-workflow fixtures reconcile rows/plots/intervals and reject stale parents or approvals.

## Testing and evidence

Hand-computable public workflows check stop-inclusive equal-problem NLL, difference-of-medians resampling, undefined short-text leakage, whole-problem pairing, report reconciliation and stale approval rejection. Gate scoring model work is separately budgeted; CPU statistics never require CUDA.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Answer

Implemented source-bound gate preparation, independently collected HF FP32 score units and CPU-only report/verification commands. Gold uses the exact original loss-bearing target IDs; sampled arms use reviewed retained IDs. All targets include actual stops.512 problem units save aligned probabilities, exact inputs and worker/runtime evidence; completed units are reused and malformed engine output replays once before blocking. Real scoring requires the qualified pilot/runtime, real collected arms and an active approved budgeted job. No baseline is collected.

The report retains own covered and common-cohort tables, raw answer/token scores, plain equal-problem NLL, stop-inclusive lengths, strict format/marker rates, coverage/retry/pass/unfinished counts, acceptance by step/cut decile and leakage counts. Whole-problem paired bootstrap uses10000 resamples/seed42, differences of resampled medians for length and differences of means for NLL, percentile95% intervals. User approved unicode-words-v1 and occurrence-weighted4-grams during implementation; design/spec clarification persisted. Short responses remain undefined. Baseline/relative sharpening are explicitly not measured, and no offset, automatic merge, equivalence or causal mediation claim is introduced.

Standalone SVG cohort and paired-interval figures have numeric input manifests; HTML, CSV, JSON and JSONL remain inspectable. Report verification recomputes rows/statistics from sealed scores. Source-bound training requests bind k0/k5 intent, report/coverage/score evidence, exact budget/forecast and required omission/difference acknowledgments. Stale requests/approvals reject; hard caps and80% reviews remain enforced. Selection receipts do not authorize training: Ticket08 readiness and Ticket09 paid execution remain separate; k2 is deferred.

Validation:72 affected CPU checks passed in249.06s across source/budget/job/gate arithmetic and public workflows, following a13-check review/UI baseline. Includes hand arithmetic, stop slicing, paired estimands, score restoration, bad-score rejection, stale ledger decisions, report mutation rejection, plot inputs and default notebook CPU cells. Canonical validation receipt and relocated verified synthetic report:artifacts/validation/sampling-sft-ticket07/. Fixture-human decisions and fake probabilities are not actual experiment findings or user approvals. No paid job, model download/inference, training or push occurred. Runbook:docs/operations/sampling-sft-gate.md. Ticket08 is next: exact training export and T15/T16 readiness.

# Sampling SFT design map

## Notes
- Original supplied design preserved in sources/; working design in design.md.
- Read-only source/upstream audits completed; targeted baseline: 5 passed.
- Rounds 1–3 answered on 2026-10-28. No implementation or cloud work begun.

## Decisions so far
- Data-recipe claim; descriptive mechanism measures; common-coverage k contrasts.
- Generate k0/2/5; later approved training k0/k5 only; k2 training conditional and separately approved.
- No automatic merge or causal/equivalence claim.
- Coverage threshold 0 / 1–5 / >5; smaller-batch edge cases remain open.
- Length-based extension with doubled block safety bound; four terminal states and bounded retries.
- Frozen sampling configuration and exact pilot restart/resume qualification.
- USD25 ceiling; sampling10/training6/storage-and-slack9; 80% review checkpoints.

## Decisions added in round 2
- T16 covers batches3–7; hand loss, finite gradients, one scheduler step; no duplication.
- Three-arm common gate cohort; arm-specific training coverage; flag pairwise uncovered-set differences >5.
- Per-attempt terminal states, per-problem dispositions, separate interruption events.
- Exact block-boundary replay, prefix-cache resets and ordered requalification.
- Immutable automatic scores plus confirmed human eligibility overrides.
- Conservative reservations, ongoing storage lifetime, stage-attributed costs and automatic post-export stop; deletion requires approval.

## Decisions added in round 3
- Defer512×8 temperature1 baseline and every NLL-offset column; no historical reuse.
- T3: exact alignment, mean error<=0.001 nats/token and max token error<=0.01; fixed coverage of short/long/stop/cap/empty-prefix fixtures.
- J64 fixed; initial pilot unfinished>2%, correct<90%, or over-budget Stage2 reservation stops for review.
- Equal-problem NLL including stop; common-cohort paired bootstrap10000/seed42/percentile95%; differences of medians for length; approved leakage diagnostics.

## Remaining facts/preflight
- Actual training runtime identity, engine compatibility/numeric flags and stop semantics.
- Deterministic fixtures/metric serialization and bounded profiling.
- Atomic resume, export/download receipts and confirmed Pod shutdown.
- Full CPU regression still in progress at this update.

## Next
Confirm consolidated shared understanding. Then spec/tickets or implementation only on user request; no cloud launch is implied.

# Sampling SFT design map

## Notes
- Original supplied design preserved in sources/; working design in design.md.
- Read-only source/upstream audits completed; targeted baseline: 5 passed.
- Round 1 answered on 2026-10-28. No implementation or cloud work begun.

## Decisions so far
- Data-recipe claim; descriptive mechanism measures; common-coverage k contrasts.
- Generate k0/2/5; later approved training k0/k5 only; k2 training conditional and separately approved.
- No automatic merge or causal/equivalence claim.
- Coverage threshold 0 / 1–5 / >5; smaller-batch edge cases remain open.
- Length-based extension with doubled block safety bound; four terminal states and bounded retries.
- Frozen sampling configuration and exact pilot restart/resume qualification.
- USD25 ceiling; sampling10/training6/storage-and-slack9; 80% review checkpoints.

## Fog — next frontier
- Clustered missing slots and test-failure fallback; common cohort definition.
- Empty outputs, engine-error retry counter, per-attempt versus per-problem counts.
- Review provenance and eligibility overrides.
- Replay scheduling/cache state and deterministic scientific records.
- Budget reservations, ongoing storage billing, and 80% checkpoint behavior.
- Baseline provenance, proposed score tolerance and pilot alarm actions still to resolve.

## Next
Ask the dependent question frontier, record answers, then confirm shared understanding before spec/implementation.

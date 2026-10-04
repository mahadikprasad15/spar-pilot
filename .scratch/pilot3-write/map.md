# Pilot 3 implementation map

## Notes

Spec and technical design are approved inputs; six local tickets were published
on 2026-10-04 after the user's approval to begin. No GPU experiment has run.

## Decisions-so-far

- Use vertical, independently inspectable behaviors rather than tickets per software layer.
- Dependency order: 01 → 02 → 03 → 04 → 05 → 06.
- Source training artifacts are not available in the local checkout; CPU fixtures
  can exercise the workflow without inventing real source evidence.
- Ticket 01 resolved: public frozen preparation/audit, source checks, token masks,
  completion integrity, failure status and relocation tests. Ten new CPU tests;
  full suite 141 passed. See ticket 01 Answer for the verification limits.

## Fog

- FineWeb `sample-10BT` was explicitly approved; resolve and record its dataset
  revision once during preparation, then reuse frozen inputs.
- Real source checkpoint verification and GPU validation remain Colab work.

## Frontier

02 — Validate and measure one batch (unblocked; not yet claimed).

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
- Ticket 02 resolved: activation engine, numerical summaries and persisted
  diagnostic workflow; full suite 152 passed. Actual GPU/source validation still
  required. See ticket 02 Answer and the instrument usage guide.

- Ticket 03 resolved: fixed-workload batch profiling, saved numerical/memory
  evidence and explicit reviewed execution freezing. See ticket 03 and the
  instrument guide; full suite 158 passed. Actual CUDA profiling still requires Colab.

## Fog

- FineWeb `sample-10BT` was explicitly approved; resolve and record its dataset
  revision once during preparation, then reuse frozen inputs.
- Real source checkpoint verification and GPU validation remain Colab work.

## Frontier

04 — Resumable all-checkpoint measurement (unblocked; not yet claimed).

# Pilot 3 implementation map

## Notes

Spec and technical design are approved inputs; six local tickets were published
on 2026-10-04 after the user's approval to begin. No GPU experiment has run.

## Decisions-so-far

- Use vertical, independently inspectable behaviors rather than tickets per software layer.
- Dependency order: 01 → 02 → 03 → 04 → 05 → 06.
- Source training artifacts are not available in the local checkout; CPU fixtures
  can exercise the workflow without inventing real source evidence.

## Fog

- FineWeb stream configuration requires explicit user selection; preparation
  accepts an explicit configuration and records it without a hidden default.
- Real source checkpoint verification and GPU validation remain Colab work.

## Frontier

01 — Freeze and audit measurement inputs.

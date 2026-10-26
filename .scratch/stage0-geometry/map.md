# Stage 0 design review

## Notes

- Original proposal: `sources/stage0-design-supplied-2026-10-26.md`.
- Working design and decision tree: `design.md`.
- User requested grill-with-docs before implementation/specification.

## Decisions so far

- Preserve the supplied proposal separately from review findings.
- No new Stage 0 GPU execution has started.
- User amended E0.6a: NLL primary, descriptive excess surprise secondary,
  separate whole-answer negative-log correctness probability; retained +/-0.02
  own_all acceptance threshold.
- User amended GPU policy: available suitable GPU, FP32/TF32-off, within-session
  pairing, new write measurements alongside new KL. Agreement gate details and
  the disjoint-cohort reward-rate fallback need clarification.
- Always measure subspace rotation/intruders; add update-to-gap ratio. Intruder
  overlap uses the old top-m subspace; m and threshold remain open.
- Save raw gradients; push is their negative, with saved A0 factor orientation.
- Save all energy components and cross term; headline fraction excludes cross term.
- Conditional V6 is a merged-weight edit, with an explicit merged-weight loader.

## Fog

- Primary claim needs explanation before a decision.
- Agreement-gate definition, confidence interval targets and correctness-probability
  aggregation need clarification.
- Intruder subspace size/threshold, gap edge cases and V6 direction selection and
  representation need specification.
- Source availability and historical settings are being checked from saved files.
- Scope, justified acceptance gates, measured budget and notebook handoff details
  follow these decisions.

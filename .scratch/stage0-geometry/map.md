# Stage 0 design review

## Notes

- Original proposal: `sources/stage0-design-supplied-2026-10-26.md`.
- Working design and decision tree: `design.md`.
- User requested grill-with-docs before implementation/specification.

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
  pairing, new write measurements alongside new KL. Agreement gate details and
  the disjoint-cohort reward-rate fallback need clarification.
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

## Fog

- Primary claim is settled; scope/budget details still need specification.
- Agreement-gate definition, confidence interval targets and correctness-probability
  aggregation need clarification.
- Gap edge cases, deterministic V6c matching and exact merged-loader acceptance
  metric/cohort need specification.
- Source availability and historical settings are being checked from saved files.
- Scope, justified acceptance gates, measured budget and notebook handoff details
  follow these decisions.

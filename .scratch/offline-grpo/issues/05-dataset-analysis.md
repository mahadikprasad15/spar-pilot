# 05: Dataset analysis

Status: ready-for-agent
Blocked by: 01 Pool routes and sealed mapping

## What to build

Use the existing untuned-model dataset scoring contract to analyze all pool answers and compare eligible Gold and sampling-SFT products. Preserve token-level scores and aggregation metadata, and distinguish dataset distance from learner diagnostics.

## Acceptance criteria

- [ ] Include all eight draws and zero-correct questions; report all answers, correct versus incorrect and capped versus uncapped, with token-weighted and equal-question or equal-answer summaries.
- [ ] Report each dataset's full coverage and the common paired cohort separately; reuse previous scores only after token, mask, scoring-model and definition compatibility checks.
- [ ] CPU fixtures test grouping, weights, masks, zero-correct retention and incompatible-source rejection; no fixture is presented as a real model measurement.
- [ ] Describe excess surprise as descriptive, not as KL or a new acceptance gate.

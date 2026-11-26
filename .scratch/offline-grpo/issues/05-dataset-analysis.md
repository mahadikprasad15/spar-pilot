# 05: Dataset analysis

Status: resolved
Blocked by: 01 Pool routes and sealed mapping

## What to build

Use the existing untuned-model dataset scoring contract to analyze all pool answers and compare eligible Gold and sampling-SFT products. Preserve token-level scores and aggregation metadata, and distinguish dataset distance from learner diagnostics.

## Acceptance criteria

- [ ] Include all eight draws and zero-correct questions; report all answers, correct versus incorrect and capped versus uncapped, with token-weighted and equal-question or equal-answer summaries.
- [ ] Report each dataset's full coverage and the common paired cohort separately; reuse previous scores only after token, mask, scoring-model and definition compatibility checks.
- [ ] CPU fixtures test grouping, weights, masks, zero-correct retention and incompatible-source rejection; no fixture is presented as a real model measurement.
- [ ] Describe excess surprise as descriptive, not as KL or a new acceptance gate.

## Implementation evidence

Additive CPU-tested offline dataset scoring and reporting are implemented; this is software evidence, not a completed real model measurement. The public fixture exercises interruption and resume, duplicate-score reuse with all draw memberships, zero-correct groups, EOS/mask validation, source tampering, historical grouped-unit seal validation, and token-weighted/equal-answer/equal-question reporting. The existing historical product audit failed closed because a referenced training plan was missing from the inspected artifact root. GPU scoring and a complete-source historical import remain unverified. Ticket remains claimed for review.

## Root review

Public prepare/run/verify/report implement pool multiplicity, full and derived common cohorts, subgroup and token/equal-answer/equal-question aggregations, sealed historical-score reuse and recovery. Existing scoring backend and token-statistic validation are reused. Combined dataset and prior Stage 0 distance regression passed 15 tests. Final focused tests passed 4 after provenance/report changes. Real historical reuse needs the missing referenced training plan restored; actual GPU scoring remains execution work.

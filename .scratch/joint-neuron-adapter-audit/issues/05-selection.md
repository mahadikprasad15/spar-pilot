# 05: Freeze independent candidate lists and matched controls

**Status:** needs-triage — draft for adversarial review

**What to build:** Read sealed discovery statistics to independently compute and freeze A, A-nb, B and D lists, optional N, relative-D sensitivity, and matched-control draws. Produce exact indices, eligible populations, scores and reproducibility metadata.

**Blocked by:** 03a. Acceptance uses a sealed, labeled fixture, so implementation can proceed while 03b/04 are pending; the report ticket separately requires 04's production collection.

- [ ] Refuse a production input with a mismatched frozen discovery manifest, missing arms or units for the declared comparison, or held-out rows; label fixture outputs as prototypes.
- [ ] A/A-nb use top-decile math activity and standardized differences, B ranks mean absolute read terms, and primary D is the absolute value of the mean paired change; freeze signed D and relative-D alongside it.
- [ ] Freeze floors, budgets, tie-breaking, activity bins, populations and seeds in the plan; report undefined statistics, underfilled lists or infeasible controls instead of silently altering criteria.
- [ ] Implement proposed fine activity quantiles 0/50/75/90/95/99/100 with deterministic tied-boundary handling, plus the candidate 50th-percentile relative-activity floor; freeze actual values only after review and do not merge bins after observing results.
- [ ] Save at least five reproducible layer/activity-matched controls per list and independent discovery split-half selections; do not pad intersections or restrict B/D to A.
- [ ] CPU fixtures prove list independence, ranking semantics, determinism and no held-out influence; verify stratified sampling against an exactly enumerable null and use a stable, non-flaky simulation check acknowledging discrete coverage.

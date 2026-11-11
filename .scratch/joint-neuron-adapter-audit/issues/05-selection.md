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

## Comments

### 2026-11-11 — predictions optional (user decision)

- Freezing predictions before examining discovery results is **optional**, not a gate (user, 2026-11-11). Nothing in 05/06 requires a prediction record. If one is made, it must be saved before discovery results are inspected, and 06 compares against it.

### 2026-11-11 — first implementation on fixtures (Claude)

- `pilot_eval/neuron_audit_selection.py` reads sealed block units through `Discovery`, discovery rows only. It refuses held-out/calibration rows, missing arms or rows, and units collected from a different sequences manifest (unless `prototype=True`, which labels every output PROTOTYPE).
- Lists per layer: A / A-nb (ceil 10% base-activity gate, signed standardized difference with pooled sample SD, low-spread excluded and reported), B per arm, D per arm (|mean paired change|, signed and mean-z change kept), relative D (layer p50 floor; zero floor/median unresolved). Primary 64 and secondary 32 budgets; ties by index; underfilled lists keep their size.
- Rank-based bins, 5 matched controls per primary list (seed base 20000, same bin counts, excluding members, unavailable if a bin is short), overlaps for the primary family (A/B, A/D, B/D within arm, D/D both directions) with per-layer and equal-layer pooled bin-matched nulls, split halves (seed 43).
- `matched_null` is vectorized: one uniform order per draw and bin, without replacement, from one seeded stream per (comparison, layer, direction). The exactness tests still pass. Note: the plan's "seeds 10000–10999" is implemented as base 10000 plus a derived stream with draw indices 0–999, not one seed per draw. **[proposed]** for review.
- Plan additions (`selection` section, `low_spread_factor`, `a_gate_fraction`) are tagged `[proposed]`.
- Tests: 12 fixture tests (planted lists recovered independently, both D/D directions, pooled stats, determinism and no held-out influence, bin-matched controls, seal/corruption, refusals, ties, bins, unresolved, low spread). All 126 audit tests pass.
- 04 must put `sequences_sha256` into each unit's identity, so production selection can bind units to the frozen sequences.

### 2026-11-11 — Colab rehearsal r1 findings (Claude)

- 05 crashed on a rehearsal with no math rows (first frozen batch only). Now refuses clearly when a group has < 2 rows; rehearsals collect covering batches (`558e77e`). Details: `../rehearsal-log.md`.

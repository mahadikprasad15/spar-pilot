# 10: Report

**Status:** needs-triage — draft v2
**Type:** task
**Blocked by:** 07–09 for real data; fixture-implementable after 04

**What to build:** `pilot_eval/attribution_report.py`.

- [ ] Scope line: base-correct, untuned-solved questions; provenance per arm; confirmation's partial independence.
- [ ] Per arm × metric gate diagnostics and pass/fail; the stated consequences of a failure.
- [ ] Primary 1–3 with paired bootstrap intervals and Holm-adjusted decisions; inconclusive is an outcome.
- [ ] Concordance analysis for the 6 removal conditions; disagreements listed.
- [ ] Descriptive: attribution maps, ρ and overlap, concentration vs random-order null, cancellation, a vs a₀ from the init check.
- [ ] Compute appendix: busy %, MFU estimate, tokens/s, peak memory, padding, wall time per stage.
- [ ] Fixture tests: planted separable / non-separable / inconclusive cases; gate trips on a planted bad estimator; unavailable R and empty masks handled.

## Comments

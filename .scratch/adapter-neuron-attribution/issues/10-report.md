# 10: Report

**Status:** ready-for-agent for code on fixtures (needed before the overnight run); running it on real outputs happens inside the chain. No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 07–09 for real data; fixture-implementable after 04

**What to build:** `pilot_eval/attribution_report.py`.

- [ ] Scope line: base-correct, untuned-solved questions; provenance per arm; confirmation's partial independence.
- [ ] Per arm × metric gate diagnostics and pass/fail, labelled a limited check on selected removals; the stated consequences of a failure.
- [ ] Primary 1 by the joint-bootstrap ratio rule (reference required; one-sided bounds); primary 2 by paired correlation differences; primary 3 tests by paired R contrasts, classification by interval bounds (descriptive). Holm over the declared family; inconclusive is an outcome.
- [ ] a vs a₀ quoted only as raw, gauge-dependent saved-factor cosine and drift, if at all.
- [ ] Concordance analysis for the 6 removal conditions; disagreements listed.
- [ ] Descriptive: attribution maps, ρ and overlap, concentration vs random-order null, cancellation, a vs a₀ from the init check.
- [ ] Compute appendix: busy %, MFU estimate, tokens/s, peak memory, padding, wall time per stage.
- [ ] Fixture tests: a noisy reference effect that makes a point-estimate rule claim equivalence but the joint rule does not; R with point estimate above 0.70 and lower bound below it classified inconclusive; planted separable / non-separable / inconclusive cases; gate trips on a planted bad estimator; unavailable R and empty masks handled.

## Comments

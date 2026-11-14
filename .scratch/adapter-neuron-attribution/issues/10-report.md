# 10: Report

**Status:** resolved (2026-11-13) for code on fixtures (needed before the overnight run); running it on real outputs happens inside the chain. No GPU run, value or budget is approved by this status.
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

### 2026-11-13 — interpretations to confirm (implementing agent)

- **Primary 1 in both metric families.** The spec puts "primary 1's decision" in the primary family *per metric*, but primary 1 is one cross-metric decision. I compute it once and include its p-value (the largest component p) in both the trace and stop Holm families. Please confirm, or say which family it belongs to.
- **"Support needs both > 0" (primary 2).** The report gives each difference's estimate, interval, two-sided p and Holm result, and flags `…_points_positive` when both point estimates are positive. I have not combined these into a single "supported" verdict. If support should mean "both Holm-rejected and positive", that is a one-line rule to add.
- **Stage D concordance directions.** Agreement is counted as ΔL with the same sign as Δaccuracy, and ΔS with the opposite sign to Δlength (stop pull is a proxy for early termination).
- **Seeds for the concentration null** are a required input (see the ticket-04 question). The resolution floor is the approved 06b rule's value.

## Answer

2026-11-13 — `pilot_eval/attribution_report.py`, tests `tests/test_attribution_report.py` (18, CPU fixtures).

- **Bootstrap.** Seeded question-level indices are shared across arms and conditions within a cohort. Resample count and seed must be given (launch record). Intervals are 95% percentile; p is two-sided. NaN (empty stop-mask) rows are excluded per resample, never 0.
- **Gate** (`validity_gate`): sign agreement for the 5 top singles (≥ `singles_min`) and the top-10 group (prediction = sum of members, labelled first-order). It reports Spearman, sign agreement, median |error| and slope over top and matched singles with paired intervals, and is labelled a limited check on selected removals. A planted wrong-sign estimator trips it. `gate_consequences` sets the labels:
  - primary 2 contrasts touching an arm without a passing gate are `unvalidated-exploratory`;
  - primary 1 rests only on directly measured contrasts;
  - primary 3 is always "attribution, not causal sparsity".
- **Primary 1** (`primary1`): D_S, D_L, D_A and r from the same resamples. The reference is required (D_A interval excludes 0) or the result is inconclusive; one-sided 95% bounds on r decide supported or contradicted. Planted supported, contradicted, no-reference and no-stop-effect cases behave as specified. A noisy-reference fixture whose point ratio is below 0.25 comes out inconclusive. An unavailable matched group gives inconclusive, not an error.
- **Primary 2** (`primary2`): Spearman with average ranks on signed mean-attribution vectors; paired bootstrap of the four differences (a)–(d). Excluded or unresolved arms give unavailable contrasts.
- **Primary 3** (`primary3`): R per order from the same resamples. It is unavailable when the interval of M_full − M_removed includes 0 or is non-finite. The full-removal reference is the `down-full` condition, or the top single that is that adapter. Tests are R(attr) − R(size) and R(attr) − mean R(random). Classification uses interval bounds only and is descriptive: a point estimate above 0.70 with a lower bound below it is inconclusive. `dense_everywhere` requires the upper bound below 0.30 in every arm.
- **Holm** runs per metric over primary 1, primary 2's four differences and primary 3's two contrasts per arm.
- **Descriptive and exploratory.** `concordance` (sign agreement, McNemar via `stage0_findings_math.mcnemar`, listed disagreements). `concentration` (neurons to 50% and 90%, cancellation, 1,000-order random-accumulation null; unavailable below the resolution floor). a vs a₀ is quoted only as raw and gauge-dependent. A compute appendix.
- `build` assembles everything from sealed-output inputs, and `render` writes Markdown with the scope line, provenance statement and partial-independence note.

### 2026-11-14 — user decisions

- Primary 1's p-value is in both metrics' Holm families (as implemented).
- Primary 2 support: both differences > 0 **and** rejected by Holm (`primary2_support`, now in the report for GRPO sharing and k-ordering).
- Concordance directions as implemented: ΔL with the same sign as Δaccuracy; ΔS with the opposite sign to Δlength.

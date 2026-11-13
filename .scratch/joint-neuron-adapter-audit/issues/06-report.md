# 06: Produce paired stability and held-out report

**Status:** needs-triage — draft for adversarial review

**What to build:** Generate a reproducible report and structured outputs for the four specified list comparisons, including held-out statistics, controls and uncertainty.

**Blocked by:** 04, 05.

- [ ] For each declared primary comparison, report observed overlap and >=1,000 seeded activity-stratified matched-null draws with median and 95% randomization interval; apply the frozen per-layer and equal-layer pooling rules, eligible populations and underfilled-bin behavior.
- [ ] Report top-64 response-content/general-background primary family for all specified arms/arm pairs; clearly mark near-miss, relative-D, top-32 and other views exploratory. Report B∩D for both primary and relative rankings.
- [ ] On held-out data evaluate fixed A selectivity, B term sizes, and D mean paired changes/sign agreement against their discovery-frozen matched controls; include split-half stability and predeclared question-level bootstrap intervals without re-ranking.
- [ ] Report per-arm/layer FP32-vs-FP64 calibration floor, its sample and coverage, selected-neuron errors where available, and the fraction of each applicable list below 5× the floor; zero floor or missing calibration is unresolved, not an all-clear.
- [ ] Show equal-example and token-weighted summaries, sample counts, truncation, leakage uncertainty, group overlap, missing arms and unsupported measurements.
- [ ] Describe source confounds, numerical resolution, gauge ambiguity, multiple exploratory comparisons, one-seed limits and absence of causal/behavioral conclusions; cross-arm signed reads require explicit reliable orientation or physical vector writes. Figures and tables trace to sealed inputs.

## Comments

### 2026-11-11 — predictions optional (user decision)

- Freezing predictions before examining discovery results is **optional**, not a gate (user, 2026-11-11). Nothing in 05/06 requires a prediction record. If one is made, it must be saved before discovery results are inspected, and 06 compares against it.

### 2026-11-11 — first implementation on fixtures (Claude)

- `pilot_eval/neuron_audit_report.py` reads a sealed 05 selection (it never rewrites or re-ranks it; a test checks file hashes before and after) and reads held-out rows through `HeldOut`, the sealed-unit reader limited to the held-out split.
- Per layer:
  - A held-out math-minus-general selectivity (equal-example and token-weighted);
  - B held-out read sizes on math and background;
  - D mean paired change with sign agreement against the discovery sign (zero/undefined counted separately).

  Each comes with frozen matched controls (unavailable controls are reported as such) and a percentile bootstrap (2,000 resamples, seed 1234, groups resampled separately; D keeps the arm pairing).
- Also: B∩D for both the primary and relative D rankings; the fraction of D entries below 5× the FP64 floor (an unresolved floor is shown as unresolved); the pooled overlap family with randomization references (labelled not a CI); split-half stability; coverage with partial label, counts and 03b flags; limitations.
- Outputs `results.json`, `report.md` and `overlaps.png` (if matplotlib is installed), sealed under `runs/joint-neuron-adapter-audit/report/<run_id>/`. A production report refuses a prototype/rehearsal selection.
- Tests: 8 fixture tests. 152 audit tests pass.
- **Not done:** per-layer figures beyond the pooled overlap plot, and a published HTML version. The report has not been run on real data.

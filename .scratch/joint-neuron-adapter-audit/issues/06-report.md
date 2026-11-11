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

# 05: Reconstruct the scientific report on CPU

**What to build:** Generate reproducible scientific tables, uncertainty and plots from verified complete summaries without model inference.

**Blocked by:** 04 — Measure full cohorts with safe resume.

**Status:** ready-for-agent

- [ ] Reconstruct primary/secondary measurements for both weighting schemes and all three views from saved sufficient summaries.
- [ ] Use the spec's paired example bootstrap; recompute nonlinear ratios and report undefined-replicate coverage.
- [ ] Save structured results, block-depth curves, module heatmaps and validation/denominator/resolution diagnostics.
- [ ] Reject incomplete or invalid evidence; keep single-seed, context and noncausal interpretation limits visible.
- [ ] CPU workflow tests reproduce independently calculated values and preserve source provenance without GPU imports/downloads.

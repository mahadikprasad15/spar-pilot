# 05: Reconstruct the scientific report on CPU

**What to build:** Generate reproducible scientific tables, uncertainty and plots from verified complete summaries without model inference.

**Blocked by:** 04 — Measure full cohorts with safe resume.

**Status:** resolved

- [x] Reconstruct primary/secondary measurements for both weighting schemes and all three views from saved sufficient summaries.
- [x] Use the spec's paired example bootstrap; recompute nonlinear ratios and report undefined-replicate coverage.
- [x] Save structured results, block-depth curves, module heatmaps and validation/denominator/resolution diagnostics.
- [x] Reject incomplete or invalid evidence; keep single-seed, context and noncausal interpretation limits visible.
- [x] CPU workflow tests reproduce independently calculated values and preserve source provenance without GPU imports/downloads.

## Answer

Implemented `activation-report` and CPU-only saved-batch iteration. Reports
verify full measurement evidence before reconstructing all 6,720 block/module
cells, both weighting schemes, secondary measurements and denominator coverage.
Primary ratios use 2,000 paired example bootstrap replicates (seed 42), with
independent FineWeb draws and explicit undefined-replicate masks/counts.

Outputs include source/runtime/code provenance, mean vectors, exact draws,
scalar replicates, JSON/CSV/Markdown, sampled validation diagnostics and
PNG/SVG block curves, module heatmaps and baseline-denominator plots. Complete
reports are hash-verified before reuse; incomplete sources and corrupt marked
reports cannot be silently accepted. See `docs/pilot3-instrument.md`.

Verification: four new CPU tests cover independent nonlinear/cancellation
examples, undefined draws, the connected saved workflow, plotted undefined
cells, seed/pairing evidence, corrupt output rejection, partial source rejection
and fresh-process CLI reuse with model imports forbidden. Full offline CPU suite:
**175 passed, 13 existing warnings, 315.79 seconds**. Inspected fixture block and
module plots visually. Real source/GPU measurements remain unverified; fixture
results are not scientific findings.

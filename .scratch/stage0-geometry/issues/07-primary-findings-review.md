# 07: Primary findings, targeted review and non-inferiority report

**What to build:** CPU commands generate paired accuracy/length/KL tables and plots, export the targeted intended-answer audit, incorporate confirmed labels and render an inspectable primary report with raw tables and evidence links.

**Blocked by:** 05, 06.

**Status:** resolved

**Execution:** Software implemented and CPU tested; real Stage 0 qualification, collections and research findings remain pending.

**Track:** Primary.

- [x] Lower 95% bound strictly above -5pp is required for an accuracy preservation claim; n=150 inconclusive risk is preregistered. All specified controls and individual random draws remain visible. Labels apply only to identical text; pending review and raw/overlay outcomes stay distinguishable. Incomplete optional work or not-applicable V6 does not block this report. Per-item and document sums reconcile with reported intervals and figures.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

## Answer

Implemented the CPU saved workflow for 14 declared paired contrasts, independent
bootstrap/McNemar and preservation checks, raw result/plot/HTML exports, targeted
blinded review, exact-text confirmed-label reuse (including the older assisted
route), durable label revisions/freeze/reveal and immutable review snapshots.
GPU 12A registers the primary analysis before collection; CPU 17–22 completes the
analysis/review/report handoff. All optional branches remain non-blocking.

- 2026-10-27: Final affected regression: 73 passed in 193.02s. Starting Stage 0
  regression: 49 passed in 58.13s. No full-suite claim, real GPU execution or new
  human scientific labels. See `execution-ticket07.md` and local validation under
  `artifacts/validation/stage0-ticket07/`.

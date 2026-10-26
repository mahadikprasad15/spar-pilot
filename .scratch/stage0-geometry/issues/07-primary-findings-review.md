# 07: Primary findings, targeted review and non-inferiority report

**What to build:** CPU commands generate paired accuracy/length/KL tables and plots, export the targeted intended-answer audit, incorporate confirmed labels and render an inspectable primary report with raw tables and evidence links.

**Blocked by:** 05, 06.

**Status:** ready-for-agent

**Track:** Primary.

- [ ] Lower 95% bound strictly above -5pp is required for an accuracy preservation claim; n=150 inconclusive risk is preregistered. All specified controls and individual random draws remain visible. Labels apply only to identical text; pending review and raw/overlay outcomes stay distinguishable. Incomplete optional work or not-applicable V6 does not block this report. Per-item and document sums reconcile with reported intervals and figures.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

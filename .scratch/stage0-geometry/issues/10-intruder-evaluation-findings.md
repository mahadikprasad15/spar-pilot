# 10: Conditional intruder evaluation and findings extension

**What to build:** Reuse the qualification, budget, behavioral/KL collection and reporting workflow for conditional merged variants, with fresh references when required, and add a separately identified findings extension.

**Blocked by:** 07, 09.

**Status:** resolved

**Track:** Conditional intruder extension.

Implementation coverage is complete. The checkboxes record software coverage
validated with CPU fixtures; real mixed-loader qualification and experiment
execution remain pending.

- [x] Mixed loaders pass approved qualification; all control masks, merged matrices and representations are bound in the frozen plan. Credit accounting is cumulative, not reset for this extension. Reports distinguish full/matched/sensitivity cohorts, undecidable/skipped modules, and either arm's eligible edits. No extension runs if the catalogue is not applicable.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [x] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

- 2026-10-27: Implemented and CPU-fixture validated; see
  `../execution-ticket10.md`. Numerical approvals and real execution remain
  separate, pending operator stages; no GPU qualification is implied.

# 15: Optional checkpoint trajectory and drift-size report

**What to build:** Select original intermediate checkpoints and verified reusable products, collect only missing same-runtime KL/write measurements when approved, and publish CPU geometry/drift-size trajectories plus saved-output endpoint length analyses.

**Blocked by:** 02, 06.

**Status:** resolved

**Track:** Optional supporting branch.

Checkboxes record implemented software and CPU-fixture validation. Real
GPU execution, measured budget approvals and scientific findings remain pending.

- [x] Historical/new compatibility is evidence-backed or separated; checkpoint/sequence identities and denominators are explicit; raw coefficient sensitivity retains prior policy boundaries. Observational size-matched curves do not become causal direction claims. Optional failures/missing products remain isolated from the primary report and total budget remains cumulative.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [x] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

- 2026-10-28: Implemented and tested; approved RSS size, token-mean FineWeb KL,
  effective-write mean, measured-only log–log curves and unclassified near2
  slopes. See `../execution-ticket15.md`. Equal-size rescaling is follow-up only.

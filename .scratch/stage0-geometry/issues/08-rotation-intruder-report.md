# 08: Gap-aware rotation and chance-aware intruder report

**What to build:** CPU analysis computes updated-matrix singular geometry, principal-angle rotation, update/gap diagnostics and intruder tables with explicit eligibility for conditional edits in either arm.

**Blocked by:** 02.

**Status:** resolved

**Execution:** Software implemented and CPU tested; real source/base-cache execution and real-weight endpoint geometry remain pending.

**Track:** Conditional intruder extension.

- [x] Always measure, with no small-update skip. Top-20 detection, m=max(k50,40), floored k30/k70 sensitivities, norm thresholds 0.3-0.7, sqrt(m/d_out) and undecidable status follow the approved rule. Candidate top-40 control overlaps, singular values and gaps are retained. Invariance under within-subspace basis changes and zero-update cases are independently verified; undefined and degenerate boundaries are explicit.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

## Answer

Implemented saved CPU commands for exact updated-weight SVD, whole-subspace
principal angles, update/sigma and update/gap ratios, chance-aware top-20 intruder
detection with top-40 candidate retention, and explicit degeneracy/undefined
statuses. CPU notebook sections 23–25 expose prepare, bounded run/resume and
verification/reporting. No small-update skip and no acceptance-number changes.

- 2026-10-27: Starting affected baseline 32 passed; final affected regression
  **44 passed in 106.77s**. Independent diagonal intruder/zero-update oracles,
  basis invariance, chance equality, source mutation, corrupted/re-sealed results,
  writer recovery and fresh-process CPU verification are covered. See
  `execution-ticket08.md` and `artifacts/validation/stage0-ticket08/`.

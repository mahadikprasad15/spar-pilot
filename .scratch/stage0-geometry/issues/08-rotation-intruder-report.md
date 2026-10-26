# 08: Gap-aware rotation and chance-aware intruder report

**What to build:** CPU analysis computes updated-matrix singular geometry, principal-angle rotation, update/gap diagnostics and intruder tables with explicit eligibility for conditional edits in either arm.

**Blocked by:** 02.

**Status:** ready-for-agent

**Track:** Conditional intruder extension.

- [ ] Always measure, with no small-update skip. Top-20 detection, m=max(k50,40), floored k30/k70 sensitivities, norm thresholds 0.3-0.7, sqrt(m/d_out) and undecidable status follow the approved rule. Candidate top-40 control overlaps, singular values and gaps are retained. Invariance under within-subspace basis changes and zero-update cases are independently verified; undefined and degenerate boundaries are explicit.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

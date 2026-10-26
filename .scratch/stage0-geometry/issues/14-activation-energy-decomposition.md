# 14: Activation-energy decomposition report

**What to build:** In one base-model pass per frozen sequence, collect all selected factors' principal, rest, cross and total energies; the CPU notebook reports bounded headline fractions and signed interference terms.

**Blocked by:** 02, 04.

**Status:** ready-for-agent

**Track:** Optional supporting branch.

- [ ] float64 sufficient sums and counts, correct token views, independent reconstruction, undefined zero denominators, base-input scope, and resumable verified units. All factors share captured base inputs rather than requiring separate arm forwards. Work runs only in approved spare budget.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [ ] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

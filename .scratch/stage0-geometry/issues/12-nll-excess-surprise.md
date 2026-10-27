# 12: Teacher-forced NLL, entropy and excess-surprise report

**What to build:** Score each unique answer once on GPU and summarize the approved dataset memberships on CPU, with per-token evidence and problem-level intervals, using a frozen measured-budget plan.

**Blocked by:** 11.

**Status:** resolved

**Track:** Optional supporting branch.

Checkboxes record implemented software and CPU-fixture validation. Real scoring,
measured-budget approvals and the scientific own_all result remain pending.

- [x] SFT masking and stopping treatment audited; NLL primary, descriptive excess surprise secondary, independently named intervals and separate whole-answer negative-log correctness probabilities. Preserve the +/-0.02 own_all gate and its sampling premises. No invalid KL identity, own-correct agreement test or duplicate subset forwards. Resume preserves membership, token mappings and counter/credit accounting.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [x] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

- 2026-10-27: Implemented saved unique-answer scoring, separate measured production
  freeze, per-token evidence and CPU reports. See `../execution-ticket12.md`.
  No model-backed execution or new scientific result is implied.

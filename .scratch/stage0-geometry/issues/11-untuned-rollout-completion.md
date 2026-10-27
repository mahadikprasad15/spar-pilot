# 11: Audited untuned rollout completion and derived cohorts

**What to build:** Recover historical raw groups/settings and use exact replay to choose reuse128 + missing384 or a fresh uniform512 cohort under a frozen measured-budget plan. Publish verified gold/own_all/own_correct/own_correct_one memberships and keep the original128 separately labelled.

**Blocked by:** 01.

**Status:** resolved

**Track:** Optional supporting branch.

Checkboxes record implemented/tested software coverage. Real historical source
audit, GPU replay, budget approval and scientific sampling remain pending.

- [x] Complete unique 512x8 identities when source compatibility supports combining groups; frozen historical ordering/seeds; strict/v2/v3 and cap outcomes retained; old/new rates descriptive only. Preserve exact 16-draw compatibility evidence. On a mismatch or unrecoverable historical seed/order, generate all512 freshly in one recorded session and retain old128 separately; never publish new384 alone as the fallback target. Zero-correct questions and deterministic one-correct selection are explicit. This branch cannot block principal interventions or consume reserved budget.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [x] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.

- 2026-10-27: Implemented; `../execution-ticket11.md` records validation. User
  amended the failed/unavailable replay route to fresh512, rather than new384
  alone. Original historical groups remain separately labelled.

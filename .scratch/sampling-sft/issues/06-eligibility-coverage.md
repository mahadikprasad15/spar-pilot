# 06: Review eligibility and publish the training coverage decision

**What to build:** The user reviews flagged responses through a saved round-trip interface, preserving automatic flexible-v3 decisions and recording confirmed intended-answer overrides separately. It publishes immutable coverage summaries, marked slots, common/union cohorts and pairwise symmetric-difference flags, and blocks>5 uncovered or unfinished required review. It never repairs text or duplicates missing problems.

**Blocked by:** 05.

**Status:** claimed

**Type:** task

## Acceptance criteria

- [ ] Provide a saved, fresh-session-restorable flagged-response review workflow with visible workload and partial-review status, using the same public commands as the notebook.
- [ ] Preserve automatic flexible-v3 verdicts, strict diagnostics and original generated text; human labels and agent suggestions have distinct provenance and revisions.
- [ ] Confirmed intended-answer overrides may change eligibility only for finished responses with extraction failures; they cannot edit tokens, supply gold or make capped/exhausted text eligible.
- [ ] Publish arm coverage, uncovered IDs/reasons, marked original training slots, three-arm common cohort and union/intersection summaries.
- [ ] Coverage0 passes only that gate;1–5 permits marked omissions subject to T16 and later approval;>5 blocks. No duplicate, gold, incorrect or unlimited-retry fallback.
- [ ] Flag pairwise uncovered-set symmetric difference>5 before training approval, and preserve each arm’s own covered training slots.
- [ ] Reject stale labels/altered sources and unfinished required review; demonstrate full label export/import and512-row reconciliation on offline fixtures.

## Testing and evidence

Saved fixtures cover review provenance and revisions, source-bound labels, cap/exhaustion ineligibility,0/1/5/6 omissions, clustered slots and512-row reconciliation. The user can inspect the workload and partial-review status before labelling.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

# 06: Review eligibility and publish the training coverage decision

**What to build:** The user reviews flagged responses through a saved round-trip interface, preserving automatic flexible-v3 decisions and recording confirmed intended-answer overrides separately. It publishes immutable coverage summaries, marked slots, common/union cohorts and pairwise symmetric-difference flags, and blocks>5 uncovered or unfinished required review. It never repairs text or duplicates missing problems.

**Blocked by:** 05.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Provide a saved, fresh-session-restorable flagged-response review workflow with visible workload and partial-review status, using the same public commands as the notebook.
- [x] Preserve automatic flexible-v3 verdicts, strict diagnostics and original generated text; human labels and agent suggestions have distinct provenance and revisions.
- [x] Confirmed intended-answer overrides may change eligibility only for finished responses with extraction failures; they cannot edit tokens, supply gold or make capped/exhausted text eligible.
- [x] Publish arm coverage, uncovered IDs/reasons, marked original training slots, three-arm common cohort and union/intersection summaries.
- [x] Coverage0 passes only that gate;1–5 permits marked omissions subject to T16 and later approval;>5 blocks. No duplicate, gold, incorrect or unlimited-retry fallback.
- [x] Flag pairwise uncovered-set symmetric difference>5 before training approval, and preserve each arm’s own covered training slots.
- [x] Reject stale labels/altered sources and unfinished required review; demonstrate full label export/import and512-row reconciliation on offline fixtures.

## Testing and evidence

Saved fixtures cover review provenance and revisions, source-bound labels, cap/exhaustion ineligibility,0/1/5/6 omissions, clustered slots and512-row reconciliation. The user can inspect the workload and partial-review status before labelling.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Answer

Implemented CPU-only public prepare/export/import/status/freeze/verify commands, a local review page with download and copyable backups, and source-bound human/agent revision histories. Required review selects all finished flexible-v3 extraction failures across retained arms/retries. The packet shows exact questions/responses without gold/scorer correctness. Confirmed human transcriptions are separately recorded with exact response excerpts; they cannot override valid wrong answers, edit tokens/text or rescue capped/exhausted responses. Agent suggestions cannot satisfy required review. Stale drafts, unknown labels, altered sources and unfinished review reject freezing.

Frozen coverage is reconstructed from original attempts and confirmed labels, with512 dispositions per arm, the original64 eight-slot batches, explicit missing IDs/reasons, first eligible attempt selection, common/union cohorts and pairwise uncovered-set symmetric-difference flags. Zero omissions passes coverage only;1–5 records T16/later-approval conditions;>5 blocks coverage. Per-arm batches.json/dispositions.json and the combined coverage.json are sealed. Training remains unauthorized.

Validation:111 combined Sampling SFT CPU tests passed in311.75s, including13 new review/UI checks. Covers separate agent provenance, partial/stale revisions, exact evidence, unchanged automatic verdicts, original-slot preservation,0/1/5/6 boundaries, pairwise differences, changed-source rejection, fresh-process verification without optional ML imports, page JavaScript backup/import behavior and default optional-notebook CPU cells. Canonical receipt and relocated synthetic reviewer bundle:artifacts/validation/sampling-sft-ticket06/. Fixture-human labels are test data, not actual user decisions.

CLI-first runbook:docs/operations/sampling-sft-review.md. No real model download/inference, paid/cloud execution, actual human audit or push occurred. Real collection/review are separate execution work. Unrelated Stage0 edits were preserved, including selective staging of this ticket's CLI additions. Ticket07 is next: descriptive gate report and selection.

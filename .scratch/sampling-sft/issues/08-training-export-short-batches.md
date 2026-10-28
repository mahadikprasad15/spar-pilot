# 08: Export the gold-matched training dataset and verify short batches

**What to build:** The user obtains a pretokenized training bundle using gold's exact per-step order and marked omissions, plus a readiness receipt proving checkpoint0 identity and T16 for batch sizes3–7. This slice independently verifies the training handoff without training a full arm or waiting for statistical-report presentation.

**Blocked by:** 01, 06.

**Status:** ready-for-agent

**Type:** task

## Acceptance criteria

- [ ] Export a stored-ID training bundle in gold’s exact per-step order with each arm’s own covered slots; every omission retains problem ID/reason.
- [ ] Verify saved tokenizer/prompt/masks and exactly one supervised end-turn; do not silently decode/re-encode or truncate targets.
- [ ] T15 compares checkpoint0 with gold bit for bit, including A0 and zero B; seed equality alone cannot pass.
- [ ] T16 exercises batches3/4/5/6/7 with existing total-supervised-token loss denominator: step completes, independent hand loss agrees, gradients are finite and scheduler advances once.
- [ ] Preserve64 nonempty steps and verified optimizer/configuration contracts; a failing test blocks for repair and never triggers duplication.
- [ ] Publish restorable training-readiness evidence independent of gate presentation. No full training or optimizer path changes are implied by this handoff.
- [ ] Use tiny offline CPU models/math through the public workflow, not mocked loss arithmetic; missing/broken local Torch is a reported missing check, not a pass.

## Testing and evidence

Tiny real CPU-model loss/gradient/scheduler checks, mask/end-turn and stored-ID checks, exact A0 comparison, all short-batch sizes and required metadata. Failure blocks readiness; no duplicate fallback. Full training still requires ticket07's approval product.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

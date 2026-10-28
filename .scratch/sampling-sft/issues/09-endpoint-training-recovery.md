# 09: Train approved endpoint arms with verified checkpoints and recovery

**What to build:** After explicit gate and budget approval, the user trains k0/k5 on verified L4 for64 steps, one seed42, with gold-matching configuration/logs and required adapters0/8/16/32/64. The workflow exports verified checkpoints and handles interrupted training honestly: verified complete recovery state permits resume; otherwise a budgeted new attempt restarts from checkpoint0. k2 never launches implicitly.

**Blocked by:** 02, 07, 08.

**Status:** ready-for-agent

**Type:** task

## Acceptance criteria

- [ ] Consume verified readiness, explicit k0/k5 post-gate approval and fresh budget reservations before training; require L4 FP32 and verified gold-matching numeric/runtime settings.
- [ ] Train only k0/k5, seed42,64 optimizer steps with gold-matching order/config/log fields and marked missing slots. Never rerun gold or implicitly train k2.
- [ ] Save required adapters at0/8/16/32/64, checkpoint0 identity evidence, step logs and source/config/order/runtime manifests.
- [ ] Optimizer state and full per-step gradients are optional captures. Resume only with verified complete adapter/optimizer/scheduler/progress/RNG recovery state.
- [ ] Without full recovery state, reject weights-only continuation; preserve the interrupted attempt and restart from checkpoint0 only as a new recorded, properly reserved/approved attempt.
- [ ] Reject changed sources, mismatched hardware, stale approvals or exhausted budget before expensive work; completed verified arms are immutable and reused.
- [ ] Verify export receipts and invoke the approved lifecycle. Tiny CPU public-workflow training/recovery tests are separate from real L4 qualification/execution.

## Testing and evidence

Public-command tiny CPU training covers step/checkpoint counts, skipped-slot exposures, immutable completed arms, optimizer-state recovery versus adapter-only refusal, hardware/approval/ledger mismatch and export. Real L4 execution evidence remains distinct from software tests.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

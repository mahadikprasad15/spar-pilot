# 09: Train approved endpoint arms with verified checkpoints and recovery

**What to build:** After explicit gate and budget approval, the user trains k0/k5 on verified L4 for64 steps, one seed42, with gold-matching configuration/logs and required adapters0/8/16/32/64. The workflow exports verified checkpoints and handles interrupted training honestly: verified complete recovery state permits resume; otherwise a budgeted new attempt restarts from checkpoint0. k2 never launches implicitly.

**Blocked by:** 02, 07, 08.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Consume verified readiness, explicit k0/k5 post-gate approval and fresh budget reservations before training; require L4 FP32 and verified gold-matching numeric/runtime settings.
- [x] Train only k0/k5, seed42,64 optimizer steps with gold-matching order/config/log fields and marked missing slots. Never rerun gold or implicitly train k2.
- [x] Save required adapters at0/8/16/32/64, checkpoint0 identity evidence, step logs and source/config/order/runtime manifests.
- [x] Optimizer state and full per-step gradients are optional captures. Resume only with verified complete adapter/optimizer/scheduler/progress/RNG recovery state.
- [x] Without full recovery state, reject weights-only continuation; preserve the interrupted attempt and restart from checkpoint0 only as a new recorded, properly reserved/approved attempt.
- [x] Reject changed sources, mismatched hardware, stale approvals or exhausted budget before expensive work; completed verified arms are immutable and reused.
- [x] Verify export receipts and invoke the approved lifecycle. Tiny CPU public-workflow training/recovery tests are separate from real L4 qualification/execution.

## Testing and evidence

Public-command tiny CPU training covers step/checkpoint counts, skipped-slot exposures, immutable completed arms, optimizer-state recovery versus adapter-only refusal, hardware/approval/ledger mismatch and export. Real L4 execution evidence remains distinct from software tests.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Implementation

Implemented the endpoint `sampling-sft-train-*` workflow and separate real-model
backend. Plans consume exact Ticket08 exports and immutable Ticket07 selection
receipts. Only selected k0/k5 can prepare; real work additionally requires full
verified T15/T16, a real gate, a fresh reserved/approved job, the reviewed per-arm
forecast and L4 FP32/runtime pins. Source and job mismatches reject before model
work. The source-bound sampler/pilot module remains unchanged, preserving its
previous qualification identities.

Training explicitly preserves each original step boundary. It reuses the gold
initializer, collator, summed-token completion loss, clipping and AdamW settings
rather than flattening short batches across adjacent steps. All64 steps remain
nonempty; omissions are recorded. The user approved arm-specific derived maximum
lengths on2026-10-29: retain every ID, record gold and arm maxima separately, and
stop at model context overflow without truncation.

The real initialized model's196 A0/zeroB pairs are compared with the exported gold
payload, and checkpoint0 retains the original adapter bytes. Required adapters
are saved at0/8/16/32/64. Optional recovery is captured at those same checkpoints,
with optimizer/scheduler/progress and Python/NumPy/Torch/CUDA RNG. It does not
write optimizer snapshots or hash the full base at every step. Recovery can replay
up to seven steps from the last verified checkpoint. Adapter-only continuation
is rejected; explicit restart-from-zero is a new recorded attempt with its own
paid approval/reservation. Interrupted attempts and their raw logs remain intact.

Completed outputs are immutable and restorable on CPU, with exact step exposures,
denominators, base integrity, runtime and declared export inventory reconciled.
Readiness at launch is retained separately; CPU fixtures explicitly disclaim
T16/real-GPU qualification. The existing lifecycle exports network and downloaded
copies before requesting its own Pod stop. No cloud provisioning or deletion
capability was added. Optional notebook sections17–18 prepare/inspect plans and
requests only. Primary operation remains CLI/RunPod scripts.

**Execution boundary:** no real L4 run, paid job, model download or trained sampler
arm is authorized or claimed by implementation. Full Ticket08 T16 remains pending
in the local environment, and all real post-gate/budget checks remain required.

## Validation

Final Ticket09 workflow checks:9 passed in493.23s. Affected jobs/budget/gate math/
instrument regression:48 passed in41.14s. These are CPU-only software checks,
including a real tiny196-factor model and real autograd/AdamW. The independent
FP64 first-step loss agrees within the approved T16 loss gate. A step9 interruption
resumes from checkpoint8; full successful logs and final adapter bytes match an
uninterrupted run exactly. Weights-only recovery rejects, an explicit checkpoint0
restart completes, runtime mismatch rejects, and changed completed logs reject.
Declared export files cover every completed-product dependency; incidental model
cards cannot make a downloaded bundle unverifiable. No tolerance was changed.

The broad initial pytest run was interrupted at228 passed,22 skipped,4 warnings
in449.58s; it is not a full-suite pass. Development diagnostics are retained
separately from the final logs, including a test-fixture lifetime failure fixed
by retaining the synthetic fixture in the canonical artifact tree.

Raw evidence and the durable synthetic workflow are under
`artifacts/validation/sampling-sft-ticket09/`. Fixture decisions are not user
approvals and fixture losses are not experimental findings. Full real L4
qualification/execution and Ticket08 full pinned-trainer T16 remain pending.

Fresh-process restoration of the durable synthetic completed arm passed without
importing Torch/Transformers/PEFT/TRL. The prior Ticket08 notebook guard was made
independent of cell positions and passed after the extension;58 unique relevant
checks passed across the focused, affected and notebook-extension groups.

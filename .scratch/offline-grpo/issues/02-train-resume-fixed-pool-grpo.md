# 02: Train and resume fixed-pool GRPO

Status: claimed
Blocked by: 01 Pool routes and sealed mapping

## What to build

Train from sealed fixed-pool windows using the existing unweighted GRPO update and the exact verified checkpoint-0 adapter. Preserve the online optimizer, batching, denominator, monitor actions and checkpoint schedule. Persist consumed responses, steps and recovery state, and resume only from verified boundaries. Do not implement the deferred TIS arm.

## Acceptance criteria

- [ ] Start verifies checkpoint-0 and configuration identity; base weights remain unchanged, trained parameters and optimizer state remain finite, and flags do not pause training.
- [ ] CPU fixtures compare the online and pool-backed paths on identical mixed and dead groups with unequal lengths: rewards, advantages, denominator, pre-clip gradients and one optimizer update agree under a previously declared criterion.
- [ ] An interrupted fixture resumes without duplicating draws, windows or updates; incomplete boundaries replay from the last verified checkpoint, with complete configuration and response records retained. Checkpoints 0, 8, 16, 32 and 64 retain adapter, optimizer, scheduler, RNG and data-cursor state.
- [ ] Fixture completion is distinct from authorization for a real training run.

## Comments

Implementation under review; this issue remains blocked by ticket 01 and is not resolved. Focused CPU tests: five passed; the pinned-TRL parity test was skipped because that stack is unavailable locally. A separate tiny-model Torch fixture passed with a mocked loss and an independent analytic gradient check; it does not establish pinned-TRL parity. Real checkpoint/source qualification, compatible historical implementation pins, real GPU smoke and scientific execution remain pending. No GPU run or model download was performed.

## Review status

Public preparation, fixed-pool training, exact source-bound checkpoint-0 loading, full-state checkpoint recovery and completion verification are implemented. Root affected regression passed 35 tests with 2 skips. Pinned-TRL parity remains pending because the available default Torch import is broken and the working alternate Torch environment lacks pinned TRL. A separate tiny-model analytic gradient/AdamW fixture passed with mocked loss. The ticket remains open until the pinned-TRL acceptance check is run. Real GPU smoke and training require the separately approved execution plan.

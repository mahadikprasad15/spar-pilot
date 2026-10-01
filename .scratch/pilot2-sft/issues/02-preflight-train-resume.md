# 02: Preflight, train and resume the rank-1 adapter

Type: task
Status: ready-for-agent
Blocked by: 01 - Prepare frozen training inputs and exploratory plan

**What to build:** Public GPU preflight and training operations that consume the
frozen plan, train the intended intervention and recover from interruptions
using complete saved training state.

- [ ] Supply exact compatible GPU dependency pins and record the actual runtime,
  device, repository revision and deterministic settings.
- [ ] Load the pinned unquantized FP32 model on a T4, attach FP32 rank-1/alpha-1
  adapters with dropout zero, enumerate all 196 intended projections and exclude
  the output head, embeddings and other base parameters.
- [ ] Persist zero-initial-write evidence and base parameter hashes; require
  frozen base parameters and identical base contents after training.
- [ ] Preflight the intended gradient-checkpointed, cache-disabled training path
  for memory, masks and finite loss/gradients without advancing the scientific
  run or changing its initialization/random state.
- [ ] Use TRL SFTTrainer with seed 42/full determinism, microbatch 1,
  accumulation 8, LR 1e-4, constant schedule, no warmup, AdamW, weight decay zero
  and gradient clipping 1.0; record all resolved optimizer settings.
- [ ] Validate completion-token loss normalization and data exposures; complete
  64 successful updates rather than relying on attempted-step counters.
- [ ] Save complete verified adapter/optimizer/scheduler/random/data-position
  states at steps 0, 8, 16, 32 and 64, with explicit checkpoint completeness.
- [ ] Resume from the latest complete compatible checkpoint; preserve failures,
  handle replay after an interruption and avoid misleading duplicate logs.
- [ ] Persist per-step loss, LR and gradient norm as structured logs and CSV,
  plus config, inputs, status and structured errors under the canonical tree.
- [ ] Stop on OOM, nonfinite values or incompatible state; never silently alter
  precision, batch, sequence content or optimization settings.
- [ ] CPU tests cover the public workflow and meaningful initialization,
  freeze, accumulation, failure and resume invariants without downloads.

## Comments

Approved as ticket 2. Real T4 execution remains unverified until GPU execution.


# Sampling SFT: endpoint training and recovery

Ticket 09 adds the CLI workflow for selected k0/k5 arms. Preparation, status and
verification are CPU operations. Training runs only as the child of an explicitly
approved, freshly reserved RunPod job on the audited L4 FP32 configuration.
Historical gold SFT is never rerun. There is no implicit k2 path.

## Prepare one attempt

Complete the [data gate](sampling-sft-gate.md), save its explicit training-selection
receipt and finish full [T15/T16 readiness](sampling-sft-training-data.md) before a
paid launch. A kernel-only diagnostic is insufficient. Preparing a plan is allowed
while readiness is pending; it does not authorize work.

```sh
python -m pilot_eval sampling-sft-train-prepare --data plans/sampling-training-data-v1/sampling-sft.training-data.json --selection runs/sampling-sft/sampling-sft-v1/gate/sampling-gate-v1/training-selections/SELECTION_HASH.json --k 0 --name sampling-k0-attempt1 --output-root artifacts
python -m pilot_eval sampling-sft-train-status --config plans/sampling-k0-attempt1/sampling-sft.training.json --output-root artifacts
```

Repeat preparation separately for selected k5. The plan binds the reviewed
selection, exact training export, original base identity, code and source hashes.
The original 64 step boundaries are retained. A missing slot reduces that step's
problem count; it never pulls the next problem forward. Each step processes one
problem at a time and divides summed CE by that step's actual supervised-token
count. All retained IDs, including the final end-turn, are used unchanged.

The loop reuses gold's initializer, collator and `completion_loss`, with FP32,
TF32 off, rank1/alpha1/dropout0 on all 196 modules, AdamW lr1e-4, betas0.9/0.999,
epsilon1e-8, zero weight decay, foreach off, clipping1 and constant learning rate.
It explicitly iterates original steps instead of relying on a flat dataloader to
infer short-step boundaries. Gold's saved batch order and seed42 are preserved.
This new runner still needs separate real L4 execution evidence; CPU tests are
not that evidence.

Gold's `max_length` is data-derived. On 2026-10-29 the user approved preserving
longer sampled traces and recording each arm's actual maximum separately.
The runner checks the pinned model's context limit before loading weights;
context overflow blocks instead of truncating or omitting tokens.

## Prepare and approve the lifecycle job

Create an explicit template JSON containing `job_id`, `kind: "gpu"`,
`mode: "runpod"`, `stage: "training"`, `network_volume_id`,
`rate_usd_per_hour`, `estimated_hours` and `setup_hours`. Do not put credentials in
this file. These are estimates, not live provider prices.

```sh
python -m pilot_eval sampling-sft-train-job-request --config plans/sampling-k0-attempt1/sampling-sft.training.json --template training-job-template.json --output-root artifacts
```

The printed JSON has exact child argv and every expected output, including
checkpoint tensors and recovery files. Pass it to `sampling-sft-job-prepare`
with the original sampling source plan. Then follow the existing
[RunPod lifecycle](sampling-sft-runpod.md) to approve and execute that job.
`scripts/run_sampling_sft_job.py` is the primary script; optional notebook section
17 only prepares/inspects plans, and section18 prepares the request.

Before a real launch the guard verifies full readiness, real gate evidence,
selected arm, output inventory and the reviewed per-arm forecast. The shared
lifecycle enforces the actual fresh reservation, USD25 total, training USD6 cap,
80% review, provider rate, source preflight and volume identity. Setup, debugging
and reruns remain training costs. A gate-time ledger hash is not required to stay
unchanged after legitimate reservations; the fresh job approval binds the current
ledger instead. An estimate larger than the selected arm's reviewed forecast
requires a new selection. Completed arms are verified/reused without model load.

## Captures and recovery

Outputs are under `artifacts/runs/sampling-sft/<source-plan>/training/<attempt>/`:

- `meta/run_manifest.json`, `meta/resolved_training.json`, `meta/runtime.json`:
  source/config/order identity, exact optimizer/numerical settings and observed
  GPU, driver, CUDA and package versions.
- `inputs/order.json`: all original slots and explicit omissions.
- `inputs/readiness.json`: the launch's exact readiness receipt; CPU fixtures
  explicitly record that full T16 was not claimed.
- `initialization-loaded/` and `results/initialization.json`: loaded model A0/B
  comparison against exported gold. Checkpoint0 retains gold's exact adapter bytes.
- `checkpoints/checkpoint-{0,8,16,32,64}`: required adapter captures.
- `logs/steps.jsonl` and `.csv`: loss, pre-clip gradient norm, learning rate,
  exposures, omitted slots, actual denominator and step counts.
- `logs/events.jsonl`, status, structured results and content-hash seals.

By default full recovery is saved at the five declared checkpoints: adapter, optimizer,
scheduler, Python/NumPy/Torch/CUDA RNG and progress with successful log records.
An interruption can repeat up to seven steps from the last verified checkpoint.
This uses more storage than adapter-only output, without hashing the full base
model or writing optimizer state at every step. `--no-recovery` disables optional
optimizer capture; only required adapters remain, and those cannot resume an
optimizer. No full per-step gradient dump is collected. No old state is deleted.

A continuation always gets a new attempt name and a new approved/reserved job.
The previous attempt and its partial work remain intact.

```sh
# Only when a verified complete recovery checkpoint exists:
python -m pilot_eval sampling-sft-train-prepare --data DATA_PLAN --selection SELECTION --k 0 --name sampling-k0-attempt2 --resume-from plans/sampling-k0-attempt1/sampling-sft.training.json --output-root artifacts
# Otherwise restart from original checkpoint0 under a new paid approval:
python -m pilot_eval sampling-sft-train-prepare --data DATA_PLAN --selection SELECTION --k 0 --name sampling-k0-restart --restart-of plans/sampling-k0-attempt1/sampling-sft.training.json --output-root artifacts
```

Resume verifies and restores state before any new optimizer update, carries the
committed log prefix forward, and rejects changed numerical runtime or source
identity. Adapter-only continuation is refused. If a killed process left writer
ownership, inspect it first and use `sampling-sft-train-recover-writer --config
PLAN --owner-token TOKEN --review-notes NOTES --output-root artifacts`. Active local
PIDs or a wrong token are refused. Never remove the marker to bypass ownership.

## Verify, export and stop

```sh
python -m pilot_eval sampling-sft-train-verify --config plans/sampling-k0-attempt1/sampling-sft.training.json --output-root artifacts
```

Verification reconciles all 64 steps, actual exposures/denominators, required
adapters, runtime, frozen base and file hashes. The approved lifecycle exports
both the network-volume and external downloaded copies, then requests its own
Pod stop and records the event. Shutdown confirmation and actual-cost settlement
remain explicit controller operations. A missing export acknowledgment is an
unresolved lifecycle state, not proof that the Pod stopped. No volume deletion or
unexported-Pod termination is performed.

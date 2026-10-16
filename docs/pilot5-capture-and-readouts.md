# Pilot 5: production capture and saved readouts

## Current state and next action

Cells 19–20 have completed on the user's Tesla T4: eight profile units, 32 sequence forwards, 20 model calls, successful verification. That is a completed **cost diagnostic**, with `qualified: false` and `production_ready: false`. Its saved FP32 arrays can support further numerical review; the diagnostic does not approve tolerances, FP16 retention or production scope.

Tickets 07–08 now implement the capture and readout machinery. **CPU 1–2 → 21** prepares the next proposal. Actual production remains gated on Ticket 04/06 qualification receipts, lens disposition, scope, budget and predictions. No successful real qualification receipt was fabricated in this release.

```text
Completed profile + saved source/corpus evidence
                  ↓
21 CPU: production proposal and storage inventory
                  ↓ reviewed qualification, scope, budget, predictions
22 CPU: immutable capture execution
                  ↓
23 GPU: one model → compatible batches → staged evidence → durable units
                  ↓
24 CPU: verify every required shard and aggregate
                  ↓ reviewed references and prereveal request family
25 CPU: norm/head basis and readout draft
                  ↓
26 CPU: bind approved readout requests and qualified lenses
                  ↓
27 CPU: saved vectors → lens projections → ranked tokens
                  ↓
28 CPU: verify and display saved token lists
                  ↓
Ticket 09: overlap, nulls, paired arm contrasts and uncertainty
```

## Experiment and conditions

There is no new fine-tuning or answer generation. Production uses the frozen **200 raw FineWeb documents, 128 content tokens each**. Each model call captures the embedding output and all 28 decoder-block outputs together.

Production review chooses endpoints or trajectories explicitly:

| Scope | Conditions | Sequence forwards | Model calls at batch 4 |
| --- | --- | ---: | ---: |
| Endpoints | disabled-adapter base, SFT64, GRPO64, existing random64 | 800 | 200 |
| Trajectory | base, SFT8/16/32/64, GRPO8/16/32/64, existing random64 | 2,000 | 500 |

Step-zero checks remain the separately declared bounded diagnostic. The random adapter is the existing norm-matched intervention; it is not a null distribution. Gaussian-vector controls are computed from saved evidence during readout and require no model capture.

The primary identity remains raw form, block-14 output, step64, positions 0–4 pooled within each document and then across documents. Capture does not depend on task-reference vocabulary review; learned-token reveal does.

## What is saved

For batch size B and hidden width D=1,536:

| Evidence | Shape | Stored precision | Purpose |
| --- | --- | --- | --- |
| Position-wise delta sums, every site | 128 × D per condition/site/batch | FP64 | Exact cohort position means and declared windows |
| Per-document primary delta and base vectors, every site | B × D for each bank | FP64 | Primary mean, paired document bootstrap, base/adapted controls |
| Primary vector norms and position-wise norm sums | B; or 128 | FP64 | Norm/coherence diagnostics with explicit undefined cases |
| Per-document token deltas, blocks 14 and 17–21 | B × 128 × D | Explicitly reviewed FP16 or FP32 | Position and subset analyses; converted storage is disclosed |
| Base block-14 sum and second moment, positions 1–127 | D and D × D | FP64 | Optional activation-shaped null; total count 25,400 |

Subtraction is FP64 after reading FP32 activations. Norms and pooled evidence are computed before retained-token conversion. Compression is lossless after that explicit conversion. It does not make FP16 values exact.

One accepted unit contains a complete batch across all frozen conditions, with one shared base reference. This lets recovery avoid persisting full base activations at all sites. An interrupted unsealed batch repeats; sealed batches remain accepted.

## Storage and measured costs

The production proposal estimates **all saved batch arrays**, including resume overhead, base moments and final aggregate arrays. At batch 4 with FP16 token retention, conservative array sizes before compression are approximately **11.86 GiB for endpoints** and **28.31 GiB for trajectories**. Metadata adds some bytes. Actual compressed sizes are measured and checked against the approved durable budget.

Those figures exceed the final six-block tensor bank alone because resumable batch sums and moments are also retained. The proposal exposes this cost before launch. It does not silently reduce the cohort, sites or retention.

The user's T4 profile measured 38.5 s model loading, 8.54 s total forward/capture/transfer work and 404.32 s base hashing across eight units. These are observations for that diagnostic, not a production ETA. Production currently hashes before and after every accepted batch. With 50 batches at batch 4, hashing can dominate runtime. Review must include that cost; no source-spec timing promise is treated as measured performance.

The readout basis is separate: an FP32 output head is about 0.9 GiB. CPU preparation can download the pinned model weight file(s) if uncached, then extracts only norm/head/tokenizer evidence. It constructs no decoder and imports no Torch; FP32, FP16 and BF16 tensor bytes are supported explicitly. The same basis can be prepared from a completed profile **before** production qualification and reused by a compatible production execution, avoiding a circular qualification dependency.

## Freeze and qualification boundary

`traces-production-propose` saves available source conditions and a review template. It does not approve anything. `traces-production-freeze` consumes a separate named review, bound to the proposal hash, with:

- Explicit condition inventory, endpoint/trajectory scope, profiled batch size and retained dtype.
- Qualification receipt binding the exact profile, model revision, selected batch and dtype. Required named checks are zero identity, base integrity, numeric lens, batching, retention, ranking and bootstrap. Real receipts also require a reviewer, reviewed limits and measured-evidence hashes.
- Qualified Jacobian provenance/shape/source/target evidence, or an explicitly approved logit-only partial scope. The inspected community artifact currently has insufficient provenance; its filename and finite matrices cannot satisfy this gate.
- Actual SFT/GRPO predictions with confidence, and approved positive time/storage budgets.

Qualification receipts are outputs of the numerical-review work owned by Tickets 04/06. A profile result with `qualified: false` cannot substitute for one. The current pending real receipts must be completed from measured evidence and user-approved limits. The code does not generate a real receipt by setting a checkbox.

Synthetic CPU fixtures use the same saved workflow but are tagged `synthetic-cpu`. Such a contract cannot launch a real CUDA backend. Runtime compatibility is compared separately from code identity: model/dependency/device/numerical settings must match the qualified runtime, while execution code is bound by its own file hashes. Different hardware requires a reviewed new execution/bridge, not rewriting old evidence or retraining earlier pilots.

## Recovery and durable release

Paths are relative to `/content/drive/MyDrive/SPAR/spar-pilot/artifacts` in Colab:

```text
plans/<name>/traces.production.proposal.json
plans/<name>/traces.production.json
runs/pilot-5/<name>/production/
    meta/{run_manifest,runtime,status}.json
    checkpoints/progress.json
    units/000000/{arrays.npz,evidence.json,complete.json}
    results/{aggregate.npz,results.json}
    logs/{run.log,errors.jsonl}
```

The notebook stages under the pinned checkout's local `artifacts/staging`. Each batch is hashed, serialized losslessly, read back, copied to a temporary durable filename, flushed and rehashed before its seal publishes. A failed sync retains the verified local stage. A retry reuses copied payloads whose hashes match and avoids repeating the corresponding forwards. Losing local staging before publication means the unfinished batch repeats. Corrupted sealed data is rejected and preserved for inspection.

Free-space checks precede model access for a new batch. Disk/write/sync failures preserve accepted units and suppress completion. Existing writer locks prevent competing runs. The approved time limit is checked at batch boundaries **per invocation**; it is not a provider billing limit or a hard interrupt during a model call. Durable storage usage includes prior accepted batches and the final aggregate.

The command closes the owned model before final CPU aggregation/verification. It announces GPU release only after required evidence is verified and then the child exits successfully. Storage verification covers readback and filesystem flush at the configured durable mount; it does not query a separate Drive remote-sync API. Cell 24 can reverify from a fresh CPU session with 1–2; no decoder is loaded.

## Saved readouts and the reveal barrier

`traces-readouts-prepare` requires reviewed references, completed verified capture, a prereveal request family and qualified lenses/explicit partial scope. Requests declare condition, role, named site and content positions. Unsupported base/adapted windows and unsupported Jacobian sites remain unavailable; exact zero vectors remain undefined.

Primary vectors use saved FP64 per-document pooling sums. Other cohort position windows use the FP64 position sums. This keeps the primary pooling order intact. Controls distinguish base mean, adapted mean, random-adapter delta and seeded norm-matched Gaussian directions.

The logit lens applies the pinned final RMS norm gain and output head. The Jacobian path first transports with `J @ v` (equivalently row-vector `v @ J.T`), then applies the same qualified final norm/head convention. FP64 projection uses bounded vocabulary chunks. Positive/negative raw token rankings and the approved normalized unique lists are saved with scores, vector norm/coherence and provenance. Tie-breaking uses ascending token ID.

Each reviewed request becomes a sealed readout unit with `vector.npy` and `result.json`. Completed units resume independently. `--max-new-units N` pauses after N new units without changing the request inventory. A completed run produces `results/readouts.jsonl` and a side-by-side `results/readouts.md`, plus structured completion metadata. No decoder inference or activation recapture occurs in this stage.

This barrier governs supported operations; directly opening internal artifact files is outside it. Unknown/prior exposure remains labelled exploratory or unverified. Token lists are measurements, not evidence of causality or a statistical SFT–GRPO difference. Ticket 09 owns those tests and comparisons.

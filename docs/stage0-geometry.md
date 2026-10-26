# Stage 0 geometry experiments

Stage 0 uses existing rank-one SFT/GRPO adapters. Its main result will be paired
intervention comparisons. Ticket 01 currently implements source readiness,
fresh-process restoration and a cumulative budget ledger. It does not perform
SVD, generation, gradients, KL, edits or GPU qualification.

## Notebooks

- `notebooks/stage-0-cpu-colab.ipynb`: setup, select existing sources, prepare,
  audit/resume, and inspect status. Use a CPU runtime.
- `notebooks/stage-0-gpu-colab.ipynb`: the shared setup/status entrypoint only at
  this stage. Do not allocate a GPU for its current cells.

Both select the same artifact root and plan name. The dedicated Colab checkout
uses a recorded release commit and refuses to switch a dirty checkout. The
commit must be published before its Colab bootstrap can fetch it. A local notebook
uses the current repository without checking out another commit.

## Source selection

The CPU notebook's selection cell records explicit prepared-source paths, relative
to the artifact root. Required labels are `sft` and `grpo`; `random` is optional.
Use the existing Pilot 3 SFT and Pilot 4 GRPO/random activation-prepared products.
The default notebook paths match those source plans. Inspect them rather than
searching for whichever similarly named run was modified most recently.

Optional `products` entries map a label to an explicit completion-receipt path.
Their receipt must contain nonempty file hashes relative to its own directory.
These checks establish payload integrity, not KL schema or runtime compatibility.
Leave the selection empty if such a product has not been identified. A missing
optional product or zero checkpoint is reported separately from endpoint readiness.

The audit verifies prepared-source receipts, their recorded training metadata,
and used checkpoint adapter/config/seal payloads. It reads the actual FP32 factors
for all 196 adapted linear modules. Initialization equality compares actual A
tensors, not seeds. Training optimizer *state* is not replayed or recursively
rehash-scanned. The base hash remains declared source evidence until a later
weight-backed verification; this slice does not download or instantiate the model.

## Public commands

From the repository checkout:

```sh
python -m pilot_eval stage0-prepare --selection artifacts/notebook/stage0-geometry-v1/source-selection.json --name stage0-geometry-v1 --output-root artifacts
python -m pilot_eval stage0-audit --config artifacts/plans/stage0-geometry-v1/stage0.plan.json --output-root artifacts
python -m pilot_eval stage0-status --config artifacts/plans/stage0-geometry-v1/stage0.plan.json --output-root artifacts
```

All notebook cells call this boundary. A fresh kernel needs only the storage and
checkout cell before `status`. An audit can be deliberately bounded using
`--max-new-units 1`; rerun the same command without the limit to finish. Completed
source units are verified and reused.

## Outputs and interpretation

For a selected name, the immutable plan lives under `artifacts/plans/<name>/`.
The source audit uses `artifacts/runs/stage0/<name>/`:

| Product | Purpose |
|---|---|
| `meta/run_manifest.json` | Plan identity and selected source bindings |
| `units/<source>/result.json`, `complete.json` | Source evidence and sealed unit |
| `results/source-audit.json` | Full audit, initialization comparison and limitations |
| `meta/status.json`, `checkpoints/progress.json` | Explicit lifecycle and completed counts |
| `logs/run.log`, `logs/errors.jsonl` | Progress and failure evidence |
| `complete.json` | Verified audit completion; not GPU or experiment completion |

`primary_sources_ready` means the selected endpoint adapter payloads passed this
audit; it does not mean the experiment is qualified to run. Full training facts
are shown only if the selected source includes a hash-bound training config.
Both `gpu_qualified` and `measurement_complete` remain false in this slice.

Missing sources are a snapshot, not invented completion. If previously missing
inputs are later supplied, select a new audit name to preserve the earlier
evidence. Changed previously bound inputs are rejected. Implementation changes
also require a new named plan; the old evidence remains inspectable on its pin.

## Cumulative budget

The ledger is shared across every Stage 0 plan under
`artifacts/runs/stage0/budget/ledger.jsonl`; `budget` is a reserved run name.
The ceiling is **20 compute units total**, not per notebook or session. No
initial zero is assumed. Before the first observation, spent and remaining units
are unknown and no GPU work is authorized. Ticket 04 will add measured job-cost
reservations and production authorization; Ticket 01 only records observations.

An observation is a JSON object with these fields:

```json
{
  "event_id": "session-01-reading-01",
  "session_id": "colab-session-01",
  "total_units": 0,
  "observed_at": "2026-10-26T12:00:00+00:00",
  "note": "Replace with verified cumulative Stage 0 consumption. Zero is valid only if confirmed."
}
```

`total_units` is cumulative experiment consumption, not account balance. Record
meter readings and derive the cumulative Stage 0 total using actual session
observations; do not silently attribute unrelated account consumption to this run.
Replace the example value with verified usage. Persist the observation under the
artifact root and submit it with:

```sh
python -m pilot_eval stage0-budget-record --config artifacts/plans/stage0-geometry-v1/stage0.plan.json --observation artifacts/notebook/stage0-geometry-v1/budget-observation.json --output-root artifacts
```

Events are hash chained and idempotent by ID. Repeating the same event is safe;
rewriting an ID or decreasing cumulative consumption is rejected. An observation
at/above 20 is retained honestly as `ceiling-reached`; it never authorizes more
work. Budget recording itself does not spend units or reserve a GPU workload.
The saved ledger head detects removal of already published trailing observations.
An append interrupted before head publication is `unpublished-observation`;
resubmit the identical observation to finish publication without adding a duplicate.

## Writer recovery

Two writers cannot own one product. Ordinary interruptions at the saved unit
boundary release ownership. An abruptly terminated process can leave `writer.json`.
Inspect its token/host/PID, confirm the previous writer has stopped, then use the
explicit recovery command with `--owner-token` and `--review-notes`. Select
`--product budget` only for ledger ownership; default is the audit product.
Recovery is logged and cannot release a live same-host process. It does not
change sources, unit receipts or completion evidence. Never delete a lock file
to bypass a conflicting active notebook.

## Evidence boundary

CPU fixture tests establish software behavior. A verified real Drive audit, target
GPU qualification and completed scientific measurements require their own saved
evidence. The source audit has not been executed on the actual checkpoint tensors
merely because these commands and notebooks exist.

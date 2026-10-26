# Stage 0 geometry experiments

Stage 0 uses existing rank-one SFT/GRPO adapters. Its main result will be paired
intervention comparisons. Tickets 01-03 implement source readiness, fresh-process
restoration, budget observations, a resumable CPU spectral cache, explicit
trained-overlap reporting and principal-edit construction. Generation, gradients,
KL and GPU qualification are later stages.

## Notebooks

- `notebooks/stage-0-cpu-colab.ipynb`: setup, select existing sources, prepare,
  audit/resume, status, spectral cache and optional projection reveal. Use CPU.
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

## Ticket 02: pinned weights and CPU SVD

The CPU notebook's sections 5-8 consume a completed source audit and a local
snapshot containing `config.json` and single/sharded safetensor weights. No model
is instantiated. The reader reconstructs the historical PEFT named-parameter hash
from FP32 tensor values, including adapted-module `base_layer` names and tied
embedding deduplication. It must equal the audited frozen base hash before the
cache is prepared. A different or incomplete snapshot stops preparation.

Place the weights under `artifacts/inputs/models/<audited-model-revision>/`.
Reuse existing files when possible. If a download is needed, use the audited
model revision explicitly, for example with the current Hugging Face CLI:

```sh
hf download Qwen/Qwen2.5-1.5B-Instruct --revision AUDITED_REVISION --include '*.safetensors' '*.safetensors.index.json' config.json --local-dir artifacts/inputs/models/AUDITED_REVISION
```

Replace both occurrences of `AUDITED_REVISION` with the source audit's exact
revision. This is optional staging, not an instruction to download another model
when a compatible snapshot already exists. Unit tests never download weights.

The four public spectral commands are `stage0-spectral-prepare`,
`stage0-spectral-run`, `stage0-spectral-verify` and `stage0-spectral-report`.
Preparation takes `--audit`, `--weights-dir`, `--name`, and `--output-root`.
Run/verify/report take the saved spectral `--config` and `--output-root`.
Report additionally requires `--review` and its own `--name`.

Run once with `--max-new-units 1` before the full cache. The first timing is a
measurement of that module, not a promise that the entire decomposition takes
minutes. Modules differ substantially in shape and CPU SVD can be expensive.
Run without the limit to finish; all previously sealed units are reused.
Large input shards are verified at command entry, not once per module. Full
SVD computations are not repeated to inspect a verified cache.

The cache contains all singular values and U/V columns through at least
`max(k70, 40)`. Ranks are the smallest reaching 30/50/70 percent squared singular
energy. Full decomposition is float64; retained bases are float32 with separate
post-conversion orthogonality checks. Retain both full and stored numerical
errors. The supplied reconstruction and orthogonality limits remain `1e-6`.

Outputs live under `runs/stage0/<spectral-name>/spectral/`: per-module `basis.npz`,
result/seal JSON, measured times/bytes, progress/errors, and a verified cache
summary. No trained-factor projection is computed by cache creation or verify.
An independently supplied synthetic backend is labelled `synthetic-cpu-fixture`;
it is never evidence that real Qwen weights were measured.

## Trained-overlap reveal

Section 9 is disabled by default. Existing bases can feed Ticket 03 without it.
Before reporting, save a review JSON with `reveal_trained_overlaps: true`,
nonempty `notes`, and `prediction_status` equal to `exploratory` or
`frozen-before-reveal`. The latter also requires `prediction_path`, relative to
the artifact root, pointing to retained frozen predictions. Do not label an
already viewed result a prospective prediction. This is an operator review
boundary, not a technical access-control or human-blinding guarantee.

The report records exposure before computing trained overlaps. It produces
all 30/50/70-percent rows for available step-zero/64 factors, chance k/d,
100-Gaussian seed-42 control bands, and the existing random adapter if available.
Zero-factor projection is undefined. Shared draws across same-dimensional bases
are disclosed; modules are not independent experimental replicates.

The unchanged Gaussian gate requires the 100-draw mean within 10 percent of
k/d per module/side/energy level. A failure saves diagnostics and stops reporting;
it does not silently retry random seeds or destroy the verified SVD cache.
The report includes projection JSONL/CSV, gaps, a summary, Markdown explanation,
SVG/PNG figures and a hash-bound completion receipt. Geometry differences alone
are descriptive and do not establish the proposed causal explanation.

## Ticket 03: principal edits and controls

CPU notebook sections 10-12 prepare, build/resume and verify the intervention
catalogue from the spectral cache. The optional overlap-reveal cell is not a
prerequisite. Commands are `stage0-edits-prepare`, `stage0-edits-run` and
`stage0-edits-verify`. Prepare takes `--spectral`, `--name`, `--output-root`;
run/verify take the saved edit `--config` and `--output-root`. Run supports
`--max-new-units` for deliberately bounded construction/resume checks.

| Variant | Source | Construction |
|---|---|---|
| V0a | SFT64 | Unedited reference; source config and tensors copied byte-for-byte |
| V0b | GRPO64 | Unedited reference; source config and tensors copied byte-for-byte |
| V1 | SFT64 | Remove b's base top-k50 output projection |
| V2 | SFT64 | Remove a's base top-k50 input projection |
| V3-0, V3-1, V3-2 | SFT64 | Remove k50 random orthonormal output directions; seeds 1000-1002 |
| V4 | SFT64 | Scale the original b per module to V1's effective-update norm |
| V4-input | SFT64 | Scale the original b per module to V2's effective-update norm |
| V5 | GRPO64 | Same output projection removal as V1 |

All targets remain plain rank-one alpha-one PEFT adapters. Magnitude controls
preserve the original update direction. Random draws use a local NumPy PCG64
generator with the declared draw seed reset per module, followed by reduced QR.
Full random bases are retained, so reproducibility does not depend on a future
QR implementation producing the same signs. No global RNG state is changed.

Per-module edit units retain all variants' factors, before/after checks and the
three random bases; each unit is sealed. A restart verifies/reuses those units,
then assembles complete standard safetensor PEFT adapters. Final serialized
factors are read back and compared with constructed factors. k=0 is byte-preserving
for factor arrays; references additionally preserve the complete original files.

Outputs live under `runs/stage0/<edit-name>/edits/`: `units`, `adapters/<variant>`,
`results/catalogue.json`, `results/catalogue.md`, progress/errors and completion
receipts. Each adapter includes its unchanged source adapter config, edited
weights, `edit.json` and a hash-bound seal. Source checkpoint files are never
modified. Changed cache, source, edited payload or frozen plan is rejected.

Checks include the declared 1e-6 removed-component residual, basis orthogonality,
and per-module effective-update norm agreement to 1e-6 relative. Failure records
the affected stage and stops; it cannot silently alter a tolerance or retry random
seeds. Zero update/target cases remain explicit. Construction does not emit the
separate trained-overlap report, although edit manifests necessarily expose
component-removal checks and update magnitudes needed to inspect the variants.

Synthetic partial-module fixtures are explicitly tagged and cannot be mistaken
for complete 196-module production edits. The real path requires all selected
modules from the verified base cache. Catalogue completion means CPU construction
only: `gpu_qualified` and `measurement_complete` remain false. V6/V6c merged-weight
interventions are a separate conditional extension and are not in this catalogue.

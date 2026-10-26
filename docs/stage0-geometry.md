# Stage 0 geometry experiments

Stage 0 uses existing rank-one SFT/GRPO adapters. Its main result will be paired
intervention comparisons. Tickets 01-06 implement source readiness, fresh-process
restoration, budget observations, a resumable CPU spectral cache, explicit
trained-overlap reporting, principal-edit construction, bounded qualification,
paired greedy collection and fixed-context KL/effective writes. Real-source and
real-GPU execution evidence remains pending. Gradients and the combined findings
report are later slices.

## Notebooks

- `notebooks/stage-0-cpu-colab.ipynb`: setup, select existing sources, prepare,
  audit/resume, status, spectral cache, optional projection reveal and edit
  construction/verification and saved greedy-result inspection. Use CPU.
- `notebooks/stage-0-gpu-colab.ipynb`: explicit pinned dependency setup/import
  checks, bounded qualification and reviews, followed by paired greedy collection.
  Fixed-context KL/write collection follows under its own bounded allocation.
  GPU inference occurs only in diagnostic and collection run cells.

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
python -m pilot_eval stage0-audit --config plans/stage0-geometry-v1/stage0.plan.json --output-root artifacts
python -m pilot_eval stage0-status --config plans/stage0-geometry-v1/stage0.plan.json --output-root artifacts
```

All notebook cells call this boundary. A fresh kernel needs only the storage and
checkout cell before `status`. An audit can be deliberately bounded using
`--max-new-units 1`; rerun the same command without the limit to finish. Completed
source units are verified and reused.

Saved-plan paths are absolute or relative to `--output-root`, as in the audit
and status examples above. The notebooks pass absolute paths.

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
are unknown and no GPU work is authorized. Ticket 04 adds measured job-cost
reservations and production authorization; Ticket 01 records observations.

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
python -m pilot_eval stage0-budget-record --config plans/stage0-geometry-v1/stage0.plan.json --observation artifacts/notebook/stage0-geometry-v1/budget-observation.json --output-root artifacts
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

## Ticket 04: bounded loader qualification and measured freeze

The GPU notebook now exposes the Ticket 04 workflow through the public CLI. First,
`stage0-qualification-prepare` binds the Ticket 03 catalogue to the exact ordered
Pilot 4 held-out evaluation config and selects its first 16 items. It records the
SFT V1 and GRPO V5 source adapters, tokenizer files, decoding settings, and a
partial merge over alternating adapted modules. This is a calibration sample for
the loader, not the production evaluation cohort.

`stage0-qualification-runtime` checks one visible CUDA device, the pinned
dependencies, stable runtime identity and Drive readback without loading a model.
The operator records current cumulative Colab usage and a bounded wall-time/rate
approval before `stage0-qualification-run` loads the already staged local model.
The run compares adapter versus unhalved partial-merged logits at every gold
answer-prediction position for both arms, records greedy completions, and checks a
deliberate double-application negative control. Each of 32 arm/prompt units is
sealed independently. A matching same-session resume reuses sealed units. If a
run is interrupted, its reservation remains active until the operator records a
new cumulative usage reading and calls `stage0-budget-settle`.

The numerical report proposes `3 × observed maximum absolute logit error`; it does
not approve that tolerance. `stage0-qualification-approve` requires a separate
review bound to the exact diagnostic receipt. The diagnostic hard-stops above
1e-3; a greedy token mismatch is accepted as a near-tie only when the reference
top-two margin at the first shared-prefix divergence is strictly smaller than
the measured local logit error. A separate production-budget review is required
to freeze the ten-variant × 150-item paired greedy plan. Its forecast covers only
greedy evaluation; KL and effective-write jobs require their own reservations.

Outputs live under `runs/stage0/<qualification-name>/qualification/`:
`meta/runtime.json`, `meta/tokenized-inputs.json`, per-unit results/seals,
`results/diagnostic.json`, review receipts, logs, reservation IDs and status.
The frozen plan is under `plans/<qualification-name>/stage0.execution.json`.
CPU fixtures exercise the public workflow, interruption/resume, source and seal
rejection, threshold gates, and budget reconciliation without loading weights.
They cannot qualify a real GPU loader. Implementation completion, synthetic
fixture approval, real-CUDA qualification, and scientific measurement are four
separate states. The 20-unit ceiling is shared across Stage 0 sessions and
notebooks.

## Ticket 05: paired greedy collection

GPU notebook sections 12–15 and CPU notebook sections 13–14 expose the saved
collection workflow. The commands are `stage0-evaluation-prepare`,
`stage0-evaluation-run`, `stage0-evaluation-status`, `stage0-evaluation-verify`
and `stage0-evaluation-recover-writer`. Preparation consumes the exact frozen
Ticket 04 `stage0.execution.json` and writes `stage0.evaluation.json` beside it.
It verifies the qualification, numerical approval, production freeze, original
cohort, and all ten adapter payloads before defining the work.

The current real qualification uses singleton inputs, so inference batch size is
1. Larger real batches are refused until separately qualified. Storage units
group eight sequential inference calls by default (`--batches-per-unit 8`),
giving 190 durable variant/group units for 1,500 generated responses. The final
group has six examples. This grouping changes save boundaries, not the model's
numerical batch: every method sees identical prompt IDs and the same ordered
`42 + inference-batch-index` seed schedule. A unit seals after all its outputs
and integrity checks are saved; an interrupted unsealed unit is repeated.
Fixtures may use larger inference batches but cannot qualify CUDA batching.

The runner loads one local-files-only FP32 model and attaches all ten standard
PEFT adapters. Before generation it compares all 392 actual loaded factor
tensors per adapter against the saved payload, including exact values and
mapping. It evaluates V0a/V0b and each edit within each ordered cohort
group. The originals are generated afresh in the same qualified runtime, not
joined to old GPU scores. It records prompt IDs, masks/positions, rendered
prompts, responses, content and raw generated tokens, EOS/cap status, scorer
versions, strict/flexible extraction, rewards, source hashes and unit/runtime
identities. EOS is removed from the content-token count but retained in raw
generated IDs; EOS in the final allowed generation slot is an EOS stop. A
response reaching the cap without EOS is invalid and has reward zero.

The first run consumes the active reservation created by Ticket 04's freeze.
Same-session retries deduct previous attempt seconds from that reservation;
they do not receive a new wall-time allowance automatically. After settling a
reservation, `run --approval <review.json>` requires a new explicit bounded
allocation. A process killed before timing publication has unknown duration:
reconcile actual cumulative usage and approve a new allocation before resuming.
Every attempt checks the global 20-unit ceiling before loading a model. A
runtime/session mismatch is rejected; there is no automatic cross-session
compatibility claim.

Base weights are SHA256 checked on load and at clean attempt completion; adapter
payloads are compared again at the end. Parameter identity/storage/version guards
cover base and adapter parameters around each generation unit. Known final
integrity failures invalidate that attempt's sealed batches. Writer ownership
is exclusive. After confirming that a stale writer has stopped, use the saved
token and notes with `stage0-evaluation-recover-writer`; recovery does not erase
responses, release reservations or clear failed integrity evidence.

Outputs are under the frozen `runs/stage0/<qualification-name>/paired-evaluation/`:

| Product | Purpose |
|---|---|
| `config.json`, `meta/run_manifest.json` | Exact source, variant, scheduling and output identity |
| `inputs/tokenized-prompts.json`, `meta/runtime.json` | Common inputs and qualified runtime |
| `meta/loaded-adapters.json` | Actual loaded-factor comparison for every variant |
| `units/<key>/responses.jsonl`, `metadata.json`, `complete.json` | Durable response group, scores and integrity evidence |
| `attempts/<id>/start.json`, `timing.json`, `final-integrity.json` | Reservation, measured attempt time and final check |
| `results/responses.jsonl` | All 1,500 verified raw automatic response records |
| `results/results.json`, `summary.csv`, `results.md` | Descriptive accuracy, invalid/cap and length summaries |
| `meta/status.json`, `checkpoints/progress.json`, `logs/*`, `complete.json` | Counts, lifecycle, errors and content-bound completion |

CPU verification reconstructs scores and summaries from sealed records and
reconciles them with the aggregate JSONL. A fresh process can verify and inspect
them without importing Torch, PEFT or Transformers. This ticket publishes raw
automatic results. Targeted human answer overlays, paired intervals and the
approved non-inferiority claim remain Ticket 07 work; KL/effective writes remain
Ticket 06. A synthetic fixture is explicitly labelled and never reported as a
real model finding.

## Fixed-context KL and effective writes (Ticket 06)

This is a separate measured workload after loader qualification. It uses the
**same frozen execution/runtime and all ten primary variants**, on the original
**150 GSM8K gold + 150 FineWeb sequences** from the audited SFT prepared product.
There is no new training or generation. Three instrument controls (untuned,
SFT-zero, GRPO-zero) use the first two documents of each corpus, four each.

GPU notebook sections **16–19** prepare, collect, reconcile usage and verify.
CPU notebook sections **15–16** inspect saved progress and verify. Each real
forward is singleton: saving eight documents per durable unit does not create an
inference batch of eight. Defaults produce **380 primary units + 3 control units**.
The main workload contains 3,000 variant/document combinations. Cached untuned
references, switching/restoration passes and bounded repeat/chunk diagnostics add
forwards; this is not a claim of only 3,000 total model calls. No completion time
is inferred from greedy decoding speed.

```sh
python -m pilot_eval stage0-drift-prepare --execution plans/<qualification>/stage0.execution.json --inputs plans/pilot3-inputs-v1/activation.prepared.json --name stage0-geometry-v1-fixed-context-v1 --output-root artifacts
python -m pilot_eval stage0-drift-run --config plans/stage0-geometry-v1-fixed-context-v1/stage0.drift.json --approval /absolute/path/drift-budget-review.json --max-new-units 4 --output-root artifacts
python -m pilot_eval stage0-drift-status --config plans/stage0-geometry-v1-fixed-context-v1/stage0.drift.json --output-root artifacts
python -m pilot_eval stage0-drift-verify --config plans/stage0-geometry-v1-fixed-context-v1/stage0.drift.json --output-root artifacts
```

The first bounded four-unit attempt covers the three controls and first SFT group.
Inspect `units/*/attempt.json` for measured group seconds and `attempts/*/timing.json`
for whole-attempt time, including model loading. An explicit conservative rate and
wall-time bound use the same budget-review fields as other Stage 0 GPU jobs.
The greedy reservation must be reconciled first; it cannot fund this workload.
Rerunning without `--approval` uses the remaining time in the matching active
allocation. A new allocation requires reconciliation and a new explicit review.
No allocation bypasses the shared **20-unit total ceiling**.

### Measurement definitions

- **KL:** full-vocabulary `KL(p_variant || p_untuned)` in nats, from FP32 logits
  with FP64 log-softmax/reduction. The target's view/mask belongs to token `t`;
  the compared logits come from `t-1`. Target zero is excluded. The output head
  projects at most 32 contexts at once under a declared 512 MiB workspace budget.
  No full `[batch, sequence, vocabulary]` array is produced or saved. Tiny signed
  KL roundoff is retained under the existing analytic FP64 bound, never clipped.
- **Writes:** actual direct LoRA branch on the adapted module's input `x`,
  compared to that module's actual base output on the same input, including
  any existing linear-layer bias. Write
  positions are unshifted and may include position zero. Save per-document,
  view and module: `sum ||delta W x||²`, `sum ||W x||²`, and
  `sum (||delta W x|| / ||W x||)`, with token counts from the frozen masks.
  A zero denominator is undefined and stops collection with evidence.
- **Magnitude:** save every module's actual FP64 `||b|| ||a||` and its sum.
  This primary catalogue has rank-one, alpha-one adapters. Conditional merged
  V6/V6c collection belongs to its later slice; a general dense-write oracle
  already tests the mathematical definition without inventing rank-one factors.
- **Aggregates:** corpus/view cells separate token-weighted and equal-document
  means. The RMS ratio `sqrt(sum update² / sum base²)` is labelled separately
  from the mean of per-token ratios. Individual random draws remain distinct.

The same local-only loader verifies all 392 factors of each loaded adapter,
including both zero controls. Base hashes, parameter guards and final payload
checks retain the qualification identity. Every variant switch checks the
untuned final hidden reference exactly; restoration is checked again afterward.
The first document of each variant's first group additionally checks strict
repeat agreement (including raw diagnostic coefficients) and alternate KL chunks.
Inherited Pilot 4 agreement and rank-one reconstruction limits are unchanged.
Zero controls require exact zero KL and direct writes across their full mappings.
No raw coefficient batch-sensitivity relaxation is generalized to these gates.

### Saved product and resume

Under `artifacts/runs/stage0/<drift-name>/fixed-context/`:

| File | Content |
|---|---|
| `config.json`, `meta/run_manifest.json` | Frozen source/runtime/tokenizer/settings and work inventory |
| `inputs/module-factor-norms.json` | Actual module-by-variant factor-norm products |
| `units/<key>/arrays.npz` | FP64 scalar KL vector; FP64 `[documents, 3 views, modules, 3 sums]` writes |
| `units/<key>/result.json` | Exact KL/write positions, tokens, examples, views, module order and numeric checks |
| `units/<key>/attempt.json`, `complete.json` | Timing, integrity, attempt identity and payload seal |
| `attempts/<id>/observations/` | Observations retained before validation, including rejected payloads |
| `attempts/<id>/start.json`, `timing.json`, `timing.complete.json` | Bounded allocation and elapsed-time evidence |
| `results/per-document.jsonl`, `results/results.json` | Reconstructed document/corpus/view/module results |
| `meta/status.json`, `checkpoints/progress.json`, `logs/` | Progress and failures |
| `complete.json` | Verified completion for this product; not a behavioral conclusion |

Incomplete writes are unsealed and recomputed. Valid sealed units are checked and
skipped. Altered inputs, adapters, reviews, runtime or implementation are rejected;
known full-integrity failure invalidates the affected attempt. Missing elapsed
measurement requires budget reconciliation. A stale writer is released only by
`stage0-drift-recover-writer --owner-token ... --review-notes ...`; a live local
writer is refused and all recovery evidence remains saved.

Historical KL/write values are **not inputs** to these new measurements. This
slice records the paired execution and runtime provenance required by Ticket 07;
it does not itself join historical values or behavioral results. KL on supplied
contexts is a distributional change, not by itself behavioral harm or evidence
of a causal mechanism. Real CUDA qualification/collection remains unexecuted in
the local software fixtures.

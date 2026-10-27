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

## Primary findings and intended-answer review (Ticket 07)

Ticket 07 is CPU-only. It verifies the completed greedy and drift products,
requires matching frozen execution/runtime/adapter identities and inference
batch size, and builds an inspectable report from their raw records. Optional
supporting work and the conditional V6 extension are not prerequisites.

### Notebook sequence

- **GPU 12A, before section 13:** prepare the drift plan and freeze the analysis
  family/hypotheses. This loads no model and reserves no GPU allocation. If a
  collection attempt or writer already exists, registration is conservatively
  marked `after-collection-start; exploratory`; it is never backdated.
- Finish the GPU collections and reconcile their usage as described above.
- **CPU 17:** restore the frozen analysis plan. If registration was omitted,
  preparation is still permitted, with its actual late timing recorded.
- **CPU 18:** compute/resume the 14 declared comparisons and publish raw tables,
  figures and HTML. A comparison is a sealed resumable unit.
- **CPU 19–21:** optionally prepare the blinded answer packet, save successive
  downloaded drafts, freeze complete labels and reveal the comparisons.
- **CPU 22:** numerically verify the raw report and publish an immutable snapshot
  with current review status. Pending review does not block the raw report.

Statistical settings are frozen at preparation: 10,000 paired percentile
resamples, seed 42, 95% intervals. The NumPy version used to generate those draws
is recorded at the first CPU analysis run rather than inherited from GPU plan
preparation. Resume/verification requires that recorded analysis version.

### Declared family and statistics

The family contains 14 edited/comparator-minus-reference contrasts:

| Family | Contrasts |
|---|---|
| Edit versus its source arm | V1, V2, V3-0, V3-1, V3-2, V4, V4-input minus V0a; V5 minus V0b |
| Unedited arms | V0b minus V0a |
| Matched magnitude controls | V1 minus V4; V2 minus V4-input |
| Individual random-subspace controls | V1 minus each of V3-0, V3-1, V3-2 |

Flexible-v3 accuracy is primary; strict is shown alongside it. Save paired
question IDs, correctness transitions, gross gains/losses, invalid and cap counts,
exact two-sided McNemar probabilities and intervals on paired accuracy changes.
“Accuracy preserved” requires the lower 95% bound to be **strictly above −0.05**.
A bound equal to −0.05 is inconclusive; an upper bound below −0.05 establishes a
loss beyond the margin. At 150 items, inconclusive intervals are plausible and
this risk is recorded before collection. All intervals are **pointwise**, not
simultaneous across the family; do not select a winner and describe the family
as a confirmed mechanism.

Length is the median of each question's edited-minus-reference token change,
with its paired interval, shorter-response share and cap counts. It is not the
difference of marginal medians. There is no new untuned greedy arm in this
catalogue, so a length increase is not automatically a claim of restoring
untuned performance or reasoning quality.

KL is averaged within each corpus/view document before bootstrap. Token-weighted
means are retained separately. Every resample carries the document's records
together; modules/tokens are not independent subjects. Per-module effective-write
changes are descriptive equal-document mean changes in per-token ratios, without
invented module-level hypothesis tests. Direct write sums and magnitude metadata
remain in Ticket 06's verified source product.

### Review protocol

Selection includes every changed extracted answer or correctness/validity bin
across the declared family, every flexible-v3-unextractable endpoint, and every
last-number diagnostic discrepancy. A wrong-to-wrong change in the extracted
number is still selected. Strict-only format failure does not force manual
review. Identical prompt/response strings are deduplicated across variants.

The reviewer page contains opaque response IDs, questions and response text;
it omits variant/checkpoint, gold and scorer fields. Its scope is intended-answer
identification, not reasoning validation. Browser storage is a convenience cache:
download drafts, save them through the public command and import the resulting
`reviewer/draft.json` before continuing. Saved drafts have an explicit revision
chain and a single reviewer identity, retained across reused and new labels. Unknown IDs, malformed numeric labels,
wrong packets and stale revisions are rejected.

An explicitly selected historical audit may supply confirmed labels, including
an explicitly named exposed revision, or a verified confirmed-only assisted
report. Its original audit receipts and case-level label evidence are verified;
the recorded historical implementation remains provenance, without requiring
that an older saved UI equal the current repaired UI code. Reuse requires **exact prompt and response
text**, without whitespace normalization or fuzzy matching, and records the old
response ID, reviewer, result hash and exposure status. Conflicting labels on
identical text stop reuse. Historical numeric answers are not copied onto new
or merely similar model outputs.

Draft labels do not affect scores. Complete new labels are frozen before reveal.
Only confirmed numeric intended answers contribute score corrections, and capped
responses retain zero reward. Absent/ambiguous labels are reported while raw
scores are retained; no numeric answer is invented. A mixed confirmed-case overlay
is labelled separately, retains automatic scores elsewhere, and does **not**
claim whole-cohort human accuracy, reasoning validity or non-inferiority. Its
intervals condition on those fixed targeted corrections, not on an exhaustive
human review. The original raw report is never overwritten.

### Public commands

All plan paths below are relative to `--output-root`; label/reuse paths may be
explicit absolute paths as shown in the notebooks.

```sh
python -m pilot_eval stage0-findings-prepare --evaluation plans/<qualification>/stage0.evaluation.json --drift plans/<drift-name>/stage0.drift.json --name stage0-geometry-v1-primary-findings-v1 --output-root artifacts
python -m pilot_eval stage0-findings-run --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --output-root artifacts
python -m pilot_eval stage0-findings-audit-prepare --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --output-root artifacts
python -m pilot_eval stage0-findings-audit-save --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --labels /absolute/path/downloaded-draft.json --output-root artifacts
python -m pilot_eval stage0-findings-audit-freeze --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --output-root artifacts
python -m pilot_eval stage0-findings-audit-reveal --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --output-root artifacts
python -m pilot_eval stage0-findings-verify --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --output-root artifacts
python -m pilot_eval stage0-findings-report --config plans/stage0-geometry-v1-primary-findings-v1/stage0.findings.json --output-root artifacts
```

`findings-run --max-new-units 1` performs one missing comparison. Repeating the
same command resumes verified units. `findings-status` inspects analysis progress;
`findings-audit-status` inspects durable label progress. `audit-prepare` optionally
accepts `--reuse-audit <behaviour.audit.json> --reuse-revision <revision-name>`;
omitting the revision selects that historical audit's original frozen reveal.
For the prior partial assisted workflow, select `--reuse-assisted-report` alongside
the audit config instead of a revision. Its sealed report is reconciled against
the original mapping, proposal sources and explicit human confirmation files.
Only confirmed numeric answers are imported; unconfirmed LLM proposals are never
imported. Historical source evidence stays unchanged.

### Output tree

Under `artifacts/runs/stage0/<findings-name>/findings/`:

| Product | Content |
|---|---|
| `config.json`, `meta/run_manifest.json` | Frozen family, hypotheses, registration timing, source/settings identities |
| `inputs/` | Verified source completion receipts, raw responses, document sums and CPU statistics runtime |
| `bootstrap/*.npy`, `*.json` | Actual 10,000 index draws, ID ordering, generator and NumPy metadata |
| `units/<contrast>/` | Sealed estimates, paired question rows and paired document rows |
| `results/results.json`, `paired-items.jsonl` | Complete raw analysis and question-level derivations |
| `results/{behaviour,length,kl,writes}.csv` | Independent table exports |
| `plots/*.svg`, `*.png`, `inputs.json` | Exportable figures and their explicit numerical inputs |
| `report.html` | Original raw report, including all comparisons and evidence links |
| `audit/reviewer/` | Blinded review page, packet and current durable draft |
| `audit/drafts/`, `labels/`, `reveal/` | Sealed revision history, frozen labels and revealed comparisons |
| `audit/private/` | Selection reasons, source mapping, preparation checkpoint and verified historical reuse |
| `reports/review-<digest>/` | Immutable report/table/figure snapshots with current review status and label evidence |

Verification checks source receipts, reconciles copied inputs against the source
records and recomputes estimates/intervals from actual saved paired records and
draws. A changed source, altered result, changed registration scope or missing
sealed artifact is rejected. The report labels synthetic fixtures prominently;
software tests are not real GPU qualification or experimental findings.

## Rotation and intruder geometry (Ticket 08, conditional extension)

This CPU-only branch depends on the completed base spectral cache (Ticket 02).
It does not wait for the primary GPU collections or Ticket 07 report, and those
primary findings do not wait for it. No model is instantiated, no new responses
are generated, and no V6/V6c edit is constructed by this ticket.

### Notebook and commands

After CPU sections 1–8, you may jump to optional sections **23–25**:

1. **23 — prepare:** bind the verified base spectral inventory and both saved
   step-64 adapters. The complete real inventory is 196 modules × two arms =
   **392 updated-matrix SVD units**. A synthetic fixture may contain fewer
   modules and is labelled explicitly.
2. **24 — measure/resume:** compute one missing unit by default. Inspect that
   unit's recorded seconds and retained bytes before removing the limit to run
   all missing units. Exact CPU SVD cost depends on module dimensions; do not
   reuse GPU generation estimates or assume this is instantaneous.
3. **25 — verify/inspect:** check seals, source identities and reconstruct the
   reported geometry from the retained arrays. Inspect eligibility for each arm.
   Conditional edit construction belongs to Ticket 09.

```sh
python -m pilot_eval stage0-intruders-prepare --spectral plans/<spectral-name>/stage0.spectral.json --name stage0-geometry-v1-intruder-geometry-v1 --output-root artifacts
python -m pilot_eval stage0-intruders-run --config plans/stage0-geometry-v1-intruder-geometry-v1/stage0.intruders.json --max-new-units 1 --output-root artifacts
python -m pilot_eval stage0-intruders-status --config plans/stage0-geometry-v1-intruder-geometry-v1/stage0.intruders.json --output-root artifacts
python -m pilot_eval stage0-intruders-verify --config plans/stage0-geometry-v1-intruder-geometry-v1/stage0.intruders.json --output-root artifacts
```

Rerun `intruders-run` without the limit to finish. Each arm/module seals its own
result; a half-written unsealed unit is recomputed, verified completed units are
reused. `--no-progress` suppresses console progress without disabling saved state.
One base matrix is loaded for both arms within a module. A stopped writer
requires explicit token/notes through `stage0-intruders-recover-writer`; a live
local writer is rejected. There is no GPU reservation because this is CPU work.

### Quantities and eligibility

Compute `W_new = W0 + b aᵀ` in FP64 from the saved rank-one, alpha-one factors.
Run a full reduced SVD and verify its full reconstruction **before truncating
stored vectors**. Keep all singular values and enough new left/right vectors
for the **base-defined** k30, k50 and k70, floored at 40. Retained vectors are
FP32 with the inherited 1e-6 orthogonality checks. No small-update skip exists.

- **Whole-subspace rotation:** principal angles between old/new top-k left
  subspaces and, separately, right subspaces. Sign flips and rotations within the
  same subspace do not count as changes. Raw angles are radians; maxima are also
  reported in degrees. Retained bases are re-orthonormalized for subspace-only
  arithmetic after checking their stored precision against the inherited gate.
- **Update scale:** a rank-one update has operator norm and Frobenius norm both
  equal to `||b|| ||a||`. Report update/sigma_k and update/(sigma_k−sigma_(k+1))
  for every base energy rank. The gap is from the base spectrum; updated gaps
  are separately recorded.
- **Intruder:** project each new top-20 **left** unit singular direction onto the
  whole old top-m output subspace and take the norm. Primary m=max(k50,40),
  threshold 0.5, with a **strict** below-threshold flag. k30/k70 sensitivities
  use the same floor; thresholds 0.3, 0.4, 0.5, 0.6 and 0.7 are all retained.
- **Chance:** sqrt(m/d_out), reported beside each overlap as an isotropic RMS
  reference. At any threshold **at or below chance**, the module is undecidable.
  Raw below-threshold values remain visible, but it has no edit-eligible indices
  and is counted separately from modules with no intruders.
- **Matching candidates:** keep overlaps, singular values and before/after gaps
  for the new top **40** directions. Detection remains restricted to the top 20;
  non-flagged top-40 directions in decidable modules form the candidate pool for
  the later matched control. This ticket does not choose matches or halve values.

Validate requested ranks against the actual singular dimension and available
bases; never clip m or silently reduce the top-40 set. The real modules and the
source spectral cache already satisfy this shape contract.

### Ties, zero updates and limitations

Always report the measurements even when the update is zero or tiny. A zero
update is an exact operator identity: principal-angle rotation is recorded as
zero. Boundary ties remain marked because a particular top-k selection may still
be nonunique. A candidate inside a repeated singular-value cluster may change
its vector identity under an equally valid SVD basis; raw flags are retained
with this limitation, without inventing a stability or significance claim.

Save exact gaps and a descriptive FP64 resolution indicator
`eps64 × number_of_singular_values × sigma_max`. This marks numerically unresolved
boundaries; it is not a new acceptance threshold or an additional intruder rule.
A positive update over exact zero gap is infinite; zero/zero is undefined; the
last singular value has no next value. JSON uses explicit statuses with null
numeric fields for these cases. No epsilon is added to a ratio denominator.
There is no “rotation under one degree” prediction or gate.

### Artifacts

Under `artifacts/runs/stage0/<intruder-name>/intruders/`:

| Product | Content |
|---|---|
| `config.json`, `meta/run_manifest.json` | Sources, settings, CPU resource declaration, code/NumPy identity |
| `units/<module>-<arm>/geometry.npz` | `new_sigma` FP64; retained `new_u` and `new_v` FP32 |
| `units/<module>-<arm>/result.json` | All principal angles, old/new boundary gaps, update ratios, top-40 overlaps, chance and threshold-sweep statuses |
| `units/<module>-<arm>/complete.json` | Unit identity and payload hashes |
| `results/results.json` | Both arms, all module results and distinct eligible/no-intruder/undecidable counts |
| `results/candidates.csv` | All 40 candidate indices, singular values, gaps, overlap/chance values |
| `results/geometry.csv` | Per-energy-rank angles, ratios and explicit undefined statuses |
| `plots/{rotation,overlap}.{svg,png}`, `inputs.json` | Inspectable figures and numerical figure inputs |
| `report.html`, `results/report.md` | Eligibility report with links to raw evidence |
| `meta/status.json`, `checkpoints/progress.json`, `logs/` | Resume state, failures and explicit writer recovery evidence |

The completion receipt covers arrays, tables, plots and reports. Verification
reconstructs reported overlap/rotation/gap quantities from sealed arrays and
source factors rather than rerunning the full updated-weight SVD. The original
full reconstruction/orthogonality evidence remains hash-bound in each unit.
Changed source weights/adapters, altered settings, corrupted payloads or changed
reported eligibility are rejected. Real-weight geometry is distinct from model
inference, and synthetic fixtures never count as research findings.

## Conditional merged intruder catalogues (Ticket 09; CPU)

This branch consumes completed principal edits and updated-weight geometry. It
builds matrices for the conditional intervention; it does not evaluate a model.
Use CPU notebook sections **26–28** after sections **23–25**. The primary
intervention/report path remains independent of this extension.

```bash
python -m pilot_eval stage0-intruder-edits-prepare \
  --edits artifacts/plans/stage0-geometry-v1-principal-edits-v1/stage0.edits.json \
  --geometry artifacts/plans/stage0-geometry-v1-intruder-geometry-v1/stage0.intruders.json \
  --name stage0-geometry-v1-intruder-edits-v1 --output-root artifacts
python -m pilot_eval stage0-intruder-edits-run \
  --config artifacts/plans/stage0-geometry-v1-intruder-edits-v1/stage0.intruder-edits.json \
  --max-new-units 1 --output-root artifacts
```

Inspect the first unit's time and disk bytes before dropping `--max-new-units` to
finish. `stage0-intruder-edits-status` and `stage0-intruder-edits-verify` take the
same config and output root. In a fresh kernel, run CPU section 1, restore the
paths in 26, then resume 27. No earlier model inference is repeated. Each
module/arm is independently sealed; unsealed partial files are recomputed, and
sealed units are validated before reuse. A changed source, setting or
implementation requires a separately named plan. Writer recovery is explicit:
`stage0-intruder-edits-recover-writer --config ... --owner-token ...
--review-notes ... --output-root ...`. An active same-host writer cannot be
released by that command.

### Operators and matched comparisons

For each applicable arm, form the trained merged weight `M = W0 + B A` and use
its saved singular directions from Ticket 08. Halving a set S means saving the
full FP32 matrix

`M_edit = M − 0.5 × sum(i in S) sigma_i u_i v_i^T`.

Construction and subtraction use FP64; retained singular vectors remain at the
precision declared by the geometry cache. The inherited relative reconstruction
gate is `< 1e-6`. A module may contain multiple flagged directions, so this is a
full matrix edit, without an artificial rank-two restriction.

| Catalogue mode | Directions halved | Module scope |
|---|---|---|
| `unhalved` | None; saves M | All modules with eligible primary flags; loader qualification control |
| `full` (V6) | All eligible primary flags | All eligible modules |
| `matched` (V6) | Eligible primary flags | Modules with enough distinct controls |
| `control` (V6c) | Matched non-flagged directions | Exactly the same modules as matched V6 |
| `ratio` / `ratio-control` | Original pairs with sigma_control / sigma_flagged in [0.8, 1.25] | Same retained pairs on both sides; no rematching |

Primary detection is still new top-20, overlap norm `< 0.5` against the old
`max(k50,40)` subspace, and only where `0.5 > sqrt(m/d_out)`. Control candidates
come from the new top-40 and must independently have overlap norm `>= 0.5`.
Choose distinct controls minimizing total absolute singular-value difference;
resolve equal objectives by control index. A short-control module is excluded
from **both matched edits**, while full V6 retains its flags. Pair ratios with a
zero flagged singular value are undefined and excluded from the sensitivity;
unrepresentable ratios are recorded explicitly, never as JSON Infinity.

An arm with no eligible flags is **not applicable**, rather than an empty
successful experiment. If both arms are inapplicable, verification still
publishes its receipt and the runner schedules no extension inference. Empty
matched/ratio variants have their own not-applicable states. Both arms use the
same rules; GRPO flags are not silently discarded.

### Saved objects and loader contract

Under `artifacts/runs/stage0/<name>/intruder-edits/`:

- `config.json`, `meta/run_manifest.json`: parent identities, selection/matching
  rules, implementation/NumPy identity and upper disk estimate.
- `units/<module>-<arm>/matrices/*.npy`: full FP32 matrices. Identical selection
  masks share a payload within the unit; source weights and adapters are intact.
- `units/<module>-<arm>/result.json`, `complete.json`: selections, before/after
  singular values, reconstruction errors, timing, bytes and payload hashes.
- `results/catalogue.json`: separate full/matched/ratio identities, original
  references, matrix paths and exact per-module adapter-disable masks.
- `results/matching.csv`: every matched pair, singular values, ratio and
  sensitivity inclusion. Excluded modules remain in the catalogue.
- `report.html`, `results/catalogue.md`, `meta/status.json`, `logs/`: inspectable
  summary, completion/resume state and errors.

Verification reconstructs each saved operator from frozen base weights,
adapter factors and singular components. Rehashing an altered matrix is not
enough to make it valid. CPU verification imports no Torch/Transformers/PEFT.

The mixed loader replaces the base parameter object only while a variant is in
use. It disables the adapter **in edited modules**, retaining the original source
adapter everywhere else, and restores original objects and flags on exit,
including errors. A deliberate double-adapter mode is restricted to the
unhalved negative-control diagnostic. The catalogue implements the existing
greedy engine interface, but remains `gpu_qualified: false`. Ticket 10 must run
and review bounded unhalved logit/greedy checks, reconcile the cumulative
20-unit budget and freeze the conditional evaluation before real inference.
No CPU catalogue or synthetic test substitutes for that qualification.

## Conditional evaluation and findings (Ticket 10)

Ticket 10 uses the shared saved workflows, with the verified Ticket 09 catalogue
as the qualification source. Primary products keep their names and are not
replaced. In both notebooks, `CONDITIONAL_EXTENSION = False` remains the default.

### Execution sequence

1. **CPU notebook 23–28:** complete geometry and the conditional catalogue. Check
   each arm's applicability. A verified not-applicable catalogue stops this
   extension before any model load or budget reservation.
2. **GPU notebook 1–4:** select `CONDITIONAL_EXTENSION = True`, restore the
   pinned environment, audit and **existing cumulative** Stage 0 usage ledger.
3. **GPU notebook 5–9:** prepare with `--catalogue`, inspect the frozen scope,
   record observed usage and explicitly bound the loader diagnostic. Test the
   unhalved saved matrices against the original adapter, using the same 16
   held-out sequences for each required arm/mask layout. Retain all answer-position
   maximum errors, worst coordinates, greedy outputs and double-adapter controls.
4. **GPU notebook 9A–11:** reconcile actual diagnostic usage, review the numerical
   evidence, approve exactly the proposed `3E` gate, and separately approve/freeze
   the measured production reservation. `E > 1e-3`, an insensitive negative control
   or an unaccepted greedy mismatch prevents qualification. A near-tie exception
   is explicitly recorded and does not become exact token agreement.
5. **GPU notebook 12–15:** prepare and collect paired greedy responses with both
   original references freshly evaluated in this same recorded runtime. Empty
   sensitivity variants are omitted from inference, while their reasons remain
   in the catalogue. Reconcile actual usage after stopping or completing.
6. **GPU notebook 12A and 16–19:** freeze the comparison family before collection,
   prepare fixed contexts, separately reserve a bounded KL/write attempt, measure
   and reconcile it. The first bounded units provide an actual throughput measure;
   a forecast is not a completion guarantee.
7. **CPU notebook, flag True, sections 17–22:** compute the extension statistics,
   prepare the targeted blinded intended-answer review, save/freeze/reveal labels,
   and publish a report snapshot. Unfinished review is reported honestly; raw
   automatic findings remain accessible.

The GPU model remains singleton FP32 with TF32/autocast off. Grouping eight calls
into a durable unit is storage grouping, not larger inference batching. A changed
session cannot resume the old paired collection: retain it and perform a newly
qualified, separately named comparison. References are rerun rather than joining
historical results without agreement evidence. Confirmed labels are reused only
for identical prompt/response text through the existing review workflow.

### Public commands and artifact identities

Use the same command families as the primary path:

```bash
python -m pilot_eval stage0-qualification-prepare \
  --catalogue artifacts/plans/stage0-geometry-v1-intruder-edits-v1/stage0.intruder-edits.json \
  --name stage0-geometry-v1-intruder-qualification-v1 --output-root artifacts
```

`--edits` and `--catalogue` are mutually exclusive. Then use
`stage0-qualification-{runtime,run,status,verify,approve,freeze}` with that qualification
config; `stage0-evaluation-{prepare,run,status,verify}` with its frozen execution;
`stage0-drift-{prepare,run,status,verify}` for its separately budgeted fixed
contexts; and `stage0-findings-*` / `stage0-findings-audit-*` for its CPU analysis.
Qualification also exposes `stage0-qualification-recover-writer`. All writer-recovery commands retain exact owner-token/notes
requirements. Never reset the ledger or release an active reservation simply to
start this branch; observe the new cumulative usage and settle it explicitly.

Default extension products are distinct from the primary path:

| Product | Canonical location below artifacts/ |
|---|---|
| Qualification and frozen execution | `plans/<Stage0>-intruder-qualification-v1/stage0.{qualification,execution}.json` |
| Diagnostic units, review receipts and forecasts | `runs/stage0/<Stage0>-intruder-qualification-v1/qualification/` |
| All generated responses, token IDs and scores | `runs/stage0/<Stage0>-intruder-qualification-v1/paired-evaluation/` |
| Scalar KL arrays and direct-write sufficient sums | `runs/stage0/<Stage0>-intruder-fixed-context-v1/fixed-context/` |
| Registered family, paired intervals, review and figures | `runs/stage0/<Stage0>-intruder-findings-v1/findings/` |

The cumulative ceiling remains **20 Colab units for all Stage 0 work**, including
primary and conditional diagnostics, failed attempts and resumed runs. Each job
needs a current observed usage and a bounded reservation. No model-backed action
is authorized by merely preparing or CPU-testing a plan.

### What differs from the primary analysis

- Every executable full, matched, matched-control, ratio and ratio-control variant
  is compared with its own original adapter. Matched V6 is additionally compared
  directly with matched V6c; the same is done for retained ratio pairs.
- Full/matched/ratio identities and module masks never collapse into one result.
  Arms without eligible flags, chance-undecidable modules, short-control exclusions
  and empty ratio masks remain visible. Shared identical matrices may reuse a
  payload, but their declared comparisons remain separate.
- Intervals use the existing saved 10,000 paired bootstrap draws, seed 42 and
  pointwise 95% percentiles. “Accuracy preserved” still requires the lower bound
  to be strictly above −5 percentage points; n=150 may be inconclusive. These
  conditional results do not replace or silently extend the primary family.
- Full-vocabulary KL is still `KL(variant || untuned)` on the original 150 GSM8K
  and 150 FineWeb fixed sequences, with corpus/view denominators preserved.
- In merged modules, the effective write is the full `(W_edit − W0)x` on the
  adapted model's module inputs. Untouched modules retain their real LoRA branch.
  The implementation computes the direct matrix action in FP64; it does not
  subtract rounded module outputs or label the full edit rank one. The denominator
  uses the original base module output, including its unchanged bias.
- `sum_factor_norm_products` is null for full-matrix variants because a single
  pair of rank-one factors no longer represents that operator. Original reference
  factor statistics remain available. Null is not zero, and the report explains
  its scope. No new proxy norm or acceptance tolerance is introduced.
- The loader retains a read-only CPU matrix cache for the current variant. It
  releases that cache on variant changes and on close, preserves original base
  parameter objects, and verifies the frozen mixed source catalogue at load and
  full attempt-end integrity checks. A source change invalidates the attempt's
  evidence rather than silently producing mixed-source results.

Implementation validation uses CPU synthetic fixtures and mathematical oracles.
Only a separately approved real-CUDA diagnostic establishes GPU qualification;
software completion does not establish a scientific finding.

## Optional untuned rollout completion (Ticket 11; R0)

This supporting branch depends only on the source audit (Ticket 01). It does not
block primary interventions or the conditional intruder report. It recovers raw
untuned sampling groups, freezes their generation identities and publishes answer
memberships for later NLL/gradient stages. It does not train an adapter.

### Route selection and execution

1. **CPU 29:** select the original baseline configuration. Restore its raw batch
   JSON/seals, runtime, referenced GRPO preparation and saved training inputs.
   The audit reports missing files and verifies that all 128 groups belong to
   the common 512 training problems. Aggregate baseline accuracy is insufficient.
2. **CPU 30:** stage the pinned local base-weight/tokenizer snapshot and prepare.
   This verifies actual base weights, freezes old groups and the common training
   order, and records explicit seeds/settings. Parquet support is required before
   GPU work; install the `stage0-rollouts` extra or PyArrow on CPU if absent.
3. **GPU setup 1–4, then 20:** inspect the current recorded GPU/software and budget.
   Use the original GPU type/software for replay when identified and available.
   An identified mismatch disables the historical join and selects fresh-uniform
   timing instead. Unknown metadata remains labelled unknown, not equal.
4. **GPU 21:** explicitly approve a bounded two-question profile. With recoverable
   old identities, replay the first two recorded groups at their saved seeds:
   16 draws, not 16 independently chosen questions. Otherwise time the first two
   frozen new-uniform groups. No production begins before profile completion.
5. **GPU 22–23:** inspect exact evidence, reconcile actual cumulative usage and
   separately approve/freeze the production route and measured forecast.
6. **GPU 24:** collect one eight-draw group first, inspect throughput, then set the
   limit to `None` to finish only missing units. Each group is sealed atomically.
   Exact same-session profile draws are reused when their input/seed identities
   match production. Stop/reconcile/renew an exhausted or unknown-time allocation.
7. **GPU 25, CPU 31:** reconcile actual usage, release the GPU after collection,
   and verify/inspect raw tables, memberships and the HTML report on CPU.

User-approved route rule, 27 October 2026:

- **Exact replay:** all 16 draws match token IDs, text, stopping metadata and
  recorded identities/seeds, with source/settings hashes and procedure verified.
  Reuse the 128 historical groups and generate only the remaining 384 groups.
- **Any mismatch or unavailable historical seed/order:** do not join or substitute
  a reward-rate test. Generate a **fresh uniform 512×8 cohort in the new session**.
  Preserve the original 128 as a separately labelled historical product.
- Old/new reward rates are descriptive. No five-point rate gate or tolerance
  fitted to replay errors is used. Same GPU model alone is not compatibility.

Historical seeds are `42 + recorded batch index`, not `42 + GSM8K source index`.
For an extension, retain the historical 128 ordering/seeds, append remaining
questions in the saved optimizer order and continue the ordered-batch sequence
at 170. A fresh-uniform route uses the complete saved optimizer order and its
explicit frozen seed sequence from 42. Both routes use HF, untuned FP32, TF32 off,
no autocast, temperature 1, top-p 1, top-k 0, cap 1024, one question and eight
prompt copies per generation group. Do not change engines to meet a timing guess.

### Commands and durable products

Public family: `stage0-rollouts-{audit,prepare,runtime,profile,profile-verify,
freeze,run,status,verify,recover-writer}`. All accept `--output-root`; audit/prepare
select `--audit`, `--baseline`, `--name`, and prepare also selects `--weights-dir`.
Other operations consume `--config` pointing to `stage0.rollouts.json`.

```bash
python -m pilot_eval stage0-rollouts-audit \
  --audit artifacts/plans/stage0-geometry-v1/stage0.plan.json \
  --baseline artifacts/plans/pilot4-untuned-sampling-v1/baseline.config.json \
  --name stage0-geometry-v1-untuned-rollouts-v1 --output-root artifacts
```

The immutable preparation/frozen execution live under
`artifacts/plans/<R0-name>/stage0.rollouts{,.execution}.json`. Products live under
`artifacts/runs/stage0/<R0-name>/rollouts/`:

| Product | Contents |
|---|---|
| `inputs/source-audit.json` | Verified old groups, missing raw inputs, seed recovery and source hashes |
| `inputs/{training,historical}.jsonl` | Frozen training/gold inputs and old draw snapshots |
| `profile/units/`, `profile/results/profile.json` | Replayed/timing groups, exact differences, route, measured cap-extrapolated forecast |
| `collection/units/` | Eight raw draws per completed group, identity/hash/integrity/attempt receipts |
| `profile/attempts/`, `collection/attempts/` | Reservation, elapsed time, final integrity and interrupted-attempt evidence |
| `results/rollouts.jsonl` | All 4096 target draws, tokens, seeds, origin, three scorers and cap outcomes |
| `results/rollouts.parquet` | Flat problem/draw/source/seed/text/token/score columns for analysis |
| `results/datasets.json` | Gold, own_all, own_correct, own_correct_one, k/8, dropped IDs and selection rule |
| `results/gold.jsonl`, `results/groups.csv` | Full saved gold rows and per-question correctness counts |
| `results/historical/` | Separately labelled original 128 groups and their memberships |
| `report.html`, `results/results.json` | Human-readable counts/route and structured summary |

`own_correct` uses current flexible-v3 correctness and excludes capped draws;
strict and flexible-v2 are retained alongside it. Historical training rewards
remain separately recorded. Every question has eight unique target draws.
Zero-success questions remain in own_all, receive observed k/8 = 0 and appear in
`dropped`; this does not establish zero true success probability. Select one
correct draw uniformly with a dedicated NumPy PCG64 seed 42, saved question order
and sorted draw indices. Negative-log(k/8) is a whole-answer quantity for eligible
questions, not per-token KL or NLL. Later tickets score each unique answer once
and aggregate via these memberships.

Reservations use the **same cumulative Stage 0 20-unit ledger**. Unknown usage or
another active reservation blocks model loading. Optional R0 cannot consume a
primary reservation. Resume checks runtime/session, source hashes, prompt IDs,
code, tokenization and group identities. Unknown prior elapsed time requires
reconciliation; restarting never resets cost. Writer recovery takes
`--phase profile|collection --owner-token ... --review-notes ...`, refuses a live
same-host writer and retains all sealed evidence and budget charges.

The forecast extrapolates measured eight-draw time to the 1024-step cap, with
model-load/final-integrity costs; it is not a guaranteed runtime. No real runtime
estimate has been established by the CPU synthetic tests. Numerical/source
failures retain diagnostics and do not widen the frozen rules.

## Optional teacher-forced answer measurement (Ticket 12; E0.6a)

This branch consumes verified Ticket 11 rollouts and memberships. It scores the
**untuned base model**, with no adapter or optimizer step. It does not recompute
answers or generate new responses. Exact duplicate prompt/answer token sequences
share one forward; repeated draws still contribute their original multiplicity
when memberships are aggregated.

### What is measured

At target answer token t, use the model distribution at context t−1:

- **Surprise:** −log p(target), in nats.
- **Entropy:** −sum_v p(v) log p(v), over the full vocabulary.
- **Rank:** one plus the number of logits strictly greater than the target logit;
  ties share rank (user-approved convention).

Primary dataset NLL is total surprise divided by counted answer tokens. Model
entropy uses the same positions and weighting. Their difference is **excess
surprise**, a descriptive statistic, **not** KL(data || model). Gold uses saved
SFT input IDs/labels: mask the prompt and supervise through the existing EOS.
Sampled answers use saved generated IDs, including observed EOS; capped answers
get no invented EOS. Cap outcomes remain in own_all as observed samples.

The model body runs once per unique answer. The output head projects bounded
context chunks, with full-vocabulary FP64 reductions; no full [sequence,vocab]
production logits are retained. Save per-token surprise/entropy/rank arrays,
target IDs and prediction-position mappings. Whole-answer negative-log(k/8)
remains separate from per-token statistics and receives no agreement test.

### Workflow and storage

- **CPU 32:** verify R0, gold masks/EOS and memberships; prepare unique scoring
  inputs. This can run in a separate CPU notebook while the sampled GPU session
  remains available. Do not disconnect/restart between R0 and scoring if keeping
  this same-session contract.
- **GPU setup 1–4, 26:** restore plan, inspect sampled-runtime agreement and the
  existing cumulative Stage0 budget. No model loads during preflight.
- **GPU 27:** explicitly reserve a bounded single-answer profile. One unique
  answer is scored and sealed. Repeat bounded single units for broader timing
  coverage if needed; no unbounded scoring is allowed yet.
- **GPU 28:** build a forecast from measured per-answer time and total squared
  sequence lengths; record actual cumulative usage and settle profile allocation.
- **GPU 29:** separately review/freeze the measured production budget. The bound
  must cover the saved forecast and fit the global **20-unit** limit.
- **GPU 30–31:** score/resume missing unique answers, reconcile actual usage and
  verify measurement completion. Full base-integrity failure invalidates that
  attempt's sealed results. Changed sampled runtime/session is rejected.
- **CPU 33:** aggregate and verify report/intervals. GPU is not needed for
  bootstrap, report generation, or rechecking saved arrays.

Public commands are `stage0-distance-{prepare,runtime,run,forecast,freeze,status,
verify-measurement,report,verify,recover-writer}`. Preparation accepts `--rollouts`
(the verified `stage0.rollouts.json`), `--name`, `--context-chunk` and
`--workspace-bytes`; every command uses `--output-root`. Other commands consume
`--config` pointing to `plans/<name>/stage0.distance.json`. `run` supports one
bounded profile unit, frozen production, explicit renewed `--approval`, unit
limits and progress suppression. `freeze` consumes a separate measured review.
Writer recovery requires the exact recorded owner token and notes and cannot
release a live same-host writer.

Outputs are under `artifacts/runs/stage0/<name>/e06a_distance/`:

| Product | Content |
|---|---|
| `config.json`, `meta/run_manifest.json` | R0/weight/runtime identities, masking and scoring policy, counts, code and NumPy identity |
| `inputs/answers.jsonl` | One input/target/context-position mapping per unique answer |
| `inputs/memberships.json`, `inputs/datasets.json` | Original problem/draw multiplicities, correct subsets, dropped IDs and probabilities |
| `units/<answer-id>/tokens.npz` | FP64 surprise/entropy and integer rank, one entry per supervised token |
| `units/<answer-id>/{metadata,complete}.json` | Runtime/attempt/integrity identity, seconds and payload hashes |
| `attempts/`, `results/forecast.json`, `production-freeze.json` | Timing, bounded-credit identity, measured forecast and immutable production review |
| `measurement.complete.json` | GPU collection receipt; CPU report can still be pending |
| `summary.parquet`, `results/summary.csv` | Dataset NLL, entropy, excess surprise and separately named intervals |
| `results/per-problem.csv` | Problem-level sufficient summaries and whole-answer negative-log correctness |
| `results/top-gold-tokens.csv` | Top50 gold token occurrences by surprise, with IDs, positions, entropy and rank |
| `bootstrap/*.npz` | Saved 10,000 whole-problem resampling index arrays, seed42 |
| `results/results.json`, `report.html`, `complete.json` | Reconstructible report, gate status and final evidence receipt |

`ci_low`/`ci_high` refer only to primary NLL. Excess-surprise bounds use
`excess_surprise_ci_low/high`. Bootstrap resamples whole problem IDs while
recomputing token-weighted ratios; it does not treat tokens or repeated draws as
independent questions. Negative-log correctness averages eligible problems
without weighting by number of correct samples. Empty correct cohorts are
explicitly unavailable, rather than zero.

The own_all excess-surprise gate remains **absolute value ≤0.02 nats/token**.
The report states its premises: identical untuned distribution, supplied prefixes,
observed stopping, and stated weighting. A finite stopped-sequence token-weighted
ratio has no exact finite-sample zero identity. Failed gates preserve the report
and mark `failed-gate`; they do not authorize wider tolerances. This gate is not
claimed to establish a valid KL identity or a reasoning-quality result.

Positional summaries use disjoint first32/middle/last32 windows. First32 takes
precedence on answers shorter than64, avoiding duplicate counting; empty windows
are absent. `results/position-windows.csv` retains NLL and separately named
excess-surprise intervals. The user approved these conventions on27 October2026. Dataset summaries also
record `n_answers_under_64`, all three window token counts (including zero), and
`n_answers_without_end_of_turn`. Counts retain answer/draw multiplicity, not only
unique forwards. Complete token evidence permits CPU-only reporting revisions.

`meta/vocabulary.json` records the full output-head row count, tokenizer size and
padded-row count, explicitly including padded rows. Rank and NLL use the same
full FP64-promoted FP32 logit vector. Every unit binds the vocabulary receipt;
verification rejects ranks or target IDs outside its full vocabulary. The final
measurement seal covers this receipt and the report displays it.

Summary Parquet/CSV include `answer_weighted_nll`, `answer_weighted_entropy` and
`answer_weighted_excess_surprise`, with their own independently named95% interval
fields. This sensitivity averages each answer's per-token mean equally, retaining
membership multiplicity. Bootstrap still resamples whole problems: all answers
of a selected problem move together, and the sensitivity denominator is the
number of answers, not the number of problems. Primary token-weighted estimates
and the own_all gate are unchanged.

# Pilot 3 instrument: diagnostic stage

Implemented stages so far: frozen input preparation/audit and validation of one
diagnostic batch across saved checkpoints 0, 8, 16, 32 and 64. Production
profiling, full-cohort resume, scientific plotting and guided Colab are later
tickets. A diagnostic completion does not mean Pilot 3 is scientifically complete.

## Inputs and commands

Preparation takes the saved Pilot 2 SFT configuration inside the current artifact
root. It verifies the associated source manifests/checkpoints, preserves the
150 held-out items and creates the approved FineWeb sample. It requires pinned
dataset/tokenizer access, but does not load model weights.

```sh
python -m pilot_eval activation-prepare --source-config SOURCE_SFT_CONFIG --name PLAN_NAME --output-root ARTIFACT_ROOT
python -m pilot_eval activation-audit --config PREPARED_CONFIG --output-root ARTIFACT_ROOT
python -m pilot_eval activation-validate --config PREPARED_CONFIG --batch-size 2 --output-root ARTIFACT_ROOT
```

Use actual paths for the uppercase placeholders. Preparation prints the prepared
configuration path. The shared Drive artifact root is beneath `SPAR/spar-pilot`.
Source evidence remains unchanged; preparation records relative paths and hashes.

Default validation uses the longest GSM8K and FineWeb inputs, so all three views
are represented. A larger diagnostic batch includes additional frozen rows;
its membership and runtime are recorded. This is a bounded instrument check,
not a batch-speed recommendation or proof that production fits in GPU memory.

Real validation requires one CUDA GPU and the validated dependency pins. CPU
tests inject controlled source access and construct tiny Qwen/PEFT models from
configuration without downloads. No GPU-family whitelist is imposed.

## What is measured

- Block outputs: adapted minus untuned output on identical model-input tokens.
- Direct module contribution: the module's own LoRA write on its adapted-pass
  input; ordinary output uses the same input, including any frozen bias.
- Per-example vector/norm sums and counts: sufficient for the approved weighting
  schemes and later example bootstrap. Full token activations are temporary.
- Exactly zero step-0 write; frozen-base identity; scaled rank-1 branch and output
  subtraction; disabled-adapter invariance after switching checkpoints.

The main reference cache contains one batch's untuned block outputs. Diagnostic
invariance checking runs additional disabled-adapter passes; it is validation
overhead. The transformer-body path avoids full vocabulary logits and is tested
against the causal-LM path's actual block hooks.

Model work is FP32/no-grad/eval, without autocast or TF32; sufficient-summary
reductions use FP64. Actual input positions and numerical settings are recorded.
Module and block quantities must not be interpreted as equivalent causal effects.

## Evidence and failures

The run's validation directory contains a deterministic diagnostic batch folder:
config and runtime identity, per-checkpoint NPZ summaries and JSON validation
evidence, structured diagnostic results, progress/status and error logs.
A checksum completion marker is written last. Completed diagnostics verify all
required evidence before reuse. An interrupted diagnostic can be retried under
the same identity; it currently repeats the bounded batch, while production
shard-level resume is implemented in its separate ticket.

Failed checks persist their stage and available numerical diagnostics, remove
temporary hooks and restore the caller's adapter weights/state. They do not
publish successful completion. Nonfinite values or a nonzero step-0 write stop
even if an approximate tolerance could otherwise hide the change.

Thresholds are fixed engineering acceptance rules. Validation saves errors,
fractions of thresholds and sampled subtraction-resolution counts; near-threshold
or unexpected behavior needs inspection and expanded validation before production.
Local CPU success is not a claim that the real Drive checkpoints passed on GPU.

## Profile and review a production batch plan (Ticket 3)

After reviewing the input audit, run the following on the single GPU. Profiling
verifies/reuses the instrument diagnostic first; it never bypasses a failed gate.
Replace the illustrative paths and names with your actual prepared plan.

```bash
python -m pilot_eval activation-profile \
  --config artifacts/plans/write-v1/activation.prepared.json \
  --output-root artifacts
```

Every candidate (1, 2, 4, 8, 16) processes the same **16 examples**: the eight
longest complete GSM8K sequences and eight longest FineWeb sequences, ordered by
length then ID within each corpus. This includes all three token views and the
actual GPU reference cache. It is a deliberately demanding profiling workload,
not a new scientific evaluation cohort.

Each candidate warms up its longest batch with all five checkpoints. Warmup and
model loading are outside timing. CUDA synchronization brackets reference capture
and each validated checkpoint measurement; peak allocated and reserved memory
are recorded after warmup. The model is loaded once per candidate, with examples
parallel within its batches and checkpoint passes sequential. No GPU name whitelist
is used.

`input_tokens_per_second` counts original unpadded sequence tokens completed through
the **whole five-checkpoint measurement workflow** per second. It is not generation
speed or a per-pass model throughput claim. Saved checkpoint timing separates
adapter switching, disabled-reference checks, hashes and rank-1 checks from the
adapted forward plus summary reductions. The latter still includes finite/zero
checks and timing instrumentation. Use the whole-workflow timing for batch choice;
this small workload does not guarantee a full-run duration.

Numeric evidence includes per-example summaries for every checkpoint. Agreement
with batch 1 requires the approved `atol=1e-5, rtol=1e-5`, exact counts and
undefined coverage, and agreement of derived measurements under both weightings.
OOM records an unsuitable candidate and continues after cleanup. Other errors,
including numerical disagreement, stop and persist diagnostics. Verified completed
candidates are reused after interruption; marked corruption stops.

The result is saved under the prepared run's `profile/results.json`. Review the
speed, memory, numeric agreement and validation evidence. The fastest passing
candidate is a **provisional recommendation**, not approval. Explicitly freeze
one reviewed passing candidate:

```bash
python -m pilot_eval activation-freeze \
  --config artifacts/plans/write-v1/activation.prepared.json \
  --profile artifacts/runs/pilot-3/Qwen--Qwen2.5-1.5B-Instruct/gsm8k-fineweb/heldout150-control150-seed42/float32/write-v1/profile \
  --batch-size 8 --name write-production-v1 \
  --review-notes 'Reviewed timing, peak memory, agreement and instrument evidence.' \
  --output-root artifacts
```

The freeze step runs on CPU and re-verifies saved diagnostic/profile hashes and
numeric evidence. It writes `plans/write-production-v1/activation.execution.json`,
containing ordered memberships for **all 300 prepared examples**, input/profile
hashes, thresholds, numerical/runtime identity and review notes. It leaves prepared
inputs unchanged. Repeating identical freeze settings is safe; changing them needs
a different execution name. Production must check the runtime again when it starts.

Full-cohort resumable measurement is Ticket 4; profiling/freezing do not claim that
the scientific run has completed. The guided Pilot 3 notebook follows in Ticket 6.

## Execute and resume full-cohort measurement (Ticket 4)

Use the frozen **execution** manifest, not the prepared-input manifest. The
execution name must differ from the preparation name so their run directories
remain separate.

```bash
python -u -m pilot_eval activation-measure \
  --config artifacts/plans/write-production-v1/activation.execution.json \
  --output-root artifacts
```

This processes all 150 GSM8K and 150 FineWeb sequences. Examples run in parallel
inside the frozen batches; the five checkpoints run sequentially on one loaded
model. Every batch gets a bounded untuned reference cache. Before reuse, each
checkpoint's disabled-adapter pass must still match that reference. The cache is
released after the batch and on errors.

For each batch the journal stores:

```text
batches/batch-000000/
  baseline/
    summaries.npz       # block counts, vector sums, magnitude sums
    metadata.json       # IDs/input hashes, axes, shapes, FP64 types
    complete.json       # written last; identity and payload hashes
  checkpoints/step-0/   # also step-8, step-16, step-32, step-64
    summaries.npz       # block deltas; module ordinary/contribution/ratio sums
    metadata.json
    validation.json     # hooks, sampled positions, gate errors and coverage
    complete.json
```

The shared baseline is for **untuned block outputs**. Each checkpoint retains its
own module ordinary-output magnitudes, because those are measured on that
checkpoint's adapted inputs. Full token activations are temporary, not saved.

The run also saves immutable `config.json` and `meta/run_manifest.json`, mutable
status/progress/logs, and `results/aggregate-sums.npz` plus `results/results.json`.
The final arrays contain verified full-cohort sufficient sums; per-example shards
remain available for equal-example weighting and bootstrap analysis. Interpretation,
plots and confidence intervals belong to the CPU report in Ticket 5.

### Resume rules

Repeat the exact command after an interruption:

1. Verify source, preparation, reviewed profile and execution/runtime identities.
2. Read completion markers for the expected batches/checkpoints. Validate hashes,
   array shapes/counts, hook identities and saved rank-1 evidence before skipping.
3. Skip a fully completed batch without model work. For a partial batch, recreate
   its untuned activations and require exactly matching saved baseline summaries.
4. Process only its missing checkpoint combinations. Replace owned **unmarked**
   leftovers; preserve completed siblings and do not count progress twice.
5. Check the frozen base and verify every required combination before publishing
   final completion. Aggregate each batch/checkpoint exactly once.

The progress file lists verified `(batch index, checkpoint step)` pairs. Markers
and payloads are authoritative; status/progress alone are insufficient. Corrupt
marked data stops. Production OOM, nonfinite outputs and failed gates stop while
preserving verified shards; they do not change batches, precision or input length.
Changed execution settings/runtime/code identity require a separately named,
reviewed plan. A completion marker whose required files have disappeared is
corrupted completion evidence, not permission to silently regenerate those files.

### Monitor the process; verify scientific completion separately

The library helper `monitor_activation_process(process, run_directory,
last_snapshot=...)` checks the actual `subprocess.Popen` child and tolerates empty,
partial, missing or invalid mutable status snapshots. It can retain a labelled
last-known snapshot. Neither a stale `completed` status nor exit code zero alone
establishes scientific completion. Use the strict CPU-only command afterward:

```bash
python -m pilot_eval activation-verify \
  --config artifacts/plans/write-production-v1/activation.execution.json \
  --output-root artifacts
```

Before restarting after a notebook-cell interruption, inspect the earlier child
PID (for example `!pgrep -af 'pilot_eval.*activation-measure'`). It can still be
running. The workflow rejects a second writer through the run lock; use one writer
for a named run. Keep the model job in its CLI subprocess so GPU ownership ends
when that process exits. The guided notebook will use these commands/helpers in
Ticket 6.

Local CPU fixtures exercise persistence and recovery, and tiny locally constructed
Qwen/PEFT models exercise hooks and reference summaries. These tests do not replace
real-source validation and CUDA execution in Colab.

## Reconstruct the scientific report on CPU

After `activation-verify` succeeds, the GPU can be disconnected. In a CPU
session, install the package's report dependencies (`pip install -e '.[report]'`)
and mount the same artifact root, then run:

```bash
python -m pilot_eval activation-report \
  --config artifacts/plans/write-production-v1/activation.execution.json \
  --output-root artifacts --name write-report-v1
```

The report verifies complete source shards and instrument gates before reading
measurements. It reconstructs token-weighted and equal-example block/module
measurements, then uses 2,000 example bootstrap draws with seed 42. GSM8K draws
are shared across question/solution views, layers, modules and checkpoints;
FineWeb draws are independent. Ratios are recomputed from each resample's sums.
Undefined replicates are excluded from intervals with explicit coverage.

Outputs live in `artifacts/reports/write-report-v1/`:

- `config.json`: full preparation/execution identity, source completion hash,
  bootstrap and weighting policy.
- `results/results.json`, `measurements.csv`, `report.md`: measurements,
  intervals, denominator/zero coverage, limitations and validation summaries.
- `results/mean-vectors.npz`: mean baseline and change vectors keyed by
  step/view/layer/weighting.
- `results/bootstrap-draws.npz`: local cohort indices in frozen input order.
- `results/bootstrap-primary.npz`: finite scalar replicate payloads and separate
  `defined-*` masks. A masked zero is storage, **not a measured zero**.
- `results/validation.jsonl`, `validation-summary.json`: sampled rank-1 errors,
  fixed-threshold fractions and below-resolution coordinate coverage.
- `plots/`: block curves with intervals, module heatmaps for both weightings,
  baseline-denominator sensitivity, PNG/SVG files and a scale manifest.

A repeat verifies and reuses a complete report. Corrupt marked outputs stop;
changing sources requires a new named report. Interrupted unmarked report work
is reconstructed from verified measurements without inference. Both report and
source hashes are checked; status alone never authorizes reuse.

Read token and equal-example profiles together: longer gold solutions carry
more weight in token averages. Their lengths are fixed teacher-forced inputs,
not the adapter's generated response lengths. Intervals describe the frozen
cohorts and one training seed. These geometry measurements do not establish
causality, task specificity or useful layers for fine-tuning.

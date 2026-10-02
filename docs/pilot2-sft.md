# Pilot 2: exploratory rank-1 GSM8K SFT

Use [the guided notebook](../notebooks/pilot-2-colab.ipynb) on a Colab
L4 (the original T4 variant remains supported). The actual Qwen GPU run is not verified locally. CPU tests cover the workflow
and a real tiny Qwen architecture built from configuration without downloads.

## What runs

- Qwen2.5-1.5B-Instruct at the Pilot 1 frozen model/tokenizer revision.
- 512 seed-42 samples from the pinned official GSM8K train split, with original
  gold annotations and final-answer markers.
- Rank-1 LoRA, alpha 1, dropout 0, on all 196 intended linear projections.
- Unquantized FP32, microbatch 1, accumulation 8, 64 optimizer updates, LR 1e-4,
  constant schedule, no warmup, AdamW, weight decay 0, gradient clipping 1.0.
- Completion-only supervision, including end-turn; no packing or truncation.
- The current L4 notebook uses matched evaluation batch 2; original T4 plans use batch 1.
- A fresh FP32 untuned baseline and checkpoints 0/8/16/32/64, greedy GSM8K only,
  using the same 150 held-out items and strict/flexible-v2 scorers.

The analysis is exploratory with no directional prediction or binary collapse
threshold. Historical accuracy 0.640 → 0.420 and length 288 → 77 are contextual.
Your older BF16 baseline is preserved and is not substituted for the matched
FP32 measurement.

## Commands

Install `requirements-pilot2.txt` and the repo package before importing training
libraries, then restart Colab. Use the same persistent artifact root as Pilot 1.
The source must be an untuned GSM8K configuration with its frozen item file
still present; a combined report or a lone downloaded config is insufficient.

```bash
python -m pip install -r requirements-pilot2.txt
python -m pip install --no-deps -e .
pytest -q
python -m pilot_eval sft-prepare --source-config artifacts/plans/baseline-batch8-v1/gsm8k-0shot.config.json --name pilot2-sft-fp32-v1
python -m pilot_eval sft-train --config artifacts/plans/pilot2-sft-fp32-v1/sft.config.json --preflight-only
python -m pilot_eval sft-evaluate --config artifacts/plans/pilot2-sft-fp32-v1/sft.config.json --checkpoint baseline
python -m pilot_eval sft-train --config artifacts/plans/pilot2-sft-fp32-v1/sft.config.json
```

After training, run `sft-evaluate` for checkpoint `0`, then `8`, `16`, `32`, `64`.
Checkpoint zero must match the untuned outputs before interpreting later changes.
Finally run:

```bash
python -m pilot_eval sft-compare --config artifacts/plans/pilot2-sft-fp32-v1/sft.config.json
```

Every operation accepts `--output-root`. In Colab the notebook supplies your
Drive artifact root to every command. Training and inference are separate
processes; each releases GPU memory on exit. Strict and flexible-v2 score the
same saved generations, without a second inference pass per scorer.

## Artifacts and recovery

The frozen plan contains tokenized training examples, actual supervised labels,
evaluation items, length diagnostics and a dated exploratory analysis plan.

Training lives under the canonical `runs/pilot-2/.../rank1-float32/<run-id>` tree.
It saves config, runtime, input order, loss/LR/gradient logs, preflight evidence,
base hashes and complete checkpoint states. Evaluation uses `runs/pilot-2-eval`.
Flexible-v2 reports preserve raw source outputs. The trajectory report contains
JSON measurements, joined paired responses, Markdown and an SVG figure.

Repeat the same training command to resume the latest complete verified
checkpoint; up to seven updates after that checkpoint may replay. Adapter weights
alone are not enough to resume training: optimizer, scheduler, RNG and trainer
state must also verify. Saved step numbers and data order determine the resumed
position. Incomplete checkpoints are skipped and corrupted sealed checkpoints
are rejected.

Repeat an evaluation command to reuse completed items and generate only missing
IDs. Preserve the pinned code/environment. Before restarting an interrupted
notebook command, inspect its saved PID and status: an earlier child process
may still be running. Do not launch another GPU job on top of it.

OOM and nonfinite/integrity failures are recorded and stop execution. They do
not trigger silent precision/batch/length changes. A changed setup requires a
recorded protocol revision; this protocol's locked fields are validated against
what the trainer actually executes.

## Reading the report

Accuracy intervals use Wilson 95%; paired differences use 2,000 paired percentile
bootstrap draws with seed 42. Invalid and capped items remain in the denominator.
Length excludes EOS/padding and includes all generated items. Ratios compare
mean checkpoint length with mean matched baseline length.

Strict-only score changes can be formatting effects. Flexible accuracy and
response length are distinct measurements. The paired intervals condition on
one training seed and the selected cohort; they do not establish reliability
across training seeds. Gold/generated length similarity does not establish
imitation as the causal mechanism.

## Primary implementation references

The training boundary follows [TRL 0.26.2 SFTTrainer](https://huggingface.co/docs/trl/v0.26.2/en/sft_trainer),
[Transformers 4.57.6 Trainer](https://huggingface.co/docs/transformers/v4.57.6/en/main_classes/trainer),
and [PEFT 0.18 LoRA](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora).
Precomputed labels are passed through a dedicated collator to preserve the
audited mask. The explicit token-summed loss is normalized across the full
accumulation window. Optional TRL entropy diagnostics are omitted to reduce
full-vocabulary temporary tensors; the SFT objective is unchanged.

## L4 variant and speed benchmark

The current notebook defaults to L4, plan `pilot2-sft-fp32-l4-batch2-v1`, and
matched evaluation batch 2. Select L4 in Colab. Section 6a profiles batches
1/2/4/8 on eight fixed prompt-length ranks with the original generation cap;
inspect tokens/sec, peak allocated memory, projected run costs and any output
differences before starting section 7. This notebook has not been GPU-tested
locally. T4 plans remain supported by the CLI and keep their original meaning.

```bash
python -m pilot_eval sft-prepare --source-config artifacts/plans/baseline-batch8-v1/gsm8k-0shot.config.json --name pilot2-sft-fp32-l4-batch2-v1 --hardware L4 --evaluation-batch-size 2
# Run the usual preflight for this new config before benchmarking.
python -m pilot_eval sft-benchmark --config artifacts/plans/pilot2-sft-fp32-l4-batch2-v1/sft.config.json
```

Changing hardware clears Colab local state, not saved Drive artifacts. Reinstall
and restart as instructed. Reuse frozen cohort identities; do not mix partial
T4 baseline responses into the new L4 scientific run. Check that the old child
process stopped before switching runtime. The benchmark does not prove speed
on every checkpoint, and never silently changes the frozen evaluation batch.

For the independent one-command tester and choosing a larger new plan, see
[inference profiling](inference-profiling.md). It can run after installation,
before training preparation or GPU preflight.

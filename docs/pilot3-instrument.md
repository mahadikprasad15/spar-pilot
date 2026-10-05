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

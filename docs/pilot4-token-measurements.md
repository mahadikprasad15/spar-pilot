# Pilot 4 Ticket 8: supplemental coefficients and KL

This is an additive forward-only product for existing SFT and GRPO adapters at
steps 0, 8, 16, 32 and 64. It preserves all previous training, responses, write
summaries, calibration and report files. The original fixed 300 sequences
(150 held-out GSM8K with complete gold solutions, 150 FineWeb passages), token
IDs, masks, base-model hash and source batch membership must match between arms.
No model training, new generated responses, retokenization or resampling occurs.

## Measurements

- Raw signed coefficient `c=A x`, from the adapted pass's actual module input,
  for all counted content tokens in all 196 modules. Raw c at checkpoint zero
  can be nonzero; zero B makes the adapter contribution zero.
- Factor A/B hashes, norms, module ordering, `alpha/r` and checkpoint identity.
  The branch is `(alpha/r) B c`. Reciprocal factor rescaling changes c while
  preserving the branch. Interpret coefficients with their saved output factor.
- Full-vocabulary KL(tuned || untuned), in nats, for paired fixed prefixes.
  Logits at t-1 predict target t. The next target's existing content view decides
  membership. Exclude target 0, padding, and predictions beyond saved text.
  A format-token context predicting the first solution token counts as solution.
- Per-token values, per-example KL sums/counts, token and equal-example KL means,
  and coefficient signed/absolute/squared sums plus reconstructed branch-norm
  sums by shard/view/module. No confidence or damage threshold is inferred.

FP32 weights, coefficients, hidden states and logits; FP64 log-softmax,
KL accumulation and summaries. Natural logarithms yield nats. The engine
preserves small signed arithmetic residuals and rejects a negative KL beyond
an analytic FP64 roundoff bound. Identical zero-checkpoint distributions must
produce exact zero. The independent CPU known-answer cases test KL direction.

## Public workflow

Use the same configurable artifact root as the previous pilots. Supply actual
saved activation **execution** configs (not reports or training configs):

```bash
python -m pilot_eval tokens-prepare \
  --sft-execution artifacts/plans/SFT_EXECUTION/activation.execution.json \
  --grpo-execution artifacts/plans/GRPO_EXECUTION/activation.execution.json \
  --name pilot4-supplement-v1 --context-chunk 32 --workspace-mib 512 \
  --output-root artifacts
python -m pilot_eval tokens-profile \
  --config artifacts/plans/pilot4-supplement-v1/tokens.prepared.json \
  --output-root artifacts
# Inspect saved profile results, raw diagnostic arrays, total cost, peak memory,
# agreement checks and projected array bytes before recording review notes.
python -m pilot_eval tokens-freeze \
  --config artifacts/plans/pilot4-supplement-v1/tokens.prepared.json \
  --review-notes 'Describe the reviewed supplemental checks and workload' \
  --output-root artifacts
python -m pilot_eval tokens-measure \
  --config artifacts/plans/pilot4-supplement-v1/tokens.execution.json \
  --output-root artifacts
python -m pilot_eval tokens-verify \
  --config artifacts/plans/pilot4-supplement-v1/tokens.execution.json \
  --output-root artifacts
# After verification the GPU may be released. This stage is CPU-only.
python -m pilot_eval tokens-report \
  --config artifacts/plans/pilot4-supplement-v1/tokens.execution.json \
  --name pilot4-supplement-report-v1 --output-root artifacts
```

32 contexts / 512 MiB is an exposed engineering starting setting, not a measured
optimal choice. The batch size is inherited from matching previously frozen
execution configs. If it does not fit, create passing matching source executions
with a smaller batch and a separately named supplemental product; don't edit
old files or mix settings within a run.

## Gates and cost

The existing execution verification requires each arm's source, instrument,
numerical calibration and batch profile to be valid. Ticket 8 adds new checks;
it does not inherit a successful output-head/coefficients check just from FP32.
Adapter file hashes are checked against frozen checkpoint evidence before and
after measurement, and source evidence is reverified before profile/production
completion publication. A shape-valid replacement cannot pass as the old checkpoint.
Its profile compares repeat, singleton vs batched, and half-sized context chunks
at all checkpoints, with a long-example capacity group and a separate group.
Batch 1 still covers both corpora. Agreement uses the existing fixed
`atol=rtol=1e-5` policy for FP32 computation comparisons, **not** a recalibrated
or widened old acceptance rule. The old calibration retains its original scope.
Raw supplemental diagnostic arrays, positions and factors are sealed separately.

One model is loaded at a time. Chunk variants use the same instrument instance;
SFT and GRPO arm loads are sequential. The engine caches only the untuned final
hidden state for a batch and checks disabled-adapter invariance after each
checkpoint switch. Output head matmuls are restricted to the context chunk
**before** vocabulary tensors exist. Stable normalization covers the entire
vocabulary for each context; vocabulary subsets are not independently normalized.

The conservative output-tensor workspace estimate is
`chunk * (96 * vocabulary_size + 8 * hidden_size)` bytes. An over-budget request
fails before allocating output logits. This is a bound on declared output-head
working tensors, not the whole decoder, CUDA allocator or model memory. The
profile records real total allocated/reserved GPU peaks and end-to-end wall
cost, including source verification, loading, diagnostic checks, hashes and
raw-evidence saving. Forward projections and array-size projections are
explicit approximations; full production lengths, I/O and recovery can differ.

## Artifacts and recovery

Plans: `plans/<name>/tokens.prepared.json`, `tokens.execution.json`.

Run: `runs/pilot-4/<model>/supplemental-tokens/<name>/`:

- `meta/run_manifest.json`, `status.json`, `runtime.json`, `timing.json`;
- `profile/`: checks, raw diagnostic arrays, module/token metadata and seal;
- `shards/<arm>/batch-<index>-step-<step>/`: `arrays.npz` (FP32 coefficients,
  FP64 KL), `metadata.json`, numerical/integrity evidence and completion seal;
- `checkpoints/progress.json`, `logs/errors.jsonl`, `results/results.json`;
- top-level completion seal binds the full required shard inventory.

Shard coefficient rows map to `metadata.coefficient_positions`; columns map
exactly to `metadata.modules`. KL rows map to `metadata.predictions`, including
context and next-target positions/IDs/views. Local example indices map to
`metadata.example_ids`. This is a compact numerical equivalent of per-token
JSONL, rather than a JSON line for each of 196 repeated coefficients.

Hash frozen weights at workload boundaries, with cheap in-process guards during
passes. For each batch, save pending checkpoint payloads and publish their seals
only after the final base hash succeeds. Interrupted/unsealed payloads are not
trusted and are recomputed. Already sealed sibling units are verified and reused.
An interruption can redo up to five checkpoint passes for the unfinished batch;
partial progress counters never claim pending units are verified. Completed
runs are CPU-verified and reused. Changed sources/runtime/code/resource settings
require a new named product; no silent fallback or old-file overwrite occurs.

Report: `reports/<report-name>/results/results.json` and `report.md`, with sealed
raw-shard index, coverage, per-example KL and scalar summaries. CPU verification
rejects missing, changed, nonfinite, wrong-position or incompatible artifacts.

## Evidence limits

Software tests use tiny locally constructed Qwen/PEFT models without downloads,
hand-worked distributions and the actual durable workflow around controlled
model dependencies. They do not establish that a real Colab/Qwen run passed.

KL is conditional on supplied fixed prefixes, not an unconditional policy
comparison. Positive KL does not indicate correctness, forgetting or damage.
Token weighting gives longer examples more influence; equal-example weighting
does not. Coefficient sign/scale is factorization-dependent. Internal and
probability changes should be read with separately measured behaviour; these
measurements alone do not identify a causal mechanism.

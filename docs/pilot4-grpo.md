# Pilot 4 implementation and source preparation

The approved protocol and nine tickets live in the Pilot 4 local issue tracker.
Scientific training is not available yet. Tickets 01–02 implement source
preparation/audit and untuned sampling. Preparation is CPU-only; sampling
loads the pinned untuned FP32 model on one GPU.

## Existing sources

Use the completed SFT config and Pilot 3 prepared-input config from the same
artifact root. Defaults in the guided notebook match the previously used
Drive plans. Preparation verifies saved source checkpoints and frozen-base
evidence, the original 512 training and 150 evaluation records, actual SFT
optimizer prompt order, and the 300 fixed GSM8K/FineWeb token/mask records.

The source's real saved optimizer order is separate from its sorted selected
indices. Both are recorded. A changed order, input, checkpoint or source
config invalidates reuse; sources are not rewritten or regenerated.

## Public commands

```bash
python -m pilot_eval grpo-prepare \
  --sft-config <root>/plans/<sft-plan>/sft.config.json \
  --measurement-config <root>/plans/<measurement-plan>/activation.prepared.json \
  --name <new-grpo-plan> --output-root <root>

python -m pilot_eval grpo-audit \
  --config <root>/plans/<new-grpo-plan>/grpo.prepared.json --output-root <root>

python -m pilot_eval grpo-check-ready \
  --config <root>/plans/<new-grpo-plan>/grpo.prepared.json --output-root <root>
```

Preparation publishes the immutable prepared config, matching audit, source
identities, status/progress and a marker written last. A repeat verifies and
reuses completed evidence. Invalid sources record a failure before publishing
a completion marker.

The readiness check currently exits nonzero with the unresolved settings.
That is expected: preparation is not a frozen protocol or preregistration.
Even manually filling placeholders cannot authorize scientific execution.

FP32 is approved for both arms; the initial BF16 source text is superseded.
Historical accuracy 0.640 is not a correctness gate. Runtime tensor dtype,
zero-adapter equivalence and target-GPU checks belong to the later preflight.

## Guided Colab notebook

Use `notebooks/pilot-4-colab.ipynb`. It currently mounts the shared Drive root,
selects the pinned code and source plans, prepares the audit and displays it, then accepts an explicit sampling
settings file and runs/resumes the baseline.
Its code pin must be pushed before a remote Colab checkout can fetch it. The
notebook clearly marks remaining stages as pending; do not interpret prepared
inputs as completed training or instrument validation.

## Tests and remaining work

Offline tests exercise the public commands and notebook preparation stages
using complete source fixtures. They do not prove the real Drive files are
present or that FP32 GRPO fits the selected GPU.

Next are tiny-model GRPO controls and the real GPU preflight. Their dependencies
are the verified preparation, not a new cohort or precision interview. Pending
scientific values remain explicit stage gates until supplied and frozen.

## Untuned training-cohort sampling baseline

```bash
python -m pilot_eval grpo-baseline \
  --config <root>/plans/<grpo-plan>/grpo.prepared.json \
  --settings <explicit-sampling-settings.json> \
  --name <new-sampling-name> --output-root <root>
```

Settings require exactly `subset_ids` (128 distinct saved training IDs, in
explicit order), `scorer` (`gsm8k-flexible-v3`), `seed`, `groups_per_batch`,
`max_new_tokens`, `temperature` (1.0), `top_p`, and `top_k`. There is no implicit
subset, cap, filter or batching choice. A batch contains eight completions per
group. A settings file confirms the scorer for this stage, not the later full
training protocol. The preparation placeholders remain unchanged.

The 128 × 8 baseline is distinct from held-out behavioural evaluation and the
deferred X1 full-cohort sampling baseline. The model loads once; generation
filters, seeds and group/draw order are frozen. Existing verified batches are
reused. An interrupted unsealed batch is redone with its original seed; source,
code/scorer or runtime mismatches reject mixed reuse. Use a new run name for a
longer-cap extension.

Artifacts are under `runs/pilot-4/<model>/sampling-baseline/<name>/`:
`config.json`, `meta/runtime.json`, status/progress, sealed `batches/`, and
`results/responses.jsonl` plus `results/results.json`. Every draw includes the
prompt, gold answer, token IDs, text, stop/cap metadata, strict/flexible score
and reward. Real HF draws also retain tokens through EOS, excluding batch pad.
Token counts exclude EOS and padding.

Capped completions receive zero reward, even if an answer is present. The
report exposes cap counts/fraction, length percentiles, invalid counts,
per-group correct counts and dead groups (all zero or all one rewards). If
censored draws could occupy the top one percent, p99 cannot justify the final
cap. No final cap is automatically chosen. Uncensored p99 still does not
predict all future training lengths; monitor policy is frozen later.

The baseline workflow and notebook command are tested with controlled CPU
generation. This does not establish real-model speed, memory fit or GPU
correctness; those require the later target-GPU preflight.


## Offline implementation validation (ticket 03)

Run with the isolated pinned model-library environment (CPU is sufficient):

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pilot_eval grpo-controls \
  --name tiny-grpo-controls-v3 --output-root artifacts
```

Notebook section 8 runs this same public command. No scientific model is
loaded and no Hub model download occurs. The command executes independent
loss/adapter-gradient and microbatch checks, then the two-seed positive and
negative learning controls. Expected: `passed: true`, algorithm evidence and
four passing control records. It saves configs, trainer arguments, dependency
versions, all sampled toy responses, step logs, initial/final adapters, frozen
base hashes and a completion marker under
`runs/diagnostics/pilot-4/tiny-grpo-controls-v3/`. Reruns verify sealed hashes;
changed implementation/config requires a new name and corrupted evidence
stops reuse. Failed runs remain failed without a completion marker.

Version 3 compares late reward with the untouched policy's exact expectation,
not a window that already contains learning. The earlier failed protocols
remain preserved; see ADR 0010. Passing is implementation evidence only.
GPU preflight and a resolved scientific protocol/preregistration remain gates
before full GRPO training. These changes are local until explicitly pushed.

## GPU preflight and reviewed protocol (ticket 04)

Notebook sections 9–12 expose the explicit settings, public preflight command,
evidence review and protocol freeze. See the
[conceptual companion](pilot4-preflight-concepts.md) for diagrams and rationale.

```sh
python -m pilot_eval grpo-preflight \
  --config <root>/plans/<prepared>/grpo.prepared.json \
  --baseline-config <root>/plans/<baseline>/baseline.config.json \
  --controls-name <verified-controls> --settings <explicit-settings.json> \
  --name <preflight-name> --output-root <root>
python -m pilot_eval grpo-freeze \
  --config <root>/plans/<preflight-name>/grpo.preflight.json \
  --review <user-review.json> --name <frozen-name> --output-root <root>
python -m pilot_eval grpo-check-ready \
  --config <root>/plans/<frozen-name>/grpo.frozen.json --output-root <root>
```

The settings explicitly confirm flexible-v3/DAPO and specify a positive token
margin above ceil(baseline p99), generation/backward group batches, evaluation
batch, gradient checkpointing, diagnostic seed and minimum free GPU memory.
Sampling filters must match the baseline. There are no silent scientific
defaults. A censored p99 needs a named baseline extension first.

The real backend loads the pinned FP32 Qwen model once, checks 196 intended
projections and zero initialization, compares untuned/zero-adapter greedy
tokens on all 150 items, then runs two identical disposable two-step trials.
Losses, gradients, draws and final state must repeat exactly; base hashes must
stay unchanged. A forced full-cap probe with 64 continuations from the longest
eight training prompts measures generation/backward memory. Artificial rewards
in this probe exercise gradients; it is not scientific training.

Preflight evidence lives in `runs/pilot-4/<model>/preflight/<name>/`, with
runtime/config, zero outputs, two trials (responses, adapter, optimizer/RNG
state), synthetic capacity responses, timings and sealed completion metadata.
Verified diagnostic units are reused after interruption. An incomplete unit
is redone. This differs from scientific training recovery at sealed checkpoint
boundaries only. A failed gate never becomes a passing result through review.

Timings cover diagnostic hash/transfer/save overhead. Reports project training,
greedy/sampled behavioral evaluation and worst checkpoint redo costs; later
activation measurement/KL costs remain unmeasured. Projections depend on future
response lengths and are not runtime guarantees.

The review requires an explicit non-inferiority margin, cap/dead-group/length
monitor thresholds and actions (`flag`, `pause`, `stop`), notes and a user-authored
timezone-aware dated prediction/falsifier. Freeze verifies source/evidence hashes
and persists these choices plus ordered groups and runtime settings. It does not
start scientific training. Changes require a named new variant.

Implementation tests exercise the public commands with controlled GPU evidence
and the real generation/training backend with a tiny locally constructed Qwen
model on CPU. They do not establish actual Qwen GPU capacity, performance or
Drive access. Run the notebook preflight on the intended GPU before freezing.

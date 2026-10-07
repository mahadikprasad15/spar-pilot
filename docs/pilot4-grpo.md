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

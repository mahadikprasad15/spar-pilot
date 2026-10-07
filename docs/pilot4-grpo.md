# Pilot 4 implementation and source preparation

The approved protocol and nine tickets live in the Pilot 4 local issue tracker.
Scientific training is not available yet: ticket 01 implements source
preparation/audit only. No GPU, model download, tokenizer or dataset access is
needed for this stage.

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
selects the pinned code and source plans, prepares the audit and displays it.
Its code pin must be pushed before a remote Colab checkout can fetch it. The
notebook clearly marks remaining stages as pending; do not interpret prepared
inputs as completed training or instrument validation.

## Tests and remaining work

Offline tests exercise the public commands and notebook preparation stages
using complete source fixtures. They do not prove the real Drive files are
present or that FP32 GRPO fits the selected GPU.

Next are baseline sampling and tiny-model GRPO controls. Their dependencies
are the verified preparation, not a new cohort or precision interview. Pending
scientific values remain explicit stage gates until supplied and frozen.

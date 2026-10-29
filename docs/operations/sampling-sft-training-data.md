# Sampling SFT: training export and CPU readiness

Ticket 08 exports reviewed targets and the original gold initialization. It does
not train an arm or authorize a paid job. The CLI is the primary interface;
sections 15–16 of `notebooks/sampling-sft.ipynb` are optional clients.

## Export

Finish and freeze the eligibility review first. This stage does not require the
Ticket 07 statistical report.

```sh
python -m pilot_eval sampling-sft-training-data-prepare --review plans/sampling-review-v1/sampling-sft.review.json --name sampling-training-data-v1 --output-root artifacts
python -m pilot_eval sampling-sft-training-data-export --config plans/sampling-training-data-v1/sampling-sft.training-data.json --output-root artifacts
python -m pilot_eval sampling-sft-training-data-verify --config plans/sampling-training-data-v1/sampling-sft.training-data.json --output-root artifacts
```

Paths to saved plans are relative to the artifact root or absolute. The output
lives under `artifacts/runs/sampling-sft/<source-plan>/training-data/<name>/`:

- `arms/k{0,2,5}/training.jsonl`: stored prompt and generated response IDs, masked
  labels, original step and slot, source and attempt identities.
- `arms/k{0,2,5}/batches.json`: all 64 original eight-slot batches, including each
  omitted problem and its reason. No duplicates or replacement gold answers.
- `arms/k{0,2,5}/coverage.json`: uncovered identities and coverage disposition.
- `initialization/checkpoint-0/`: exact gold adapter configuration and tensors.
- `results/export.json`: T15, optimizer, rank/alpha, checkpoint and data contracts.
- `export.complete.json`: content hashes used to verify restored exports.

Every selected response must end with exactly one gold-matching supervised
end-turn ID. An alternative sampler stop cannot be silently converted. A prompt
mismatch, extra stop or unfinished response blocks export. No retokenization,
truncation or text repair occurs. All 64 steps must remain nonempty; more than
five uncovered problems blocks the handoff. Exporting k2 data does not select it
for training.

T15 compares all 196 rank-one A/B pairs and the adapter payload byte for byte
against gold checkpoint 0, with exact zero B. It certifies the exported payload;
Ticket 09 must also check the initialized real model against that payload.

## Check short batches

Use an existing CPU interpreter with the pinned trainer stack. This constructs
a tiny random Qwen model locally; it downloads no model and uses no GPU.

```sh
python -m pilot_eval sampling-sft-training-data-check --config plans/sampling-training-data-v1/sampling-sft.training-data.json --mode full --python /absolute/path/to/pinned/python --output-root artifacts
python -m pilot_eval sampling-sft-training-data-status --config plans/sampling-training-data-v1/sampling-sft.training-data.json --output-root artifacts
```

T16 runs one optimizer step for each batch size 3–7 through the real SFTTrainer,
using the production model initializer, collator and loss. Accumulation stays at
8 to exercise a final partial step. The loss denominator is the actual total of
supervised shifted tokens, including end-turn tokens, across that step.

An independent Python FP64 log-sum-exp calculation must agree with both the
reported trainer loss and summed microbatch losses within
`abs(error) <= 1e-6 + 1e-6 * abs(reference)` (approved 2026-10-29). Finite gradients,
exactly one optimizer/scheduler step, correct token denominators and ordered
examples are required. Frozen base weights must remain unchanged.

Each attempt retains its request, environment versions, measurements, console,
readiness result and hash seal in `checks/<attempt>/`. Missing dependencies,
wrong versions, a broken import or a failed gate cannot become a pass. Status
verifies the sealed evidence before displaying readiness. No environment install
or threshold adjustment is automatic.

`--mode kernel` is a weaker diagnostic: it exercises the production loss and
collator with real CPU autograd and optimizer, but does not run SFTTrainer.
Even a passing kernel receipt leaves `t16_passed=false` and
`training_ready=false`. It cannot substitute for T16.

## Training boundary

Readiness is separate from authorization. Full training still requires Ticket
07's explicit k0/k5 selection, the budgeted-job approval and Ticket 09's real L4
FP32/runtime/config checks. A failed short-batch path stops for repair and never
triggers duplication. Full optimizer resume state is not part of this export.

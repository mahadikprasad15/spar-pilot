# Pilot 2 — rank-1 SFT

## Question

How do correctness, response format and generated length change during a small
rank-1 SFT intervention on GSM8K gold solutions?

## Design and configuration

| Field | Executed recipe / variant distinction |
|---|---|
| Model | Pinned `Qwen/Qwen2.5-1.5B-Instruct`, inherited evaluation identities |
| Training cohort | 512 distinct GSM8K train problems, seed 42 |
| Supervision | Completion-only gold solutions, including end-turn; no packing/truncation |
| Adapter | Rank 1, alpha 1, dropout 0; q/k/v/o/gate/up/down in all 28 blocks |
| Trainable scope | 196 LoRA modules; original model parameters frozen |
| Precision | Unquantized FP32 |
| Optimizer | AdamW; LR 1e-4; weight decay 0; gradient clipping 1.0 |
| Schedule | 64 updates; constant LR; no warmup; microbatch 1, accumulation 8 |
| Checkpoints | 0, 8, 16, 32, 64 |
| Evaluation | Same 150 held-out GSM8K items; matched FP32 untuned baseline; greedy |
| Hardware/batching | Separate T4/L4 run variants; saved run config overrides current notebook defaults |

Completion loss is token-summed and normalized over the accumulation window.
Rank and alpha constrain adapter capacity/scaling; they are separate from the
optimizer learning rate.

## Decisions

The run is exploratory. Historical accuracy/length-collapse numbers are not
binary success thresholds. A fresh FP32 untuned baseline was evaluated rather
than mixing the earlier precision variant into the training trajectory.

Flexible-v3 was introduced after the original scores exposed a strict-valid,
flexible-invalid extraction bug. The new audit uses saved responses, preserves
old scores and is explicitly post hoc.

## Results and execution status

The preserved flexible-v3 audit reports the following on 150 held-out items:

| Checkpoint | Flexible-v3 accuracy | Mean response tokens |
|---|---:|---:|
| Untuned / 0 | 0.6800 | 215.53 |
| 8 | 0.7133 | 221.48 |
| 16 | 0.7067 | 189.46 |
| 32 | 0.6200 | 128.04 |
| 64 | 0.6600 | 137.78 |

Source: the preserved `sft-behaviour-audit.json` in the
`pilot4-results-so-far-2026-10-13` local report snapshot; see
[artifact inventory](../artifacts.md). Values are descriptive; intervals and
paired records remain in the source audit. That audit verifies exported
responses/legacy scores but explicitly does not independently reverify every
original training/runtime artifact.

## Outputs

Tokenized training examples and supervised labels; matched baseline and
checkpoint responses; adapter and full recovery states; runtime/trainer
arguments; loss/gradient history; paired score/length tables and rescoring audit.

## Limitations

One training seed. Cohort bootstrap intervals do not establish cross-seed
reliability. The response-length reduction does not identify its cause or imply
that every capability deteriorated. Scorer revision and source-audit limits
must accompany the reported trajectory.

## References

- [Specification](../../.scratch/pilot2-sft/spec.md)
- [Execution guide](../pilot2-sft.md)
- [L4 variant decision](../adr/0004-pilot2-l4-throughput.md)
- [Flexible-v3 decision](../adr/0005-gsm8k-flexible-v3.md)
- [Notebook](../../notebooks/pilot-2-colab.ipynb)
- Implementation: `sft.py`, `sft_backend.py`, `sft_evaluation.py`, `sft_reporting.py`, `sft_score_audit.py`.

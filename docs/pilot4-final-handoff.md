# Pilot 4 final comparison and Colab handoff

Ticket 9 consumes verified saved evidence; it runs no training, generation or
model loading. Source runs and reports remain read-only. The guided notebook
will publish a separately named final report and hash-indexed download inventory.

## Report inputs

A JSON selection identifies five canonical sources under the artifact root:
`grpo_evaluation` (execution config), `sft_writes`, `grpo_writes`,
`control_writes` (sealed report directories), and `tokens_execution` (the
Ticket 8 frozen execution config). SFT behavioural source runs are derived
from the verified GRPO matching contract, avoiding an unrelated exported
response file masquerading as a verified training source.

## Reading comparisons

- Compare greedy correctness and generated length on the same 150 item IDs.
  Keep strict and flexible-v3 results separate; v3 scoring of legacy SFT text
  is a disclosed offline correction. GRPO sampled accuracy averages eight
  observed draws within each item, then items; it is not pass-at-eight.
- Internal profiles use the existing fixed question/solution/FineWeb inputs,
  views and both token/equal-example weighting conventions. Signed mean-vector
  cosines require resolved nonzero directions and compatible source identity.
- KL(tuned || untuned) measures supplied next-token contexts, not unconditional
  policy distance. Equal optimizer steps, equal write norm and equal KL are
  different progress comparisons. An early SFT checkpoint is a preview, not a
  lower-learning-rate run.
- SFT gold completions and GRPO sampled completions differ. Even when cohort,
  FP32 precision, adapter budget and optimizer settings match, do not call the
  comparison objective-only causal evidence.
- Missing SFT gradient consistency remains unavailable. One norm-matched random
  adapter is a reference intervention, not a null distribution, percentile or
  empirical model-write noise floor. Its realized activation magnitude must
  be reported separately from its weight-norm match.

## Recovery and release

Training recovers only at sealed checkpoints 0/8/16/32/64. Restore optimizer,
scheduler, RNG, data order and preceding gradient. Preserve incomplete attempts
as diagnostics; exclude them from the accepted history. Up to 32 optimizer
steps may be redone. Evaluation resumes at sealed item/draw batches; activation
and supplemental measurements resume at their frozen atomic batch units.

Disconnect the GPU only after scientific stages have exited successfully and
required raw evidence has passed CPU verification. Reporting and downloads need
no model loading. A notebook being saved does not guarantee its artifacts were
saved; keep the canonical artifact root on mounted Drive.

## Software versus scientific completion

Offline controlled workflow tests establish software and artifact contracts.
The actual Qwen/source/CUDA gates still must pass in Colab before interpreting
scientific measurements. Null or inconclusive outcomes can be complete; corrupt,
incomplete or incompatible required evidence cannot produce a complete report.

## Commands

```bash
python -m pilot_eval grpo-final-report --selection plans/final/selection.json --name final --output-root artifacts
python -m pilot_eval grpo-final-verify --selection plans/final/selection.json --name final --output-root artifacts
python -m pilot_eval grpo-final-export --selection plans/final/selection.json --name final --archive-name final-download-v1 --output-root artifacts
```

Use actual root-relative paths in the selection. Final outputs live under
`reports/<name>/`: `config.json`, `complete.json`, `results/results.json`,
`results/report.md`, standalone SVG figures, SFT paired responses/bootstrap
indices, and `results/download-inventory.json`. Three separately named
subreports retain behavioural, write and token details. Earlier write-comparison
reports use v1; the new checkpoint-direction product uses v2 and must use a new
name rather than replace v1.

The exporter saves `exports/<archive-name>/bundle.tar`, `manifest.json`,
`config.json`, and `complete.json`. It streams files, verifies archived payload
hashes, and reuses a completed verified bundle. An uncompressed bundle needs
additional Drive space approximately equal to the inventory size. Download via
Drive web UI; notebook variables and local Colab checkout are temporary.

## Plot contract

- `behaviour.svg`: greedy accuracy/length and paired checkpoint-minus-zero changes
  with 95% item-bootstrap intervals.
- `sampled-endpoints.svg`: GRPO sampled mean correctness/length at 0 and 64;
  SFT sampled measurements are explicitly unavailable.
- `write-depth.svg`: block relative writes by view, weighting and checkpoint;
  existing per-arm intervals and denominators are retained in structured sources.
- `direction.svg`: signed SFT–GRPO cosines at all checkpoints, unresolved gaps.
- `kl.svg`: both fixed-context KL weightings, no invented uncertainty intervals.
- Six module figures: step-64 projection facets for all views and both weights,
  including the random reference's realized activation magnitudes.

All formulas/populations/limits and a concrete cosine/weighting example are in
`report.md`. Full numerical tables stay in `results.json`; the source-report
CSV/NPZ products are part of the download inventory. An inventory describes
source artifacts; the exporter includes both those and the final report itself.

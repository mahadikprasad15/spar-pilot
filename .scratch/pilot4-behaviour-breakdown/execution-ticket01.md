# Ticket 01 execution evidence

Executed 2026-10-22 using saved Drive outputs. No model inference, model download,
GPU work or changes to original files. The importer/scorer commands and notebook
are implemented locally; this release has not been pushed.

## Sources

Local imported bytes and historical receipts:
`artifacts/imports/pilot4-behaviour-breakdown-2026-10-22/`.
The selection is `selection.json`; the separate verification record is
`source-verification.json`.

- GRPO: `pilot4-grpo-fp32-v1-behaviour-v1` native config and responses.
  Response SHA-256:
  `4af7926a3def742d9eef128f3b7306b783aada413f785624054dacb16ee971ec`.
  Config, responses and result summary match the historical `complete.json`
  file inventory. Its runtime file was not imported or independently checked.
- SFT: native `pilot2-sft-fp32-l4-batch8-v1-step-0` and `step-64`.
  Step-0 response SHA-256:
  `b809596e4d77fdbf4654044ad9c3c7af9bd4e3bb7ba73cbb37a7e524006a6c5a`.
  Step-64 response SHA-256:
  `e2be48d2799abf0e065820c9bff9cc61f9dd0d66c6cf576d0a3a19281879577f`.
  Response hashes match native result summaries. Config hashes are recorded on
  import; no GRPO-style native completion seal is claimed for SFT.

These checks establish exported-byte consistency, not independent historical
training or checkpoint verification. Raw files remain local/ignored artifacts.

## Execution and results

Plan: `artifacts/plans/pilot4-behaviour-breakdown-2026-10-22-v1/behaviour.analysis.json`.

Run: `artifacts/runs/pilot-4/behaviour-breakdown/pilot4-behaviour-breakdown-2026-10-22-v1/`.

Public `behaviour-prepare`, `behaviour-score`, and `behaviour-verify` completed.
Status is completed, 600/600 responses, with source snapshots, four scoring units,
raw/harmonized records, readable report and verified completion inventory.

| Method / checkpoint | N | Strict correct | Flexible-v3 correct | Last-number correct |
|---|---:|---:|---:|---:|
| SFT / 0 | 150 | 8 | 102 | 107 |
| SFT / 64 | 150 | 97 | 99 | 100 |
| GRPO / 0 | 150 | 8 | 102 | 107 |
| GRPO / 64 | 150 | 75 | 114 | 117 |

All 150 IDs match across endpoints. No prompt/gold conflicts. Declared greedy
input/decoding contracts match, including the 1,024-token cap. The step-0 exported
content and declared settings match; method-specific records remain retained.
The last-number diagnostic is not a human-corrected accuracy estimate.

This stage does not identify which questions improved, assess sampled reliability,
label intended answers, or establish reasoning quality. Those are later tickets.

## Verification

- Session baseline, before implementation: 394 passed, 35 skipped, 4 warnings.
- New public-workflow file: 32 passed.
- Affected regressions: 45 passed in 28.52 seconds, including the new file,
  offline rescoring, SFT score audit, and notebook session tests.
- New notebook stages were executed in a fresh subprocess with Torch,
  Transformers, Datasets, PEFT, Accelerate and Hugging Face Hub imports blocked.
  A subsequent fresh process restored settings and displayed the verified report.
- No complete post-change suite or live Colab notebook run is claimed.

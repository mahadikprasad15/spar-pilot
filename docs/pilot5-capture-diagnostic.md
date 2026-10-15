# Pilot 5 bounded capture diagnostic

## Run sequence

1. Fresh CPU runtime, updated shared notebook: **1–2 → 14**. Reuses source selections and corpus. No model loads.
2. Fresh GPU runtime: **1–2 → 4 → 15 → 16**. Skip prior audit, smoke, corpus and vocabulary cells. Cell 4 installs/checks the pinned environment; cell 15 starts the actual diagnostic. Cell 16 is CPU verification and can run immediately in the same runtime.
3. Send the final JSON summary and any error evidence. After the child exits successfully and 16 verifies, GPU resources can be released. No production run starts automatically.

Ticket 03 reference review is independent and still required before learned readouts. Do not Run all.

## Declared workload

Four frozen documents: first two in source order from legacy-150 and new-50 respectively. Each has 128 content positions. Base, SFT step-0 and GRPO step-0 produce **12 singleton sequence forwards/model calls**, each capturing embedding output and blocks 0–27 together. Sites are before final model norm. One pinned FP32 model is loaded and both saved adapters attached. A fixed numeric check applies the actual final RMS norm and first 32 output-matrix rows to a base pooled block-14 vector: one auxiliary norm/head calculation, not another sequence forward. It displays no decoded token list.

Zero adapters must cover all 196 projection modules each, have rank one/scaling one, finite parameters and exactly zero B weights. Their output differences must be exactly zero; this mathematical identity is separate from pending approximate GPU numerical gates.

The base is hashed before work and after each new condition before sealing. These reads are timed and printed; no full optimizer/trajectory audit is repeated. Loading, per-condition capture, evidence processing and array saving are recorded. The first actual Colab duration is unknown: installation/model download, device throughput and Drive throughput can dominate this small workload. A cheap diagnostic is not grounds for claiming production timing.

## Saved objects and recovery

- Plan: `artifacts/plans/<diagnostic-name>/traces.capture-diagnostic.json`.
- Run: `artifacts/runs/pilot-5/<diagnostic-name>/capture-diagnostic/`.
- `meta/run_manifest.json`, runtime and explicit lifecycle status.
- `units/base/`: full FP32 named-site activation arrays, FP16 conversion errors and numeric norm/head evidence.
- `units/sft-zero/`, `units/grpo-zero/`: FP64 differences and exact comparison coordinates/values/allowances.
- Each unit has a hash seal; interrupted unsealed work is repeated. A fresh process reuses sealed conditions only under the same runtime/source/implementation identity.
- `results/base-moments.npz`: FP64 block-14 Σh and Σhhᵀ; the JSON count is **508** (4 × 127), excluding content position zero. Full production would be 25,400.
- `results/results.json`: unique workload plus reused/newly sealed counts, timings, scope and pending review.
- `logs/errors.jsonl`: condition, error and diagnostic coordinates when available. Underlying numerical evidence is retained on failure; failed conditions are not sealed complete.

At width 1,536 and 128 input positions, the uncompressed base arrays are about 87 MiB. Each full FP64 zero-difference unit is about 174 MiB; arrays are currently saved uncompressed for simple exact readback. The 18 MiB second-moment matrix adds to this, for roughly 453 MiB across the bounded run before metadata; actual input lengths and measured bytes govern. This is diagnostic evidence, not the production retention format.

Completed runs route to the CPU verifier before GPU probing. Changed runtime or implementation does not silently reuse numerical evidence. If a new diagnostic is needed, change its name in cell 14 and prepare it once; the selected name persists for cells 15–16. Earlier experiments and adapters are preserved.

## What it establishes

CPU tests establish math and orchestration on independent fixtures. The real diagnostic can establish bounded zero-adapter identity and produce runtime-specific numerical evidence. It does not qualify trained nonzero deltas, all token rankings, full-cohort statistics, every hardware/batch configuration or production storage precision. The recorded completion state is `awaiting-review`, with `qualified: false`.

The approved improved design is carried through: A7 individual positions 0–4 and whole-document pooling, A6 base moments at positions 1–127, and explicit storage sensitivity. Pure CPU `storage_sensitivity` reconstructs primary vectors after conversion and for identical seeded whole-document bootstrap draws. It returns numerical/rank error summaries without decoded learned lists and without inventing acceptance limits. Real nonzero sensitivity/profile work is declared separately before production in Ticket 06.

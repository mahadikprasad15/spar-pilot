# 05: Units, manifests, export and restore

**Status:** resolved (2026-11-13) — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** none

**What to build:** this study's unit and export layer over `BlockStore` and `neuron_audit_export`.

- [ ] Units: A+B per (arm, batch); C per (arm, condition); D per (arm, condition, batch). Sealed with IDs and input, plan, code and runtime hashes; durable status; duplicate refusal; hash-verified resume.
- [ ] Parameterized artifact root; per-unit config, logs, per-example results, partial manifest.
- [ ] Cumulative hard-cap enforcement at unit boundaries, including export time, with elapsed budget persisted across restarts; time projections advisory only (longest-first fixture must not stop a healthy run).
- [ ] Seal every output type of this study in an export-supported layout (BlockStore units for arrays; sealed run directories with hashed manifests for the launch record, freezes, trace/response JSONL, per-unit results, summaries, receipts); extend `neuron_audit_export.snapshot` for them with a declared expected set.
- [ ] Partial and complete export with online upload and fresh readback; clean-root restore of a complete run and of a failed partial run, each bringing back the morning summary and the missing-work list. An incomplete scope is labelled partial.
- [ ] `pilot_eval/attribution_gpu.py`: NVML sampler, synchronized throughput, padding fraction, MFU estimate under the stated FLOP model, profiler trace hook.
- [ ] Tests: a fixture with only summary files is exported and restored (it is omitted by today's snapshot); interrupted and corrupted units; duplicates; cap reached mid-stage; partial export lists missing units; restore into an empty root.

## Comments

## Answer

2026-11-13 — `pilot_eval/attribution_units.py`, `pilot_eval/attribution_gpu.py`, a backward-compatible `expected_runs` argument to `neuron_audit_export.snapshot`. Tests: `tests/test_attribution_units.py` (16, CPU, mocked Hub); the existing export tests still pass.

- **Units** go in one `BlockStore` at `<root>/attribution/store`. Keys are `ab-<arm>-b<NNNNN>`, `c-<arm>-<condition>` and `d-<arm>-<condition>-b<NNNNN>`. A unit's identity holds its kind, arm, batch or condition, example IDs and input/plan/code/runtime hashes. Per-unit config, log and per-example results go in the sealed index.
  - Resume reuses a sealed unit only when its identity matches, after hash verification. A different identity is refused, and so is an example already sealed in another unit of the same (kind, arm, condition).
  - Interrupted units are reported as `interrupted` and corrupted ones as `corrupt`. `Units.manifest(expected)` gives sealed, interrupted, corrupt, missing and the missing-work list.
- **Everything that is not an array** (launch record, freezes, trace and response JSONL, summaries, receipts) goes in a sealed run directory written by `seal_run`: files, then `complete.json` with their SHA-256s, the layout `snapshot` already archives. A sealed directory is never rewritten with different content. The morning summary is `attribution/summary/` (`run_summary.md`, `run_summary.json`, `missing_work.json`), written by `write_summary`.
- **Snapshot contract.** `snapshot(..., expected_runs=[...])` reports `missing_expected_runs`, and the snapshot is `partial` unless every expected unit and run directory is sealed. `export()` runs snapshot then upload with fresh readback. `restore_summary()` does a clean-root restore and returns the proof, summary and missing-work list. Both a summary-only fixture and a failed partial run restore into an empty root.
- **`Budget`.** Per-stage and total hard caps in seconds, checked at `boundary()` between units. `charge()` adds export time. Elapsed time is persisted, so a restart continues the remaining cap, and a different cap set is refused. Caps must be fixed numbers (null refused). `projection()` is advisory only; a longest-first fixture runs to completion.
- **`attribution_gpu`.** 1 Hz NVML sampler (unavailable on CPU is recorded, not fatal; a summary needs ≥ 30 s of samples), `synchronized` timer, throughput, padding fraction, MFU estimate under the stated FLOP model (2N forward + 2N per activation-only backward; excludes attention scores and padding), Chrome profiler trace. Targets are reported as non-gates.
- The online upload path is the existing `OnlineHubProcess` client. Its real use with this study's outputs is exercised in 06a.

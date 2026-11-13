# 05: Units, manifests, export and restore

**Status:** ready-for-agent — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
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

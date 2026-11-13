# 05: Units, manifests, export and restore

**Status:** needs-triage — draft v2
**Type:** task
**Blocked by:** none

**What to build:** this study's unit and export layer over `BlockStore` and `neuron_audit_export`.

- [ ] Units: A+B per (arm, batch); C per (arm, condition); D per (arm, condition, batch). Sealed with IDs and input, plan, code and runtime hashes; durable status; duplicate refusal; hash-verified resume.
- [ ] Parameterized artifact root; per-unit config, logs, per-example results, partial manifest.
- [ ] Cumulative cap enforcement at unit boundaries, including export time; separate caps per tier.
- [ ] Partial and complete export with online upload and fresh readback; clean-root restore.
- [ ] `pilot_eval/attribution_gpu.py`: NVML sampler, synchronized throughput, padding fraction, MFU estimate under the stated FLOP model, profiler trace hook.
- [ ] Tests: interrupted and corrupted units; duplicates; cap reached mid-stage; partial export lists missing units; restore into an empty root.

## Comments

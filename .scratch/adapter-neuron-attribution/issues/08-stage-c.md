# 08: Stage C on confirmation

**Status:** needs-triage — draft v3
**Type:** task
**Blocked by:** 07

- [ ] Up to 19 conditions per arm and metric from the freeze; per-example ΔL, ΔS, KL.
- [ ] Retained effect R inputs (full removal of the top `down_proj` adapter always present).
- [ ] Per (arm, condition) units; utilization summary; verified export.

## Comments

### 2026-11-13 — review round 3 (Codex), adopted

- `stage0_drift_backend.DriftEngine` is a pattern only: its KL is variant ‖ untuned with an FP64 head. Stage C uses KL(full arm ‖ condition), FP32, with a same-arm unablated reference.
- An unavailable matched group is a recorded skipped condition, never an error that aborts the chain.

# 04: Run resumable multi-arm audit with measured head option

**Status:** needs-triage — draft for adversarial review

**What to build:** After successful GPU qualification and once partial export is available, run the frozen full Phase 1A collection with progress, durable bounded work units and validated resume. Include only the previously measured and approved head scope.

**Blocked by:** 03b, 07.

- [ ] Use FP32 weights/forwards and reductions with TF32 and autocast off; verify each process's interpreter, packages, actual cache/offline load, runtime and frozen batch/mask hashes before work.
- [ ] Execute identical example order, membership and padding in every arm; show arm, unit, completed/total, timestamp and stale-progress warning. An SSH disconnect is not a failed job.
- [ ] Write and seal resumable immutable dense blocks and per-token read data; reject changed sources, corrupt units or unsafe concurrent writers. Revalidate sources once per session unless their immutable contract is proven; do not repeatedly rehash history.
- [ ] Enforce one Pod, one tmux session and one writer via operator runbook and advisory PID/host/heartbeat marker; stale or ambiguous marker requires investigation. Do not describe it as a reliable distributed lock. Validate output/temp storage operations and preserve failure evidence. Omitted or subset head coverage is explicit.

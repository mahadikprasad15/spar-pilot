# 14: Pod chain integration and real-path CPU test

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 08, 09
**Group:** F

Run the two arms through the existing `c_attribution_chain.Chain` with this run's stage list: preflight (IEEE readback, `check_interpreter`, `storage_probe`, disk check against D24's total), train, evaluate, seal, export, teardown. No new runner.

- [ ] Red/green: spec test 20 (real path on a saved tiny Qwen2, only CUDA, volume, Hub and stop stubbed).
- [ ] HF cache on the container disk; every download hashed.

## Comments

## Answer

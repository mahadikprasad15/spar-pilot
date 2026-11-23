# 14: Pod chain integration and real-path CPU test

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 08, 09
**Group:** F

Run the two arms through the existing `c_attribution_chain.Chain` with this run's stage list: preflight (IEEE readback, `check_interpreter`, `storage_probe`, disk check against D24's total), train, evaluate, seal, export, teardown. No new runner. Build on `c_attribution_backend.TransformersBackend` for precision (`set_ieee_precision`, readback, refusal), pad IDs, release and memory; never `sampling_sft_training_backend.open_engine` (legacy TF32 flags).

- [ ] Red/green: spec test 20 (real path on a saved tiny Qwen2, only CUDA, volume, Hub and stop stubbed).
- [ ] HF cache on the container disk; every download hashed.
- [ ] Red/green: spec test 24 (no legacy TF32 setter; IEEE readback required).

## Comments

## Answer

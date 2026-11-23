# 14: Pod chain integration and real-path CPU test

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 08, 09
**Group:** F

Run the two arms through the existing `c_attribution_chain.Chain` with this run's stage list: preflight (`TransformersBackend.preflight`: IEEE readback, the record's `disk` block sized from D24, `HF_HOME`, offline worker, `volume_identity`; `runtime` fingerprint), train, evaluate, seal, export, teardown. No new runner. One training unit per arm: on a relaunch the chain keeps verified completed arms and restarts an interrupted arm from step 0 (D24). Build on `c_attribution_backend.TransformersBackend` for precision (`set_ieee_precision`, readback, refusal), pad IDs, release and memory; never `sampling_sft_training_backend.open_engine` (legacy TF32 flags).

- [ ] Red/green: spec test 20 (real path on a saved tiny Qwen2, only CUDA, volume, Hub and stop stubbed).
- [ ] HF cache on the container disk; every download hashed.
- [ ] Red/green: spec test 24 (no legacy TF32 setter; IEEE readback required).

## Comments

## Answer

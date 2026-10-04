# 01: Freeze and audit measurement inputs

**What to build:** Prepare verified Pilot 2 checkpoint sources and fixed GSM8K/FineWeb token sequences, then expose an inspectable saved input audit without loading model weights.

**Blocked by:** None (can start immediately).

**Status:** claimed

- [ ] Verify all five source checkpoint manifests and frozen-base hash evidence; resolve relocated files without modifying historical configs.
- [ ] Preserve the 150 source GSM8K IDs, complete questions/gold solutions and actual chat template; pin FineWeb configuration/revision and freeze the approved bounded sample.
- [ ] Save token IDs, tokenizer-aware question/solution/user masks, exclusions and source hashes; reject silent truncation or insufficient eligible inputs.
- [ ] Repeated preparation verifies and reuses immutable artifacts; source/input/config corruption or mismatch stops with clear diagnostics.
- [ ] Public commands produce an audit showing rendered inputs, counted positions, lengths and provenance.
- [ ] Offline CPU workflow tests cover actual saved outputs, independent mask expectations and invalid-source behavior.

## Comments

- 2026-10-04: Breakdown approved; implementation authorized. Companion spec and design define scientific contracts.

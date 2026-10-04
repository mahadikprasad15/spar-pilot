# 01: Freeze and audit measurement inputs

**What to build:** Prepare verified Pilot 2 checkpoint sources and fixed GSM8K/FineWeb token sequences, then expose an inspectable saved input audit without loading model weights.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Verify all five source checkpoint manifests and frozen-base hash evidence; resolve relocated files without modifying historical configs.
- [x] Preserve the 150 source GSM8K IDs, complete questions/gold solutions and actual chat template; pin FineWeb configuration/revision and freeze the approved bounded sample.
- [x] Save token IDs, tokenizer-aware question/solution/user masks, exclusions and source hashes; reject silent truncation or insufficient eligible inputs.
- [x] Repeated preparation verifies and reuses immutable artifacts; source/input/config corruption or mismatch stops with clear diagnostics.
- [x] Public commands produce an audit showing rendered inputs, counted positions, lengths and provenance.
- [x] Offline CPU workflow tests cover actual saved outputs, independent mask expectations and invalid-source behavior.

## Comments

- 2026-10-04: Breakdown approved; implementation authorized. Companion spec and design define scientific contracts.

## Answer

Public activation preparation/audit commands are implemented. Ten new offline
workflow tests pass; full regression suite: 141 passed, five expected PEFT
warnings. Confirmed red/green cycles for missing command, omitted completion
payloads, missing failure status and mistakenly counted control tokens.

Real Drive source files and Qwen/FineWeb downloads were not accessed locally.
Tests exercise verified synthetic source artifacts and controlled tokenizer/data
implementations. Real source and tokenizer preflight remain Colab work; this
ticket does not claim an executed scientific measurement.

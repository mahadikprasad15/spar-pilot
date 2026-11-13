# 03a: Prove and persist one-arm neuron measurements on CPU

**Status:** needs-triage — draft for adversarial review

**What to build:** Using fixed-input fixtures and compatible existing hook seams, implement the actual `down_proj` pre-hook collector, independently validate its PEFT calculations, and persist immutable dense FP32 blocks and indexes for a tiny base/adapted cohort. This is the reusable artifact contract, not GPU qualification.

**Blocked by:** 01, 02.

- [ ] Reconstruct actual z from the active gate/up paths; compare an independent PEFT A-output capture to aᵀz and adapter-branch output to s*b*c_t. Correctly cover signed cancellation, gauge transformations and optional head slice reconstruction.
- [ ] Inject an off-by-one mask, wrong layer and wrong sign in test fixtures and show the corresponding independent check fails. No model downloads.
- [ ] Test padded, varying-length examples, response/reply-start/termination masks, weighted reductions and exact base/adapted pairing; distinguish absent adapter fields from zero.
- [ ] Persist immutable per-arm/group/layer/statistic/row-block FP32 arrays with stable row indices, counts, hashes and token-level c_t offsets; verify reread and unit corruption handling.
- [ ] Independently check saved/live key maps and active routing in fixtures, including wrong keys and legitimate zero weights; no claim of a qualified real GPU.
- [ ] Implement CPU fixtures for the proposed 1e-6 layer-max z rule and 1e-5 sum-of-absolute-terms read/branch rules, including zero-scale and cancellation cases. These remain proposed until the candidate plan is approved.
- [ ] Test the pure activity-matched null on a small exactly enumerable fixture and seeded Monte Carlo draws; check sampling without replacement and expected discrete coverage. Test the production selection/report integration again in 05/06.

## Comments

### 2026-11-10 — first implementation (Claude)

- Env: the base miniforge torch is broken (`libtorch_cpu.dylib` missing). An isolated `.venv/` (gitignored) holds torch 2.9.0, transformers 4.57.6 and peft 0.18.0, per `requirements-pilot2.txt`. Run with `.venv/bin/python -m pytest`.
- `pilot_eval/neuron_audit_collector.py`: FP32 `down_proj` pre-hook collector built on `ActivationEngine` conventions (padding/position IDs, scoped hooks, autocast off, TF32 off, `disable_adapter()` for untuned, frozen-base hash). Dimensions come from the model config. Independent checks: z vs act(gate)*up, both hooked separately; PEFT `lora_A` output vs a.z with **saved** a; PEFT `lora_B` output × **live** scale vs s·b·c with saved b; output−base_layer as a non-gating diagnostic; masked reductions recomputed per example. Also routing call counts, `check_live_adapter` (saved vs live keys, shape, dtype, bitwise, merged/disabled/active/scale; zero-B classified), `adapter_effect` (active vs disabled logits; disabled vs independent base, tolerance left null) and `collect_cohort`.
- `pilot_eval/neuron_audit_blocks.py`: sealed units per (arm, row range). Logical [rows, J] FP32 blocks per (group, layer, stat), packed in one file with an index, plus per-token c_t/row/position/group arrays. Zero-count rows must be NaN. Sealed units are never rewritten; interrupted attempts are moved aside as evidence; a corrupt unit raises `CorruptUnit`; hashes are checked once per process.
- `pilot_eval/neuron_audit_math.py`: tolerance checks (proposed 1e-6 layer-max, 1e-5 sum-|terms|; zero reference = exact or unresolved), gauge helpers, rank ties by index, matched null (seeded, without replacement within bins, `Unavailable` when underfilled), exact convolution plus brute-force enumeration, and the reference interval (labelled not a CI).
- Tests: 30 (`test_neuron_audit_{math,blocks,collector}.py`). Wrong sign, swapped layer and wrong scale each fail the independent checks; a shifted mask and a wrong layer fail the pure checks. A test caught a real bug: the branch check initially used the saved scale on both sides.
- Stored stats: mean z and mean |z| (FP64 accumulation, FP32 storage); adapted arms add mean a_j z_j, mean |a_j z_j| and c_t. Token-weighted means are derivable from the stored counts.
- **Not done:** the optional `o_proj` per-head slice (Phase 1B); it is deferred until 03b decides head scope. Tolerances are still the candidate plan's proposals. Nothing here qualifies a real GPU.

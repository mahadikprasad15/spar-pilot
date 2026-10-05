# 03: Profile and freeze production batching

**What to build:** Measure workload-specific throughput, memory and numerical agreement, then freeze a reviewed passing batch plan for production.

**Blocked by:** 02 — Validate and measure one batch.

**Status:** resolved

- [x] Profile batches 1/2/4/8/16 including the longest inputs and actual reference cache; save synchronized timing/memory/evidence.
- [x] Compare summaries to batch 1 at the accepted tolerance; identity/count/undefined coverage must match.
- [x] Benchmark OOM makes a candidate unsuitable; other failures stop. No GPU-family whitelist or generation-derived timing estimate.
- [x] Freeze ordered membership and numerical/runtime identity only for a reviewed passing candidate; preserve prepared inputs.
- [x] Offline workflow tests verify failed-candidate handling and rejection of unsupported evidence at freeze.


## Answer

Implemented public `activation-profile` and `activation-freeze` commands. The
fixed workload contains the longest eight complete inputs per corpus; every
candidate measures the same 16 IDs, all views and all five checkpoints. Real
reference caches, warmup, CUDA synchronization, allocated/reserved peaks and
separate validation timing are included. Generation timing is not reused.

Profiles verify immutable preparation and instrument evidence, persist per-example
NPZ summaries and rank-1 evidence, compare sufficient sums plus derived token/
example-weighted measurements, and require exact counts and undefined coverage.
OOM candidates are saved as unsuitable; other failures stop with logs/status.
Completed candidate artifacts are hash-verified and reused after interruption.

Explicit review freezes a separate immutable execution manifest: all 300 IDs in
ordered batches, input/profile/diagnostic hashes, runtime, FP32/FP64 settings and
unchanged acceptance thresholds. Prepared input files are preserved. Freeze can
run on CPU, but production must verify the saved runtime again.

Six new offline workflow tests cover happy-path/reuse, CLI review and immutable
settings, corruption, OOM continuation/refusal, other-error stopping and ratios
whose small denominators magnify differences. Tiny real Qwen/PEFT tests also
compare individually measured examples with padded batch summaries and verify
separate timing. No model download or real GPU measurement was performed.

Usage and timing scope: `docs/pilot3-instrument.md`. 2026-10-05 full regression: **158 passed**, with 13 existing PEFT tiny-model
and PyTorch TF32 API warnings. `git diff --check` also passed. Real Drive sources, CUDA timing/memory and
full-cohort execution remain runtime checks; production belongs to Ticket 04.

# 03: Profile and freeze production batching

**What to build:** Measure workload-specific throughput, memory and numerical agreement, then freeze a reviewed passing batch plan for production.

**Blocked by:** 02 — Validate and measure one batch.

**Status:** claimed

- [ ] Profile batches 1/2/4/8/16 including the longest inputs and actual reference cache; save synchronized timing/memory/evidence.
- [ ] Compare summaries to batch 1 at the accepted tolerance; identity/count/undefined coverage must match.
- [ ] Benchmark OOM makes a candidate unsuitable; other failures stop. No GPU-family whitelist or generation-derived timing estimate.
- [ ] Freeze ordered membership and numerical/runtime identity only for a reviewed passing candidate; preserve prepared inputs.
- [ ] Offline workflow tests verify failed-candidate handling and rejection of unsupported evidence at freeze.

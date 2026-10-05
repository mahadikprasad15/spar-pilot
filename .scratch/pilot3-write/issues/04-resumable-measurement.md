# 04: Measure full cohorts with safe resume

**What to build:** Process every frozen batch/checkpoint combination while preserving validated progress across interruption and reconnection.

**Blocked by:** 03 — Profile and freeze production batching.

**Status:** claimed

- [ ] Persist baseline/checkpoint summaries with hashes and completion markers last; reuse only verified combinations.
- [ ] Recreate temporary baseline activations for a partial batch and verify saved baseline summaries before continuing.
- [ ] Enforce production-batch instrument gates, bounded cache, no duplicate writers and final frozen-base identity checks.
- [ ] Interrupted/resumed output matches uninterrupted output without double counting; corrupt completed shards and incompatible identities stop.
- [ ] Production OOM/nonfinite/gate failure preserves progress without silent batch/precision/length changes.
- [ ] Monitoring tolerates transient status reads without confusing them with computation success or failure.

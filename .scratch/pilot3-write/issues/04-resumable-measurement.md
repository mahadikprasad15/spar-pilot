# 04: Measure full cohorts with safe resume

**What to build:** Process every frozen batch/checkpoint combination while preserving validated progress across interruption and reconnection.

**Blocked by:** 03 — Profile and freeze production batching.

**Status:** resolved

- [x] Persist baseline/checkpoint summaries with hashes and completion markers last; reuse only verified combinations.
- [x] Recreate temporary baseline activations for a partial batch and verify saved baseline summaries before continuing.
- [x] Enforce production-batch instrument gates, bounded cache, no duplicate writers and final frozen-base identity checks.
- [x] Interrupted/resumed output matches uninterrupted output without double counting; corrupt completed shards and incompatible identities stop.
- [x] Production OOM/nonfinite/gate failure preserves progress without silent batch/precision/length changes.
- [x] Monitoring tolerates transient status reads without confusing them with computation success or failure.


## Answer

Implemented `activation-measure` and CPU-only `activation-verify`, with a public
read-only frozen-execution verifier and subprocess progress observer. Production
uses all 300 prepared examples and the reviewed, ordered batch memberships.

Each batch saves one untuned block-summary shard and five separate checkpoint
shards. Numeric NPZ payloads are FP64 and metadata binds IDs, token/mask input
hashes, source/config identity, axes and shapes. Markers are written last. Resume
verifies expected combinations and skips completed work; a partial batch must
recreate matching baseline summaries before its missing checkpoints proceed.
Unmarked owned payloads are replaced; marked corruption stops without regeneration.

Production checks exact step-0 zeros, frozen-base/checkpoint identities, canonical
hooks, rank-1 sampled-token coverage and the spec's unchanged thresholds. Cache
release happens after each batch and on errors. OOM/nonfinite/gate failures retain
verified progress and log batch/checkpoint diagnostics, without fallback. Run locks
prevent a conflicting caller from changing the owner's progress. Preparation and
execution must have distinct names to preserve input-run metadata.

The completed run saves per-example shards, verified aggregate sufficient sums,
structured completion results and final payload hashes. CPU verification requires
every configured combination, passing evidence and matching totals. Progress lists
verified unit IDs; monitoring uses the actual child process and tolerates transient
mutable-status reads, which never establish scientific completion.

Thirteen new CPU workflow tests cover full execution/reuse, exact resumed versus
uninterrupted arrays, partial leftovers, public commands, failure gates, cache
release, OOM recovery, baseline mismatch, corrupt/missing marked evidence,
runtime/batch incompatibility, conflicting writers, progress reads and naming.
Existing tiny real Qwen/PEFT tests now exercise reference sufficient sums and cache
release too. Actual source checkpoints/CUDA execution remain Colab runtime gates.

Usage and artifact layout: `docs/pilot3-instrument.md`. 2026-10-05 full regression: **171 passed**, with 13 existing PEFT tiny-model
and PyTorch TF32 API warnings. `git diff --check` passed. Scientific reporting/intervals/plots remain Ticket
05; the guided notebook remains Ticket 06.

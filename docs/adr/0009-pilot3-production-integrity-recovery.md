# ADR 0009: Production batch integrity and reviewed code-only recovery

Status: accepted by user instruction to fix stopped production measurement,
2026-10-06. Real Colab recovery remains pending.

## Diagnosis

Profiling used bounded exact hashes but production still called two whole-base
hashes per checkpoint unit. At batch 8, 300 examples and five checkpoints create
190 units and 380 repeated hashes. Earlier T4 hash timings were approximately
29 seconds; no full production speedup is inferred from that timing alone.

## Decision

Use the existing integrity scope around each production batch. Preserve exact
SHA256 before and after the batch, parameter identity/storage/shape/dtype/device
and version guards inside, disabled-reference invariance, rank-1 checks, all
numerical thresholds and source hashes. Buffer the pending checkpoint summaries
for one batch in CPU memory; publish its new baseline/checkpoint payloads only
after the final exact hash passes. A boundary failure publishes no new units.
Interrupted work before that final check repeats the pending units in that batch.
Previously marked units remain unchanged and are skipped.

For a fresh 300-example/batch-8 run this uses 76 batch-boundary hashes, plus
startup/final checks. It does not remove forward passes, hook reductions or disk
verification. No numerical agreement tolerance is changed.

## Provenance and recovery

The frozen execution, profile and calibration manifests remain unchanged.
`activation-recover-measurement` verifies their chain and every existing marked
shard before writing `meta/production-recovery.json` in the same run directory.
The recovery records original/actual runtime, unchanged execution hash, review
notes, integrity/publication policy, completed units and preserved-file hashes.
Only the Git revision may differ; hardware, numerical settings and package
versions must match. The user reviews this narrow implementation change; this
record is not general authorization for arbitrary future code changes.

The actual runtime must match the recovery target on resume. Final completion
includes the recovery record in its hashed inventory. CPU verification checks
both preserved historical evidence and the final numeric aggregates. Old units
were protected with the previous, more frequent hashes; new units carry workload
integrity evidence. The summaries and scientific settings are identical.

Progress now shows batch/checkpoint/hash stages. The completed-unit count can
stay unchanged while an entire batch is buffered, then advance after integrity
passes. No train/evaluation/calibration/profile rerun is required for this scoped
recovery. A numerical runtime change remains forbidden.

## Tests and limits

CPU production-interface tests reject failed final hashes before publication,
exercise interruption/resume, compare preserved bytes, reject numerical-setting
changes and verify the final artifacts. Exact hash/version mutation detection is
also tested at the engine interface. Fixtures do not establish real GPU throughput.

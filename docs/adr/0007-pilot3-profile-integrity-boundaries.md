# Pilot 3 profiling integrity boundaries

Status: accepted for profiling performance recovery, 2026-10-06.

## Evidence

The user's T4 profiler spent 83 minutes without completing batch-size candidate
1. Process evidence showed approximately one CPU core busy, GPU utilization 0%
at the observation, and interruption inside frozen-base SHA256 byte hashing.
This identifies repeated hashing as a concrete cost; it does not provide a
complete wall-time attribution or a GPU speedup estimate.

## Decision

Keep exact source-compatible SHA256 before and after each bounded profiling
warmup/timed workload. Use parameter identity/storage/shape/dtype/device/version
guards at checkpoint and measurement boundaries within that workload. Publish
candidate summaries only after the final exact hash succeeds. Preserve all
rank-1, invariance, zero, finite, hook, batch-agreement and file-integrity gates.
This implements spec.md's before/after workflow hash contract without moving
its acceptance thresholds. Scientific measurement and standalone validation
continue their existing per-measurement full hashes.

Record the profiling policy in the immutable identity and boundary hash time
in saved evidence. Show chunk/checkpoint/hash stages in the notebook monitor.
Archive the failed zero-candidate attempt and previous validation evidence,
verify preservation hashes, then revalidate under the new code identity.

## Alternatives and limits

Keeping two full SHA256 passes per checkpoint/example caused excessive host
transfer/hash work. Using version counters alone could miss persistent `.data`
edits; the final exact hash is therefore mandatory. Keeping a full duplicate
GPU base for equality would require about another 6 GB and reduce T4 headroom.
The guards do not defend against adversarial `.data` edit-and-restore between
checks. This controlled inference instrument performs no optimizer updates.

CPU tests verify fewer hash calls, mutation rejection, unchanged summaries,
publication ordering, and recovery preservation/refusal. Actual T4 throughput
remains to be measured in Colab. No claimed elapsed-time speedup is inferred
from the hash-call reduction alone.

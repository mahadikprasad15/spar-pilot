# Sampling SFT private HF archive with external download verification

Status: accepted, user selection 2026-10-30.

Use a RunPod network volume for active state and the private HF dataset
`Prasadmahadik/spar-sampling-sft-artifacts` for durable sealed exports. Preserve
canonical artifact-relative paths, exact commit revisions and file hashes.
Credentials remain outside experiment configs/logs. Transfer only declared,
sealed job payloads; preserve prior exports and reject conflicting restores.

HF archive success is additional evidence, not a substitute for the existing
network-volume and independent downloaded-copy receipts. An external controller
restores the pinned HF revision and verifies the actual files before issuing its
receipt. HF transports this acknowledgement back to the worker. Only then may
the existing runner request its own Pod stop. Provider confirmation and billing
remain distinct. Same-host synthetic probes cannot qualify real shutdown.

Failures retain evidence and require operator attention within the existing
budget; no GPU provisioning, deletion, budget increase or scientific scope
expansion is authorized by adding this transport. Full later analyses remain
separate consumers of the frozen sample/adapters bundle.


Resilience follow-up approved2026-10-30: retry only transient acknowledgement
transport errors with bounded backoff and persisted events. Start the ack clock
after upload, cap it by remaining total allocation, and interrupt stalled HF calls
at that bound. Forecast allowance from compressed/unpacked size and measured
rates; preflight disk capacity on artifact/cache/destination filesystems. Existing
independent-copy, no-delete and hard-budget policies remain unchanged.

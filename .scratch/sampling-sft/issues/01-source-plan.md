# 01: Audit existing sources and prepare a restorable experiment plan

**What to build:** A user supplies the saved gold-SFT/GRPO identities and obtains an immutable source inventory and resolved Sampling SFT plan, restored in a fresh CPU session. It verifies the512 training IDs, model/tokenizer, gold order/config/checkpoints/A0 and original L4 training evidence. Missing facts produce explicit blocked status instead of invented defaults. Gold remains unchanged.

**Blocked by:** None.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Prepare, verify and restore the same immutable plan through public CPU commands and the incremental notebook, without CUDA or model downloads in tests.
- [x] Verify the exact 512 training identities, pinned model/tokenizer, gold prompt/masks, original training order/configuration and checkpoint0 payload; preserve original sources.
- [x] Preflight requires evidence of original L4 training, not merely an L4 evaluation manifest. Missing or conflicting facts produce explicit blocked status.
- [x] Resolve k0/k2/k5, data seed0, training seed42, L40S sampling, L4 FP32 training, J64, cap1024, checkpoint0/8/16/32/64 and all decoding defaults without silent fallback.
- [x] Bind source, plan, algorithm/template and runtime identities; source changes reject reuse and completed artifacts are not overwritten.
- [x] Distinguish source-audited, software-tested, real-qualified and executed status. Record deferred baseline/offset, evaluation and internals; launch no paid job.

## Testing and evidence

Public-command fixtures reject changed identities and evaluation-versus-training provenance confusion; notebook restores and displays evidence without CUDA. No GPU qualification is claimed.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

2026-10-28: Claimed after user requested Ticket01 implementation. Scope is CPU source/plan/notebook only; no paid execution.

## Answer

Implemented the public CPU prepare/audit/status/verify workflow and explicit inspected-owner recovery, with one readable source notebook and an operations guide. Reused existing cohort/configuration and rank-one payload checks. Source manifests bind raw file hashes, missing-file observations, model/tokenizer pins, exact gold batches, prompt masks/chat-template identity, approved proposer text/hash and all five original gold adapters. A blocked audit is a valid saved diagnostic; downstream consumers must inspect source_ready.

The source plan resolves approved recipe controls. Engine-specific defaults, scheduling, worker numeric settings and actual stop IDs are explicitly unresolved and block future engine execution; source_ready never implies engine-ready. No values are silently inherited or fabricated on CPU. These are Ticket03 qualification inputs, not a completed runtime_hash.

Validation: red tests first demonstrated absent public commands; new recovery/schema/template cases likewise failed before their implementation. Final focused regression:47 passed in114.26s across the Sampling SFT workflow and existing Stage0/SFT source checks. Includes a fresh-process no-model-import check and execution of every notebook CPU cell. Notebook JSON schema and Python syntax validated. Canonical synthetic demonstration retained under artifacts/validation/sampling-sft-ticket01; it is explicitly not real Drive/model evidence.

Actual source readiness remains unexecuted against the full Drive tree. No real decoder/tokenizer was loaded, no paid job launched, no training or GPU qualification occurred, and nothing was pushed. The earlier source-audit fixture is not a scientific result.

A full baseline regression was started as required, then stopped as a duplicate after253 passed/24 skipped (no failures; explicit KeyboardInterrupt). The inherited post-change full regression later completed669 passed/39 skipped/4 warnings; it collected before Ticket02 tests existed and is not a commit-isolated validation of later changes. The local Torch import limitation remains; none of the47 focused checks skipped.

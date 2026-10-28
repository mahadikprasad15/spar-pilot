# 01: Audit existing sources and prepare a restorable experiment plan

**What to build:** A user supplies the saved gold-SFT/GRPO identities and obtains an immutable source inventory and resolved Sampling SFT plan, restored in a fresh CPU session. It verifies the512 training IDs, model/tokenizer, gold order/config/checkpoints/A0 and original L4 training evidence. Missing facts produce explicit blocked status instead of invented defaults. Gold remains unchanged.

**Blocked by:** None.

**Status:** ready-for-agent

**Type:** task

## Acceptance criteria

- [ ] Prepare, verify and restore the same immutable plan through public CPU commands and the incremental notebook, without CUDA or model downloads in tests.
- [ ] Verify the exact 512 training identities, pinned model/tokenizer, gold prompt/masks, original training order/configuration and checkpoint0 payload; preserve original sources.
- [ ] Preflight requires evidence of original L4 training, not merely an L4 evaluation manifest. Missing or conflicting facts produce explicit blocked status.
- [ ] Resolve k0/k2/k5, data seed0, training seed42, L40S sampling, L4 FP32 training, J64, cap1024, checkpoint0/8/16/32/64 and all decoding defaults without silent fallback.
- [ ] Bind source, plan, algorithm/template and runtime identities; source changes reject reuse and completed artifacts are not overwritten.
- [ ] Distinguish source-audited, software-tested, real-qualified and executed status. Record deferred baseline/offset, evaluation and internals; launch no paid job.

## Testing and evidence

Public-command fixtures reject changed identities and evaluation-versus-training provenance confusion; notebook restores and displays evidence without CUDA. No GPU qualification is claimed.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

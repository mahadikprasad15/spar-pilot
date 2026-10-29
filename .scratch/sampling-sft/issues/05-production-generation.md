# 05: Collect the three full sampling arms with reconciled attempts

**What to build:** The user launches approved k0/k2/k5 generation for512 problems under one frozen runtime. It resumes sealed work, retains raw attempts/events, separates same-request engine replays from up-to3 model retries, and saves a512-row final-disposition product per arm with progress and budget snapshots. No temperature1 baseline runs.

**Blocked by:** 04.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Generate k0/k2/k5 on all512 problems under one frozen runtime/algorithm identity with data seed0 and declared deterministic request/retry seeds.
- [x] Require real pilot replay/score qualification and all approval/budget prerequisites before paid Stage2; never launch the deferred temperature1 baseline.
- [x] Resume verified sealed blocks only; changed source/runtime/configuration and competing writers reject continuation.
- [x] Retain initial and up-to3 model-outcome retries, stop at first eligible answer, and keep same-seed engine replays separate from model retries.
- [x] Each terminal trace attempt has exactly one of four end states. Preserve failures and interruptions in operational events; do not count disconnects as model outcomes.
- [x] Publish512 final dispositions per completed arm, each covered row referencing exactly one eligible finished attempt; reconcile attempts, dispositions, tokens and progress. Engine-error-blocked runs cannot be called complete.
- [x] Persist raw responses/IDs/proposal decisions, score/finish metadata, rejected attempts, source hashes and cost snapshots; offline fixtures demonstrate interruption and all outcomes.

## Testing and evidence

Offline end-to-end fixtures reconcile all attempt states, retries, covered/uncovered identities, interrupted requests and completion. Real Stage2 remains blocked until the pilot's actual qualification and budget prerequisites pass.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Answer

Implemented CLI-first full k0/k2/k5 collection on the saved512 training problems. The generation plan binds the original source IDs/prompts, one runtime, algorithm/code identities, data seed0 and retry seeds1001/1002/1003. Stage2 freeze requires real pilot/profile/T3 evidence, exact-plan/report approval and current budget headroom; every actual invocation still requires its active reserved paid job. CPU fixtures cannot satisfy those gates. No deferred temperature1 baseline, evaluation, internals or training is launched.

Per-arm writers own contiguous, hash-chained attempt rounds and completed blocks. Raw proposals/IDs/scores/decisions, response texts, terminal states, RNG state, worker evidence, operational events/timings and budget snapshots are retained. Resume abandons only partial scientific blocks, preserves earlier seals and skips finished responses. Completed states finalize without another model load. Same-seed engine replay is separate from up-to3 model retries; a second engine/protocol failure blocks completion and preserves failed attempt diagnostics.

Each completed arm has a reconciled attempt table and exactly512 covered/uncovered dispositions. Covered rows select the first automatically eligible finished attempt; uncovered rows have four exhausted model attempts. Eligibility remains automatic-only with separate human review pending, and training_allowed stays false. Cost snapshots preserve actual/reserved experiment totals rather than inventing final bills. Explicit owner-token recovery and changed-source/runtime/payload rejection are covered.

Validation:98 combined CPU tests passed in169.44s, including three full512-row fixture arms, retries, capped/exhausted/stop-only/error cases, partial-round restoration, sealed-byte preservation, provisional progress, approval binding, decimal budget boundaries and CPU-only finalization. Default optional-notebook CPU cells run in regression. Canonical evidence and relocated verified synthetic arms:artifacts/validation/sampling-sft-ticket05/. These are test fixtures, not research findings.

No real model download/inference, provider call, paid job, rental/deletion or push occurred. Real L40S qualification/generation and paid approvals remain pending. Runbook:docs/operations/sampling-sft-generation.md. Ticket06 is the next implementation slice: separate eligibility/coverage review.

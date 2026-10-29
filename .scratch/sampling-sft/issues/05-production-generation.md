# 05: Collect the three full sampling arms with reconciled attempts

**What to build:** The user launches approved k0/k2/k5 generation for512 problems under one frozen runtime. It resumes sealed work, retains raw attempts/events, separates same-request engine replays from up-to3 model retries, and saves a512-row final-disposition product per arm with progress and budget snapshots. No temperature1 baseline runs.

**Blocked by:** 04.

**Status:** claimed

**Type:** task

## Acceptance criteria

- [ ] Generate k0/k2/k5 on all512 problems under one frozen runtime/algorithm identity with data seed0 and declared deterministic request/retry seeds.
- [ ] Require real pilot replay/score qualification and all approval/budget prerequisites before paid Stage2; never launch the deferred temperature1 baseline.
- [ ] Resume verified sealed blocks only; changed source/runtime/configuration and competing writers reject continuation.
- [ ] Retain initial and up-to3 model-outcome retries, stop at first eligible answer, and keep same-seed engine replays separate from model retries.
- [ ] Each terminal trace attempt has exactly one of four end states. Preserve failures and interruptions in operational events; do not count disconnects as model outcomes.
- [ ] Publish512 final dispositions per completed arm, each covered row referencing exactly one eligible finished attempt; reconcile attempts, dispositions, tokens and progress. Engine-error-blocked runs cannot be called complete.
- [ ] Persist raw responses/IDs/proposal decisions, score/finish metadata, rejected attempts, source hashes and cost snapshots; offline fixtures demonstrate interruption and all outcomes.

## Testing and evidence

Offline end-to-end fixtures reconcile all attempt states, retries, covered/uncovered identities, interrupted requests and completion. Real Stage2 remains blocked until the pilot's actual qualification and budget prerequisites pass.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

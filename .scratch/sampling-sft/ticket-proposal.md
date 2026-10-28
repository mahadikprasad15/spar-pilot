# Sampling SFT — proposed vertical slices

Status: awaiting user approval of granularity and dependency edges
Date: 2026-10-28
Parent: confirmed Sampling SFT specification

All slices use the approved public saved-workflow boundary and the same incremental notebook/runbook. Each includes saved schemas, command behavior, user-visible outputs and offline tests. No broad prefactor or schema-only slice is currently justified. Dependencies describe implementation/product contracts; passing a software ticket does not imply its real GPU qualification or paid execution was performed.

## 01 — Audit existing sources and prepare a restorable experiment plan

**Blocked by:** None.

**What it delivers:** A user supplies the saved gold-SFT/GRPO identities and obtains an immutable source inventory and resolved Sampling SFT plan, restored in a fresh CPU session. It verifies the512 training IDs, model/tokenizer, gold order/config/checkpoints/A0 and original L4 training evidence. Missing facts produce explicit blocked status instead of invented defaults. Gold remains unchanged.

**Verification:** Public-command fixtures reject changed identities and evaluation-versus-training provenance confusion; notebook restores and displays evidence without CUDA. No GPU qualification is claimed.

## 02 — Run budgeted jobs through verified export and Pod shutdown

**Blocked by:** 01.

**What it delivers:** The user can reserve a bounded job, run it through the shared lifecycle, reconcile cost, verify network-volume/download receipts and request own-Pod shutdown. The ledger enforces25 total and10/6/9 sub-caps,80% review, storage lifetime and stage-attributed setup/reruns. A safe fixture/no-op job demonstrates the full workflow without renting compute. Destructive actions remain refused without approval.

**Verification:** Fake-provider workflow tests cover reservation conflicts, cap/80% rejection, interruption uncertainty, failed export, stop-request versus confirmation and deletion guards. Real provider evidence remains a separate authorized check.

## 03 — Execute a bounded likelihood-search trace and qualify engine scores

**Blocked by:** 01, 02.

**What it delivers:** A public command runs the declared k recipe on bounded inputs and saves raw proposals, stored token IDs, scores, acceptance decisions and terminal states. It handles gold-guided prompts, greedy whole-trace regrowth, J64/cap1024, engine replay and exact slicing. Independent HF comparison emits the20-trace score receipt with both approved thresholds. Bounded throughput candidates select and freeze one L40S runtime without automatic retuning later.

**Verification:** Tiny offline engines/math fixtures exercise the complete trace path, strict improvement and all terminal/error cases; a separate real qualification checklist requires worker flags, token alignment and category-complete score fixtures. All real profile/check work requires reservations and first-spend approval.

## 04 — Qualify the32-problem pilot through interruption and exact resume

**Blocked by:** 03.

**What it delivers:** A user runs the fixed k5 pilot and receives sealed block progress, a fresh-process restore path, exact uninterrupted-versus-resumed scientific comparison, throughput/cost estimates and the explicit pilot alarm table. Failed replay or alarms block production; approved configuration changes create new attempts and requalification evidence.

**Verification:** Public-workflow fault injection covers partial-block discard, writer ownership, fixed request groups/order, cache resets, preserved sealed bytes and runtime mismatch. Real kill-after-block2 replay is recorded separately from fixture success.

## 05 — Collect the three full sampling arms with reconciled attempts

**Blocked by:** 04.

**What it delivers:** The user launches approved k0/k2/k5 generation for512 problems under one frozen runtime. It resumes sealed work, retains raw attempts/events, separates same-request engine replays from up-to3 model retries, and saves a512-row final-disposition product per arm with progress and budget snapshots. No temperature1 baseline runs.

**Verification:** Offline end-to-end fixtures reconcile all attempt states, retries, covered/uncovered identities, interrupted requests and completion. Real Stage2 remains blocked until the pilot's actual qualification and budget prerequisites pass.

## 06 — Review eligibility and publish the training coverage decision

**Blocked by:** 05.

**What it delivers:** The user reviews flagged responses through a saved round-trip interface, preserving automatic flexible-v3 decisions and recording confirmed intended-answer overrides separately. It publishes immutable coverage summaries, marked slots, common/union cohorts and pairwise symmetric-difference flags, and blocks>5 uncovered or unfinished required review. It never repairs text or duplicates missing problems.

**Verification:** Saved fixtures cover review provenance and revisions, source-bound labels, cap/exhaustion ineligibility,0/1/5/6 omissions, clustered slots and512-row reconciliation. The user can inspect the workload and partial-review status before labelling.

## 07 — Produce the descriptive gate report and explicit training selection

**Blocked by:** 06.

**What it delivers:** CPU commands restore eligible data and HF score evidence to produce gold/k-arm plain-NLL, length, format, leakage, acceptance and coverage views. The common-cohort paired bootstrap uses approved estimands, reports absent baseline/offset honestly, and creates a source-bound request for training k0/k5. No automatic merge or training launch occurs.

**Verification:** Hand-computable public workflows check stop-inclusive equal-problem NLL, difference-of-medians resampling, undefined short-text leakage, whole-problem pairing, report reconciliation and stale approval rejection. Gate scoring model work is separately budgeted; CPU statistics never require CUDA.

## 08 — Export the gold-matched training dataset and verify short batches

**Blocked by:** 01, 06.

**What it delivers:** The user obtains a pretokenized training bundle using gold's exact per-step order and marked omissions, plus a readiness receipt proving checkpoint0 identity and T16 for batch sizes3–7. This slice independently verifies the training handoff without training a full arm or waiting for statistical-report presentation.

**Verification:** Tiny real CPU-model loss/gradient/scheduler checks, mask/end-turn and stored-ID checks, exact A0 comparison, all short-batch sizes and required metadata. Failure blocks readiness; no duplicate fallback. Full training still requires ticket07's approval product.

## 09 — Train approved endpoint arms with verified checkpoints and recovery

**Blocked by:** 02, 07, 08.

**What it delivers:** After explicit gate and budget approval, the user trains k0/k5 on verified L4 for64 steps, one seed42, with gold-matching configuration/logs and required adapters0/8/16/32/64. The workflow exports verified checkpoints and handles interrupted training honestly: verified complete recovery state permits resume; otherwise a budgeted new attempt restarts from checkpoint0. k2 never launches implicitly.

**Verification:** Public-command tiny CPU training covers step/checkpoint counts, skipped-slot exposures, immutable completed arms, optimizer-state recovery versus adapter-only refusal, hardware/approval/ledger mismatch and export. Real L4 execution evidence remains distinct from software tests.

## 10 — Assemble the adapter handoff and final research-status report

**Blocked by:** 07, 09.

**What it delivers:** One verified manifest assembles source/plan, attempts/coverage/reviews, gate report, training logs/adapters and export receipts. The notebook/runbook presents a readable fresh-session route, exact completion/qualification status, limitations and deferred consumers. It can show truthful partial status while training is pending; it cannot label absent evaluation or internals as completed.

**Verification:** Complete public-workflow fixtures from audit through handoff, stale-parent/hash rejection, report/raw reconciliation, partial versus completed status and notebook release consistency. No paid execution or GitHub push is implied by the final software ticket.

## Dependency review

- 01 is the initial implementation frontier.
- 02 provides controls before real paid engine work; no production job bypasses it.
- 03's bounded trace path precedes04's pilot qualification;04 precedes05's full production.
- 06 needs05's attempt/disposition contract. Its fixture review path can be implemented without waiting for real512-problem collection.
- 07 and08 are independent after06;08 additionally uses01's gold identities. The08 handoff does not require gate plots, but09 requires both07 approval and08 readiness.
- 09 lists02 explicitly because it consumes the budget/lifecycle contract as well as training approval and readiness.
- 10 needs the gate and training product contracts. Its partial-status mode does not require completed real GPU execution to be implemented or tested.
- Every ticket integrates its notebook/runbook route;10 consolidates them rather than introducing the first runnable notebook.

## Approval request

Review granularity, genuine blocking edges, and whether any slice should be merged or split. Once approved, publish one ready-for-agent file per ticket in dependency order. Do not claim or implement tickets merely because this proposal exists.

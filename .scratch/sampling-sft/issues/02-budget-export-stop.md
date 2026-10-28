# 02: Run budgeted jobs through verified export and Pod shutdown

**What to build:** The user can reserve a bounded job, run it through the shared lifecycle, reconcile cost, verify network-volume/download receipts and request own-Pod shutdown. The ledger enforces25 total and10/6/9 sub-caps,80% review, storage lifetime and stage-attributed setup/reruns. A safe fixture/no-op job demonstrates the full workflow without renting compute. Destructive actions remain refused without approval.

**Blocked by:** 01.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Provide a complete reserve → bounded job → cost reconciliation → verified export → own-Pod-stop workflow through public commands and the notebook/runbook using fake providers in CPU tests.
- [x] Enforce USD25 total and sampling10/training6/storage-and-shared-overhead9 sub-caps across outstanding reservations and spent amounts; concurrent or tampered ledger updates cannot bypass limits.
- [x] Reserve GPU hourly rate × estimated hours ×1.5 plus non-duplicated setup cost before launch; reject projected cap crossing and pause for explicit review at80% of any sub-cap or total.
- [x] Count setup, idle, debugging and reruns against the served stage; reserve storage for its expected lifetime including stopped-Pod periods. Shared costs count once.
- [x] Retain uncertain interrupted reservations until reconciled; publish cumulative ledger snapshots without inventing actual billing values.
- [x] Require verified network-volume and downloaded-copy receipts before automatic stop; distinguish stop-requested from externally confirmed stopped. Failed exports do not authorize destructive fallback.
- [x] Refuse network-volume deletion or unexported termination without explicit approval; first paid GPU work needs a post-preflight go-ahead. Demonstrate the lifecycle without real rental.

## Testing and evidence

Fake-provider workflow tests cover reservation conflicts, cap/80% rejection, interruption uncertainty, failed export, stop-request versus confirmation and deletion guards. Real provider evidence remains a separate authorized check.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

2026-10-28: Claimed at user request. CLI/run-script workflow is primary; notebook optional. Fake-provider testing only, no paid job.

## Answer

Implemented a CLI-first dollar ledger and saved job lifecycle, with the same commands available in the optional notebook and a standalone run script. Immutable requests bind the source plan, output inventory and implementation; one shared atomic ledger holds all attempt reservations. Integer-microdollar arithmetic enforces25 total and10/6/9 sub-caps, current/projected80% reviews, idempotent reservations, actual/interim billing observations and partial storage-lifetime forecasts. Unknown launched GPU cost blocks another paid launch until reconciliation.

The bounded runner permits reviewed project CLI/test entrypoints, never direct shell/cloud-management commands. It checks source readiness, explicit paid approval, own-Pod/volume identity and provider quote, owns/cleans the child process group, and preserves uncertain launch state without automatic replay. Training execution remains explicitly blocked until Ticket09 integrates the post-gate contract. No provisioning, volume deletion or termination API is implemented.

Exports verify hashes and independent network/downloaded receipts; an already durable network tree may be verified in place. A live download is attested on the external client. Stop intent is persisted to the network control record before own-Pod stop; confirmation and actual billing are separate operations. Failed export, missing receipt, provider error and uncertain charges remain blocked without destructive fallback. Copies/receipts are operator evidence, not authentication or proof of a physical mount without preflight.

Validation: red/green public-command tests;37 final budget/lifecycle checks passed in11.65s, including real OS reservation contention, fake paid lifecycle, external bundle confirmation, rate mismatch, timeout-group cleanup and launch-scope guards. A combined source/Stage0/SFT/budget/lifecycle regression passed78 checks in84.35s before the final guard refinements; notebook schema and execution paths were checked. The inherited full run completed669 passed/39 skipped/4 warnings in2621.17s; it collected before Ticket02 tests existed and is not a commit-isolated final full-suite claim. Some Torch-backed checks skipped due the local missing libtorch_cpu library.

A canonical zero-cost script smoke is retained under artifacts/validation/sampling-sft-ticket02. No real provider calls, Pods, bills, model forwards, paid jobs, deletion or push occurred. Real CLI schema/permissions, attached storage/locking and live shutdown still require separately approved preflight/qualification.

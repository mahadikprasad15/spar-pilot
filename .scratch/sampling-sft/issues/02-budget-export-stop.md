# 02: Run budgeted jobs through verified export and Pod shutdown

**What to build:** The user can reserve a bounded job, run it through the shared lifecycle, reconcile cost, verify network-volume/download receipts and request own-Pod shutdown. The ledger enforces25 total and10/6/9 sub-caps,80% review, storage lifetime and stage-attributed setup/reruns. A safe fixture/no-op job demonstrates the full workflow without renting compute. Destructive actions remain refused without approval.

**Blocked by:** 01.

**Status:** ready-for-agent

**Type:** task

## Acceptance criteria

- [ ] Provide a complete reserve → bounded job → cost reconciliation → verified export → own-Pod-stop workflow through public commands and the notebook/runbook using fake providers in CPU tests.
- [ ] Enforce USD25 total and sampling10/training6/storage-and-shared-overhead9 sub-caps across outstanding reservations and spent amounts; concurrent or tampered ledger updates cannot bypass limits.
- [ ] Reserve GPU hourly rate × estimated hours ×1.5 plus non-duplicated setup cost before launch; reject projected cap crossing and pause for explicit review at80% of any sub-cap or total.
- [ ] Count setup, idle, debugging and reruns against the served stage; reserve storage for its expected lifetime including stopped-Pod periods. Shared costs count once.
- [ ] Retain uncertain interrupted reservations until reconciled; publish cumulative ledger snapshots without inventing actual billing values.
- [ ] Require verified network-volume and downloaded-copy receipts before automatic stop; distinguish stop-requested from externally confirmed stopped. Failed exports do not authorize destructive fallback.
- [ ] Refuse network-volume deletion or unexported termination without explicit approval; first paid GPU work needs a post-preflight go-ahead. Demonstrate the lifecycle without real rental.

## Testing and evidence

Fake-provider workflow tests cover reservation conflicts, cap/80% rejection, interruption uncertainty, failed export, stop-request versus confirmation and deletion guards. Real provider evidence remains a separate authorized check.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

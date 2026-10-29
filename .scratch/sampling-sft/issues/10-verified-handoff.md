# 10: Assemble the adapter handoff and final research-status report

**What to build:** One verified manifest assembles source/plan, attempts/coverage/reviews, gate report, training logs/adapters and export receipts. The notebook/runbook presents a readable fresh-session route, exact completion/qualification status, limitations and deferred consumers. It can show truthful partial status while training is pending; it cannot label absent evaluation or internals as completed.

**Blocked by:** 07, 09.

**Status:** claimed

**Type:** task

## Acceptance criteria

- [ ] Assemble one source-backed handoff inventory binding original plan, raw attempts/IDs, events, review/coverage, gate statistics, training logs and adapters to verified hashes.
- [ ] Restore and verify the inventory in a fresh CPU session; stale/corrupt parents or missing required checkpoints block completed status.
- [ ] Provide a readable shared notebook/runbook with release pin, CPU/GPU labels, public stage calls, output/status views and resume instructions; no accumulated repair-cell flow.
- [ ] Report software-tested, source-audited, real-qualified, generation-complete, review-complete and training-complete states separately; support honest partial reports.
- [ ] Confirm network-volume/download receipts and stop evidence while retaining explicit approval requirements for deletion.
- [ ] List evaluation, internals, Stage0 research, baseline/offset,k2 training,k10,k5+5,self-correct and extra seeds as deferred or separately approved, never as completed.
- [ ] End-to-end offline workflow and notebook checks reconcile all products; no paid job or GitHub push follows from ticket resolution.

## Testing and evidence

Complete public-workflow fixtures from audit through handoff, stale-parent/hash rejection, report/raw reconciliation, partial versus completed status and notebook release consistency. No paid execution or GitHub push is implied by the final software ticket.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Implementation

Added CPU handoff prepare/build/verify/status and local source-release validation.
The manifest binds the original sources, frozen generation histories and IDs,
review revisions and coverage, HF gate evidence/statistics, data/readiness, selected
training logs/checkpoints, budget and lifecycle controls. Partial snapshots are
first-class; fixture completion cannot establish real GPU qualification or overall
research completion. Source-audited, software-tested, real-qualified, generated,
reviewed, trained and exported/stopped states remain distinct.

Optional absent observations retain explicit absence bindings. Required missing
checkpoints and corrupt parents are errors. Handoff verification performs no
model imports or live provider calls. Portable copies use exactly the declared
inventory plus the handoff's four output/seal files. Existing validators can
refresh disposable status projections; those are excluded from the scientific
inventory. No earlier hash-bound execution module was changed.

Receipt integrity, job identity, separate copy roles and external shutdown proof
are verified. Stop-requested alone never counts as confirmed stopped. Deferred
evaluation/internals/Stage0 research/baseline-offset/k2/k10/k5+5/self-correct/extra
seeds remain explicit. Snapshot creation authorizes no paid or destructive action.
CLI is primary; notebook19–20 are optional CPU clients. Runbook:
docs/operations/sampling-sft-handoff.md. Source-pin finalization follows the
implementation commit, without pushing.

# 10: Assemble the adapter handoff and final research-status report

**What to build:** One verified manifest assembles source/plan, attempts/coverage/reviews, gate report, training logs/adapters and export receipts. The notebook/runbook presents a readable fresh-session route, exact completion/qualification status, limitations and deferred consumers. It can show truthful partial status while training is pending; it cannot label absent evaluation or internals as completed.

**Blocked by:** 07, 09.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Assemble one source-backed handoff inventory binding original plan, raw attempts/IDs, events, review/coverage, gate statistics, training logs and adapters to verified hashes.
- [x] Restore and verify the inventory in a fresh CPU session; stale/corrupt parents or missing required checkpoints block completed status.
- [x] Provide a readable shared notebook/runbook with release pin, CPU/GPU labels, public stage calls, output/status views and resume instructions; no accumulated repair-cell flow.
- [x] Report software-tested, source-audited, real-qualified, generation-complete, review-complete and training-complete states separately; support honest partial reports.
- [x] Confirm network-volume/download receipts and stop evidence while retaining explicit approval requirements for deletion.
- [x] List evaluation, internals, Stage0 research, baseline/offset,k2 training,k10,k5+5,self-correct and extra seeds as deferred or separately approved, never as completed.
- [x] End-to-end offline workflow and notebook checks reconcile all products; no paid job or GitHub push follows from ticket resolution.

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

## Validation and release

Final handoff/release group:14 passed in82.24s. Affected source/job/budget/gate
arithmetic and earlier notebook guards:71 passed in117.40s. The public offline
fixture assembles source audit through generation, human coverage, gate,
readiness/export and a completed tiny-CPU endpoint into one hashed inventory.
A copied minimal bundle restores in a fresh CPU process with no model imports.
Missing selected checkpoints, modified approval/source evidence, changed optional
absence observations and corrupt receipts reject. Stop-requested alone does not
establish externally confirmed shutdown. Fixture stages never become real GPU
qualification or overall research completion.

Notebook20-section flow includes its source-release check, CPU/GPU labels,
optional public calls and fresh-session instructions. The code pin is
9f0e2f0ae2d1e5752efaf5744caf93844bb2ec79. The pinned code contains its public
commands; an independent notebook/pin test checks local execution sources. The
subsequent notebook-pin commit changes presentation/control metadata only.

The broad initial pytest run was interrupted after253 passed,24 skipped and
4 warnings in607.18s; it is not a full-suite pass. Development logs are separated
from final validation. Raw final logs, a pinned partial snapshot and the durable
offline fixture live in artifacts/validation/sampling-sft-ticket10/.
No real GPU qualification, generation, training, evaluation, paid job, model
download, cloud stop/deletion or GitHub push was performed by this ticket.
Full Ticket08 T16 and all real execution/approval gates remain pending.

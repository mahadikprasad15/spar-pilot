# 04: Qualify the32-problem pilot through interruption and exact resume

**What to build:** A user runs the fixed k5 pilot and receives sealed block progress, a fresh-process restore path, exact uninterrupted-versus-resumed scientific comparison, throughput/cost estimates and the explicit pilot alarm table. Failed replay or alarms block production; approved configuration changes create new attempts and requalification evidence.

**Blocked by:** 03.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Run and restore the fixed32-problem k5 pilot with block-sealed scientific evidence, operational events and true progress counters through the notebook/runbook.
- [x] Checkpoint only completed block boundaries, preserve writer ownership and scheduler/RNG state, discard and redo partial scientific blocks, and preserve already sealed bytes.
- [x] Submit logical request groups in fixed recorded order; clear prefix cache at block boundaries in both qualification and production policy, retain within-block caching and count added prefill work.
- [x] Compare uninterrupted with kill-after-block2/resume: all scientific IDs/scores/acceptances/traces/steps/coverage are exactly equal; timestamps/PIDs/events are separate.
- [x] Reject runtime mismatch. On replay failure block Stage2 and record ordered requalification: disable caching; constrain scheduling; enable supported deterministic modes. Each change creates a new runtime identity.
- [x] Publish measured lengths/throughput/memory and Stage2 reservation plus every alarm: unfinished initial attempts>2%, initial correctness<90%, or insufficient budget headroom.
- [x] No automatic prompt/J/cap adjustment; prompt/J changes require approval, new plan/run identity and complete pilot rerun. Fixture success is not real L40S qualification.

## Testing and evidence

Public-workflow fault injection covers partial-block discard, writer ownership, fixed request groups/order, cache resets, preserved sealed bytes and runtime mismatch. Real kill-after-block2 replay is recorded separately from fixture success.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

2026-10-29: Claimed after Ticket03 software closure at user request. No paid execution authorized.

## Answer

Implemented the CLI-first fixed32/seed0 k5 pilot, runtime-bound completed-block chain, fresh-session RNG restoration, explicit writer recovery, partial-science abandonment with preserved history, exact scientific comparison and first-difference diagnostics. Operational timestamps/cache/replay/error events remain separate. A controlled SIGKILL after block2 has a fresh-process recovery path; only real-CUDA blocks plus an external -9 job receipt can satisfy real qualification. No graceful pause or CPU fixture can certify it. Runtime/source/environment/sealed-file changes reject reuse. Completed blocks finalize on CPU without model reload if only finalization was interrupted.

The CPU report retains all initial outcomes, flexible-v3/strict scores, lengths, acceptance by step/cut decile, timing/memory evidence, explicit forecast inputs and all approved alarms. Failed replay remains reportable but blocked; no prompt/J/cap/tolerance adjustment occurs. The proposed Stage2 cost uses the approved1.5 buffer and current ledger headroom; it is not a reservation or paid approval.

Validation:88 combined source/budget/lifecycle/sampler/engine/instrument/pilot CPU tests passed in97.62s. Includes a real local CPU child killed with SIGKILL, stale-owner rejection/recovery, fresh-process restore, exact scientific equality, preserved sealed bytes, partial-block redo, transient engine errors, environment tamper rejection, runtime mismatch, failed replay coordinates and CPU-only finalization. Notebook JSON/cells are clean; default CPU cells run in regression. Canonical synthetic evidence:artifacts/validation/sampling-sft-ticket04/.

Real L40S execution/throughput/T3/replay, actual provider/storage behavior and Stage2 paid approval remain pending. No model forward, download, paid job, Pod rental/deletion, or push occurred. Runbook:docs/operations/sampling-sft-pilot.md. Ticket05 is next, not implemented by this slice.

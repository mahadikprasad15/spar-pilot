# 04: Qualify the32-problem pilot through interruption and exact resume

**What to build:** A user runs the fixed k5 pilot and receives sealed block progress, a fresh-process restore path, exact uninterrupted-versus-resumed scientific comparison, throughput/cost estimates and the explicit pilot alarm table. Failed replay or alarms block production; approved configuration changes create new attempts and requalification evidence.

**Blocked by:** 03.

**Status:** ready-for-agent

**Type:** task

## Acceptance criteria

- [ ] Run and restore the fixed32-problem k5 pilot with block-sealed scientific evidence, operational events and true progress counters through the notebook/runbook.
- [ ] Checkpoint only completed block boundaries, preserve writer ownership and scheduler/RNG state, discard and redo partial scientific blocks, and preserve already sealed bytes.
- [ ] Submit logical request groups in fixed recorded order; clear prefix cache at block boundaries in both qualification and production policy, retain within-block caching and count added prefill work.
- [ ] Compare uninterrupted with kill-after-block2/resume: all scientific IDs/scores/acceptances/traces/steps/coverage are exactly equal; timestamps/PIDs/events are separate.
- [ ] Reject runtime mismatch. On replay failure block Stage2 and record ordered requalification: disable caching; constrain scheduling; enable supported deterministic modes. Each change creates a new runtime identity.
- [ ] Publish measured lengths/throughput/memory and Stage2 reservation plus every alarm: unfinished initial attempts>2%, initial correctness<90%, or insufficient budget headroom.
- [ ] No automatic prompt/J/cap adjustment; prompt/J changes require approval, new plan/run identity and complete pilot rerun. Fixture success is not real L40S qualification.

## Testing and evidence

Public-workflow fault injection covers partial-block discard, writer ownership, fixed request groups/order, cache resets, preserved sealed bytes and runtime mismatch. Real kill-after-block2 replay is recorded separately from fixture success.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

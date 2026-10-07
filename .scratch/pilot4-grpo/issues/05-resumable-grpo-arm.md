# 05: Train and resume the complete 64-step GRPO arm

**What to build:** Run the frozen rank-1 GRPO arm once, with all rollouts,
diagnostics and scientifically resumable checkpoints durably saved.

**Blocked by:** 04 — Preflight the target GPU and freeze the resolved protocol.

**Status:** ready-for-agent

- [ ] Execute 64 optimizer steps covering the exact 512 training prompts once,
      eight prompts/eight completions per step; prove totals and group identity.
- [ ] Save all 4,096 committed completions with text/token IDs, prompt/group/draw
      identity, cap status, strict/flexible scoring, reward and advantage.
- [ ] Log loss, reward, dead groups, length mean/90th percentile, cap fraction,
      learning rate, pre-clip gradient norm, chosen consistency statistic and
      adapter norm per step; represent undefined values with coverage.
- [ ] Seal checkpoints at 0/8/16/32/64 with optimizer, scheduler, RNG, trainer,
      data-order and rollout-progress state required for exact continuation.
- [ ] Verify hashes and resume without duplicated committed rollouts/logs;
      demonstrate future group and RNG equivalence after interruption.
- [ ] Apply approved stop/pause/validity-monitor policies and save explicit
      failure evidence; a dead child process cannot appear indefinitely active.
- [ ] Publish successful units only after final integrity-boundary checks;
      measure overhead and avoid per-projection full-model hashes.
- [ ] Offline workflow tests exercise interruption, corruption, incompatible
      settings, locking, no duplicate IDs and completed-run reuse.
- [ ] Notebook progress reports optimizer steps, prompts, completions and
      verified resume position with an explanation of each counter.

## Comments

- October 7, 2026: approved as ticket 5. Real scientific training waits until
  ticket 4's full-run prerequisites have passed and been frozen.

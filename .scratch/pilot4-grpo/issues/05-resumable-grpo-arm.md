# 05: Train and resume the complete 64-step GRPO arm

**What to build:** Run the frozen rank-1 GRPO arm once, with all rollouts,
diagnostics and scientifically resumable checkpoints durably saved.

**Blocked by:** 04 — Preflight the target GPU and freeze the resolved protocol.

**Status:** claimed

- [ ] Execute 64 optimizer steps covering the exact 512 training prompts once,
      eight prompts/eight completions per step; prove totals and group identity.
- [ ] Save all 4,096 committed completions with text/token IDs, prompt/group/draw
      identity, cap status, strict/flexible scoring, reward and advantage.
- [ ] Log loss, reward, dead groups, length mean/90th percentile, cap fraction,
      learning rate, pre-clip gradient norm, consecutive flattened pre-clip
      gradient cosine and adapter norm per step; represent undefined first-step
      or zero-gradient comparisons with reasons and coverage.
- [ ] Seal checkpoints at 0/8/16/32/64 with optimizer, scheduler, RNG, trainer,
      data-order and prior-gradient diagnostic state, binding accepted rollout/
      log history to each sealed boundary using existing checkpoint helpers.
- [ ] Recover training only from the latest sealed checkpoint. Preserve/exclude
      later incomplete-attempt records, restore state and redo subsequent steps.
      Demonstrate future group/RNG equivalence and gradient-cosine continuity;
      no mid-window continuation or partial-rollout recovery is required.
- [ ] Apply approved stop/pause/validity-monitor policies and save explicit
      failure evidence; a dead child process cannot appear indefinitely active.
- [ ] Publish successful units only after final integrity-boundary checks;
      measure overhead and avoid per-projection full-model hashes.
- [ ] Offline workflow tests exercise interruption, corruption, incompatible
      settings, existing locking, accepted-history IDs and completed-run reuse.
      Reuse existing atomic writes/sealing; build no new persistence framework.
- [ ] Notebook progress reports optimizer steps, prompts, completions and
      verified resume position with an explanation of each counter.

## Comments

- October 7, 2026: approved as ticket 5. Real scientific training waits until
  ticket 4's full-run prerequisites have passed and been frozen.
- October 7 amendment: checkpoints are the sole training recovery boundaries.
  Up to nearly 32 steps can require redo; incomplete attempts are diagnostic
  evidence rather than additional scientific completions.

- October 8: user approved length-change monitor as absolute fractional change
  in current step mean tokens versus step n−8; starts at step 9. A zero previous
  mean is undefined. The training manifest explicitly records this definition.

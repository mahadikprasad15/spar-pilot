# 03: Validate the GRPO algorithm on offline tiny models

**What to build:** Demonstrate that the intended group arithmetic, reward,
policy gradient and optimizer update work through the training boundary before
loading the scientific model.

**Blocked by:** 01 — Verify source evidence and prepare the GRPO matching audit.

**Status:** claimed

- [ ] Resolve the exact pinned trainer API and expose every relevant trainer
      argument, with eight prompts/eight draws and one fresh rollout window
      per optimizer step independent of memory microbatching.
- [ ] Verify the selected whole-window completion-token loss using an
      independent differentiable calculation on unequal-length examples;
      compare adapter gradients across microbatch/accumulation configurations.
- [ ] Verify group-standardization convention/stabilizer, zero-sum advantages,
      zero policy gradients for all-dead groups, padding masks and cap handling.
- [ ] Distinguish a zero gradient from zero Adam parameter movement when
      previous momentum exists; preserve diagnostic correctness.
- [ ] Build offline tiny real Transformers/PEFT positive-learning and
      flipped-advantage negative controls: two seeds (42/43), 20 steps each.
      In both seeds the steps 16-20 mean reward must rise by at least 0.2 versus
      steps 1-5 for the positive control and fall by at least 0.2 for the negative.
- [ ] Freeze an attainable toy task and initial policy with expected reward
      near the middle of its range before acceptance runs; paired controls
      share settings except advantage sign. Diagnose failures without tuning
      the frozen criterion or task after observing outcomes.
- [ ] Verify consecutive-step flattened pre-clip gradient cosine after full
      accumulation, with stable parameter order; first-step and zero-gradient
      comparisons are undefined with reasons/coverage.
- [ ] Verify that only intended adapter parameters train and that smoke/control
      state cannot become scientific steps or consume scientific RNG/order.
- [ ] Save resolved dependency/algorithm evidence and a compatibility decision;
      no upgrade rewrites previous pilots' environments or evidence.
- [ ] Make the controls runnable and explained from the command/notebook
      boundary without model downloads or GPU requirements.

## Comments

- October 7, 2026: approved as ticket 3. DAPO and BNPO are not accepted as
  interchangeable labels; equivalence is decided by the actual denominator
  and independent accumulation-window gradient check.
- October 7 amendment: learning controls and consistency statistic are now
  explicitly specified; they are implementation checks, not statistical claims.

- October 7 implementation: group/sample-standardization, pinned real TRL DAPO
  whole-window loss, independent differentiable adapter gradients and memory
  microbatch equivalence verified on CPU. Dead-group policy gradients are zero;
  existing Adam momentum can still move parameters. The initial one-token
  fixture used a zero padding embedding; construction was corrected without
  changing the task/optimizer/criterion. Original failed evidence is preserved.
- Learning acceptance remains FAILED, not resolved. Both seeds/signs fail the
  approved 0.2 reward-change criterion. Diagnosis: the random frozen two-token
  readout bounds expected reward to [0.3731, 0.6269] at seed 42 and
  [0.3928, 0.6072] at seed 43. This is inadequate expected learning headroom,
  not evidence that the GRPO policy gradient is reversed or broken.
- Asked for approval of a versioned toy-fixture correction: fixed unit output
  direction orthogonal to the initial prompt state (initial reward 0.5), with
  a wider attainable range; preserve the binary task, all 20-step/two-seed
  criterion, learning rate and old evidence. No correction acceptance run is
  authorized until that reply. Command/notebook control handoff is not complete.
- Evidence: artifacts/runs/diagnostics/pilot4-ticket03-zero-embedding-failure/
  and artifacts/runs/diagnostics/pilot4-ticket03-learning-failure/.

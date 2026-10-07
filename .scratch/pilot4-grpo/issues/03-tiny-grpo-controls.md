# 03: Validate the GRPO algorithm on offline tiny models

**What to build:** Demonstrate that the intended group arithmetic, reward,
policy gradient and optimizer update work through the training boundary before
loading the scientific model.

**Blocked by:** 01 — Verify source evidence and prepare the GRPO matching audit.

**Status:** ready-for-agent

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
      flipped-advantage negative controls with declared criteria before runs.
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

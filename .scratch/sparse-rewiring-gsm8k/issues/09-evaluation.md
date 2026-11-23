# 09: Evaluation wiring

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03
**Group:** C

Validation CE, greedy validation and greedy cohort150 for rewire checkpoints, reusing `c_attribution_generation.greedy` / `score` / `batches` and flexible-v3. A rewire-checkpoint loader (`sft_evaluation` only loads PEFT checkpoints 0–64). Gold-sft step 0 and 64 references on validation run in a separate child process.

- [ ] Red/green: checkpoint 0 equals untuned outputs on the tiny model; spec test 22 (pad and EOS from tokenizer files).
- [ ] Cohort150 only at 64 and final (D12); prior uses recorded from sealed units.

## Comments

## Answer

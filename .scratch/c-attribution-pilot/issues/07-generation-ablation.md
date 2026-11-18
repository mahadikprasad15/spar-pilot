# 07: Generation-ablation stage

**Status:** needs-triage
**Type:** task
**Blocked by:** 06

**What to build:** `pilot_eval/c_attribution_generation.py`.

- [ ] Conditions per arm from the freeze: intact, top10-adapters, match10-adapters, top10-neurons, match10-neurons, top64-neurons, match64-neurons; plus untuned. Never combine adapter and neuron removals.
- [ ] Modes: `greedy` (initial launch) and `sampled` (8 draws, T = 1, top_k 0, top_p 1, rep 1.0, seeds 42 + batch index; for the follow-up). Both use 1,024 cap, EOS 151645, FP32, the qualified batch rule, left padding, and settings passed explicitly over `generation_config`.
- [ ] Question sets: `cohort150` (results labelled exploratory) or `heldout-N` (GSM8K test minus cohort150 minus the 20 rehearsal items, seeded; sealed ID list).
- [ ] Each unit saves responses JSONL (prompt, generated IDs including EOS, text, stop reason, flexible-v3 score, reward) and a config file (AGENTS.md).
- [ ] Summary per condition: accuracy with Wilson interval; Δpp vs intact; % of intact; top − match difference with a paired bootstrap; gains and losses and exact McNemar p; invalid and capped counts; mean and median tokens. Untuned row. The labels "exploratory (selected on the same questions)" and "greedy ≠ sampled J" are written whenever they apply.
- [ ] Tests: tiny-model greedy and sampled runs reproduce under a seed; an intervention changes logits only where expected; summaries are hand-checked on a fixture; labels appear for the cohort150 set and not for heldout.

## Comments

## Answer

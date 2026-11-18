# 10: CPU whole-chain test through the real backend

**Status:** needs-triage
**Type:** task
**Blocked by:** 09

**What to build:** `tests/test_c_attribution_realpath.py`, modelled on `tests/test_attribution_realpath.py` (lesson 2026-11-16: the real boundary catches what fakes cannot).

- [ ] A real-format world: `save_pretrained` Qwen2 base and PEFT arms; a `tokenizers` tokenizer with the chat template; behaviour JSONL with sampled and greedy rows built the way the real files are (EOS, capped, invalid, all-same groups); the input bundle from 08; the launch record sealed from the **actual** rehearsal YAML.
- [ ] Run the whole chain with the real `TransformersBackend`, stubbing only CUDA memory counters, the volume mount, the Hub and the stop command.
- [ ] Put back each recorded Pod failure (wrong preflight argument, YAML text number, nested type, HF cache on the volume, wrong stop syntax) and show the chain stops with that error before GPU work.
- [ ] An interrupted unit plus a deleted local root resumes from the volume and finishes.

## Comments

## Answer

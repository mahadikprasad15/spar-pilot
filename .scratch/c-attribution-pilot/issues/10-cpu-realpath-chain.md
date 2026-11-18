# 10: CPU whole-chain test through the real backend

**Status:** needs-triage
**Type:** task
**Blocked by:** 09

**What to build:** `tests/test_c_attribution_realpath.py`, modelled on `tests/test_attribution_realpath.py` (lesson 2026-11-16: the real boundary catches what fakes cannot).

- [ ] **A real-format world:**
  - `save_pretrained` Qwen2 base and PEFT arms;
  - a `tokenizers` tokenizer with the chat template;
  - a `generation_config.json` copied from the pinned real one;
  - behaviour JSONL and source configs built the way the real files are (EOS, capped and invalid rows; all-same groups);
  - the input bundle from ticket 08, including a catalog revision;
  - the launch record sealed from the **actual** rehearsal YAML, including a hot-layer list.
- [ ] **The whole chain** runs with the real `TransformersBackend`. Only CUDA memory counters, the volume mount, the Hub and the stop command are stubbed.
- [ ] **Dependencies must execute.** The test **fails, not skips,** if torch, transformers, PEFT or tokenizers are missing. CI for this ticket records the dependency versions. (The local Mac environment currently lacks a working torch; the test must run in an environment that has one.)
- [ ] **Old metrics.** `trace` and `stop` outputs from the old path are bit-identical on this world.
- [ ] **Recorded Pod failures.** Put back each one and show that the chain stops with that error before GPU work:
  - the wrong preflight argument;
  - a YAML number read as text;
  - a nested type;
  - the HF cache on the volume;
  - the wrong stop syntax;
  - a missing catalog;
  - a modified adapter after restore.
- [ ] **Resume.** An interrupted unit plus a deleted local root resumes from the volume and finishes.

## Comments

- 2026-11-18 (round 1): the review asks that the real dependencies actually execute rather than being skipped, and that old-metric behaviour is preserved.

## Answer

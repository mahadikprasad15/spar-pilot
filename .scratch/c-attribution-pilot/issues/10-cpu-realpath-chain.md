# 10: CPU whole-chain test through the real backend

**Status:** resolved (2026-11-19; awaiting the pre-GPU review)
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
- 2026-11-19 (implementation): built; it found one real bug (below).

## Answer

`tests/test_c_attribution_realpath.py` (11 tests) on `tests/c_attribution_realpath_fixture.py`: a 28-layer saved
Qwen2, two saved PEFT arms, a `tokenizers` tokenizer with a chat template, the pinned `generation_config.json`,
real-format behaviour rows and configs (EOS, capped, invalid, all-same and mixed groups), an effective-write table
giving {11, 15, 16, 24}, a git-committed catalog that `catalog verify` passes, the bundle exported and restored
through the fake Hub, and the record sealed from the actual rehearsal YAML with CPU sizes (cap 12, budget 256,
500 resamples). torch, transformers, PEFT and tokenizers are hard imports (fail, never skip).

- Whole chain completes through the real backend in about 16 s; trace and stop are bit-identical with and
  without seqlogp on this world.
- Recorded failures stop before GPU work: wrong CLI argument; YAML number as text; nested wrong type; wrong stop
  syntax (seal refused); HF cache on the volume; missing catalog; adapter modified after restore; the real
  volume check without its identity file (preflight stops; the Pod stop is still requested).
- Resume: an interruption after the first sealed unit, then the local root deleted, resumes from the volume
  (take-over of a dead writer), reuses the restored unit and completes.
- Bug found: `c_attribution_checks.resolve_processors` patched `_get_logits_processor` on the PeftModel, but
  generate resolves processors on the inner transformers model, so on any real arm (check 9 / (f)) it raised.
  Fixed, with a PEFT regression test. Also: the row loader now takes the cap and EOS from the bound policy
  (it used the module constant for the cap); production requires 1024 / 151645.
- Stubbed only: CUDA availability and memory counters, the volume mount/identity and disk facts, the Hub, the
  stop command. The world's controls are mostly unavailable (28 x 32 reads; layer64 takes every positive read):
  reported as analysis, as the rule says.


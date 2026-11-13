# 01: Trace preparation code

**Status:** ready-for-agent — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** none. CPU only; running it on a GPU happens in 06a (calibration) and 06b (discovery, confirmation).

**What to build:** the path from frozen prompts to sealed base-correct teacher-forcing rows.

- [ ] Refuse unless `inventory-v2` and `sequences-v2` verify by hash; record both.
- [ ] Generate under the pilot protocol at a given batch manifest; save prompt IDs, generated IDs, stop reason and text before processing; config and all responses as JSONL.
- [ ] Score with `score_gsm8k_flexible_v3`; keep correct, uncapped rows; teacher-forcing rows are prompt IDs + generated IDs, never re-tokenized.
- [ ] Seal the exact stop IDs passed to the decoder (pilot backend: `tokenizer.eos_token_id` = 151645 only; 151643 is padding, not a stop) with their strings; refuse a configuration whose stop set differs from the decoder's.
- [ ] Masks: trace targets (all content tokens incl. first; excl. `<|im_end|>`, newline, prompt, header, padding); stop-pull prefix positions before the last non-empty content line, by token offsets.
- [ ] Accept/reject counts and reasons by split, with question and response lengths; minimums checked after all mask exclusions; stop below them.
- [ ] Per-arm provenance (checked sources, unknowns) copied from the `sequences-v2` leakage record into the manifest.
- [ ] Split-visibility flags in the manifest; confirmation outputs refused by any reader before the freeze.
- [ ] Tests: single stop ID sealed from a decoder fixture; a mismatched stop set refused; generated IDs preserved; single-line, `####`, prose and boundary-crossing stop-pull masks; empty mask recorded; minimum check after exclusions; hash-mismatch refusal.

## Comments

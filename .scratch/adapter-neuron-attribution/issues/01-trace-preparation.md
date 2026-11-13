# 01: Trace preparation code

**Status:** resolved (2026-11-13) — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
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

### 2026-11-13 — question for the user (implementing agent)

- **Rehearsal generation items.** The spec says "GSM8K test items outside the ADR 0001 cohort, seed 42, 20" but not the draw rule. A natural reading is `sorted(random.Random(42).sample([i not in select_gsm8k_indices(1319)], 20))`. That is a choice, and it is not in the spec or an ADR, so I have not implemented it. 06a and 06b need it. Please confirm or give the rule.

## Answer

2026-11-13 — `pilot_eval/attribution_traces.py`, tests `tests/test_attribution_traces.py` (18, CPU; one uses a tiny random Qwen2 for `HFGenerator`).

- `verify_sources` refuses unless `inventory-v2` and `sequences-v2` verify (the existing `verify` functions). It records both seal hashes and copies the per-arm provenance from `leakage_report.json`; the statement is "no overlap with available sources". `load_gold` reads gold answers from the GSM8K train parquet after checking its hash against the sequences manifest.
- `prompt_rows` takes math rows of one sequences-v2 split (`calibration`, `discovery`, `heldout`). Prompt IDs = the frozen prefix through the assistant header (`input_ids[:reply_start+1]`), so nothing is re-tokenized.
- `generate` follows the frozen batch manifest's order and refuses a manifest that doesn't cover each row exactly once. `HFGenerator` does greedy FP32 generation with left padding and the decoder's EOS; generated IDs include the final EOS.
- `seal_responses` seals every raw response (prompt IDs, generated IDs, stop reason, text) plus the run config before any scoring. `seal_rows` then scores with `score_gsm8k_flexible_v3` and keeps correct, EOS-terminated (uncapped) rows. A teacher-forcing row is exactly prompt IDs + generated IDs. Rows, per-response records (reason, question and response token counts) and a manifest (counts, stop IDs, provenance, visibility, response-file hash) are sealed. The minimum is enforced after sealing, so the evidence survives a stop.
- `seal_stop_ids` takes the stop IDs from the decoder config (`eos_token_id`; pilot: {151645}) with their strings and the pad ID. `metric_stop_ids` refuses any other stop set unless it is labelled a decoding change.
- **Masks** (`build_masks`) are target positions.
  - `trace_targets` = every generated content token, the first included and the final `<|im_end|>` excluded.
  - `stop_targets` = content tokens whose decoded end offset is at or before the start of the last non-empty line, so a token crossing that newline is in the last line. Offsets come from incremental decoding of the generated IDs, and a decode that isn't a prefix of the next is rejected.
  - A single-line response has an empty stop mask (`single-line-response`).
- **Choices made from the spec wording:**
  - "Minimums checked after every mask exclusion" is implemented as rows with a non-empty stop mask (`after_mask_exclusions`). The trace mask is never empty for an accepted row. Stop-mask-empty rows stay in trace analyses and are counted separately.
  - The "trailing newline" in the exclusion list is the template newline after `<|im_end|>` in the joint-audit sequences. Generated IDs end at `<|im_end|>`, so it never appears here. A newline inside the content is a content token.
- `load_rows` refuses heldout (confirmation) rows until a sealed selection freeze is given. Calibration is `visible`, discovery `attribution-only`, heldout `blind-until-freeze`.
- Minimums come only from the launch record (a missing value is refused); the proposed 120/60 are not hard-coded.

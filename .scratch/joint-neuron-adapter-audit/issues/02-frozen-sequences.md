# 02: Freeze paired audit sequences and leakage report

**Status:** needs-triage — draft for adversarial review

**What to build:** From pinned source data and the inventory, build and seal reproducible discovery/held-out manifests, exact token IDs and all position masks. Produce overlap, provenance, filtering, truncation and group-size reports.

**Blocked by:** 01.

- [ ] Check union of available question IDs and normalized hashes for every arm; explicitly report unknown provenance and insufficient clean examples.
- [ ] Freeze a separately screened calibration split targeting 25 math and 25 general-background examples, disjoint by question/conversation from discovery and held-out; report an insufficient pool rather than reusing protected examples silently.
- [ ] Pin background source, license, filters and saved manual review; general and number-containing near-miss groups are disjoint by conversation. Extract first turns only and test with a multi-turn record.
- [ ] Correctly exclude padding, prompt and template tokens; test reply start, end-of-turn, truncation and zero-token edge cases.
- [ ] For the pinned ordinary Qwen chat template, inspect the assistant header newline, response content, `<|im_end|>` and trailing newline using actual token offsets; save decoded mask views for three representative examples. Treat a token spanning a boundary as ambiguous, not automatically a standalone newline token.
- [ ] Frozen seed and stable IDs reproduce the manifest; unresolved source or selection decisions stop preparation.
- [ ] Seal token IDs, example order and length metadata for a common cohort. The GPU qualification uses these records to freeze the final shared batch membership, padding and position manifest after benchmarking.

## Comments

### 2026-11-10 — first implementation (Claude)

- User decisions (2026-11-10): math rows use `protocol.gsm8k_messages(question)` + the GSM8K reference answer (Gold SFT training format). Background is `HuggingFaceH4/ultrachat_200k@8049631c…` `test_sft` (MIT), first turn only.
- Code: `pilot_eval/neuron_audit_sequences.py` (`prepare` / `verify` / `review`), tests `tests/test_neuron_audit_sequences.py` (25: masks, boundary-crossing rejection, empty/truncated, multi-turn first turn, filter labels, leakage by ID/normalized text/test text, seeded reproducibility, insufficient pool stops, unsealed inventory stops, review binding, resume/changed/corrupt). One test checks the jinja2+tokenizers renderer against the transformers-rendered Gold items; it matched all 512 locally (text and IDs).
- Sealed run: `pilot1/artifacts/runs/joint-neuron-adapter-audit/sequences/sequences-v1/` from plan `pilot1/artifacts/plans/joint-neuron-adapter-audit-v1/sequences.plan.json` (artifact root `/Users/prasadmahadik/spar`, because Sampling SFT lineage lives in `hf-sampling-sft/unpacked`). Counts: math 200/100/25, general 200/100/25, near-miss 70/30. 258,088 tokens; max 1,173; 167,524 response-content tokens; none truncated.
- Leakage: every arm's lineage covers the same 512 `gsm8k:train` IDs plus 150 test IDs; 6,961 eligible math candidates remain. No duplicate-text or test-text matches. Online GRPO is `partial` (its frozen plan `plans/pilot4-grpo-fp32-v1-frozen-v2/grpo.frozen.json` is not local). Cleanliness is reported as "no overlap with available sources", never "clean".
- **Proposed values that need review:** the code/math/digit filter regexes (documented in `filter_report.json`) and `max_tokens=2048` with `exclude`. Background review is **pending**: inspect `review/background_sample.jsonl`, then record it with `python -m pilot_eval.neuron_audit_sequences review --run_dir … --reviewer … --decision accept|reject`. 03b must refuse while the review is pending.
- Batch membership/padding/position IDs are deliberately not frozen here (03b).
- Re-sealed as `sequences-v2` against `inventory-v2` (plan `sequences.v2.plan.json`); `sequences.jsonl` is byte-identical to v1. Use v2 downstream. Background review is still pending.

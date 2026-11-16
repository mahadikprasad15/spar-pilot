# Response to review round 6

Baseline: `review-round-6.md` (Codex, on `753de57`). Each point was checked against the code before deciding.
Every fix below was written test-first (red, then green) and committed separately. After these fixes all 218
attribution tests pass on CPU. Nothing here was run on a GPU.

| # | Disposition | Commit | Test |
|---|---|---|---|
| 1 | **Adopt.** Confirmed: `_start`, `observed_volume_identity` and `attribution_stages.build` (real backend construction) all run outside `Chain.run`'s teardown. Fixed in the watchdog rather than in Python, so that *any* exit of the writer process is covered: the CLI, construction, `_start`, or a crash anywhere: if the writer has exited and the done file is absent (checked again after 2 s, since it is written just before exit), the watchdog copies sealed units and requests the stop at once. Its summary records `reason: writer-exited-without-completing`. Ownership checks unchanged. | `43f2a80` | `test_r6_1_watchdog_acts_at_once_when_the_writer_dies_without_completing`; `…still_stands_down_when_the_writer_finished_normally` |
| 2 | **Adopt.** Confirmed. `invariance_budgets(sweep)`: the smallest and the selected budget among candidates the sweep actually exercised *and* that fitted; a budget skipped (`longest-example-exceeds-budget`) or OOM is never packed. One valid budget gives one comparison. | `14936f5` | `test_r6_2_invariance_uses_only_exercised_fitting_budgets` |
| 3 | **Adopt.** Confirmed: the sequences-v2 manifest pins `inputs.tokenizer.files` (`neuron_audit_sequences.py:370`: `tokenizer.json`, `tokenizer_config.json`, which holds the chat template), but `bind_sources` never compared them. `bind_sources(…, tokenizer_dir=…)` now refuses a changed or unpinned tokenizer, and the backend's `ChatTokenizer` is built with the same pins. Not yet verified against the real manifest on the Pod: the Pod's tokenizer files (`c0382117…`, `5b5d4f65…`, verified by `prepare` against inventory-v2) are the project's pinned revision, so a match is expected; a mismatch would now stop the run at startup (and, with point 1, stop the Pod at once). | `a9813ce` | `test_r6_3_tokenizer_files_are_bound_to_the_sequences_manifest` |
| 4 | **Adopt.** Confirmed with Codex's reproduction. Only `stage_d.pilot_results.*.path` is text; `…filters.*` keep any scalar type (they are compared with the rows' own fields). The chain test fixture used the wrong `pilot_results` shape (plain strings); corrected to `{path, filters}`. | `92430e3` | `test_r6_4_pilot_filter_values_keep_their_type_and_paths_must_be_text`; `test_r6_4_the_production_draft_has_no_type_errors` |
| 5 | **Adopt.** Confirmed. Note on severity: Stage D runs after the freeze, so this did not let a cohort output influence any selection; it did break the backend's stated contract and ran a cohort prompt through a model outside the protocol's generation runs. The probe is now always `rehearsal_items()[0]`. | `137ff52` | the production end-to-end CPU test asserts every probe is a rehearsal prompt |
| 6 | **Defer (accepted for the supervised rehearsal; open for unattended production).** Agreed with Codex's labelling. The rehearsal is supervised, the total-deadline watchdog is the backstop, and with point 1 a crash no longer waits for it. Before an unattended overnight launch: `ctx.boundary()` between sweep candidates, invariance budgets and generation candidates, with a fake-clock test. Recorded on ticket 11 with round-5 items 1–3. | — | — |
| 7 | **Adopt in part.** The ordering defect is fixed: `parameter_count` now sums tensor shapes from the snapshot's safetensors headers and loads no model, so nothing reaches the GPU before preflight. Multiple resident copies (untuned + arm + transient FP64; edited Stage D base) are real but are a capacity question, not a defect: on 40 GB, 06a peaked at 27.4–31.6 GB with the same copies. The supervised Pod run measures them (the sweep and the FD check already record peaks); I have not changed model lifetimes without a measurement showing they matter on 80 GB. | `158d37f` | `test_r6_7_parameter_count_reads_the_weights_header_without_loading_a_model` |
| 8 | **Adopt.** Stage D run config now records `prompt_template` (the actual render path: `protocol.gsm8k_messages(question)` + the pinned chat template with `add_generation_prompt=True`), `prompt_ids`, `tokenizer` (identity from the backend) and `seed: 'greedy (no sampling)'`. | `3fe719c` | production end-to-end CPU test asserts the fields |
| 9 | **Adopt.** `INTEGER_FIELDS`: counts, seeds, budgets, `max_new_tokens` and the generation batch rule must be ints; thresholds, caps and ratios may be real. The approved rehearsal record passes unchanged. | `2afd276` | `test_r6_9_counts_seeds_and_budgets_must_be_integers` (5 cases) |

## The real-path CPU test (Codex's "next step")

Agreed; this is the test that would have caught Pod launches 1 and 2. Status: not built yet in this round. Plan:
a tiny random Qwen2 + rank-1 PEFT snapshot and a real `tokenizers` WordLevel tokenizer with a Qwen-shaped chat
template, written to disk; a sealed inventory and a sequences-v2 run produced by the real
`neuron_audit_sequences.prepare` with that tokenizer; a GSM8K-shaped test parquet; the launch record built from a fresh
copy of `launch-record-pod-rehearsal.yaml` through `seal` (only sizes reduced for CPU); the real
`TransformersBackend` and `Stages` through `Chain`, with only CUDA accounting, the volume mount and the Hub stubbed;
plus an injected startup failure and the skipped-budget case. One open difficulty: base traces keep only rows the
untuned model answers correctly, which a random tiny model does not; the fixture will derive the gold answers from
the tiny model's own greedy outputs (two passes), so the real acceptance code runs unmodified.

## What the next supervised Pod run must record

Per Codex: peak memory at each qualification step (already in `details.json`), elapsed time per stage (budget
file), sealed calibration row lengths (rows run), stage outcomes, backup verification and the actual stop state.
No approved tolerance or cap changes.

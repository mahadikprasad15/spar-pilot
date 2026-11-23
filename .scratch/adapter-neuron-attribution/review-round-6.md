# Attribution pipeline review, round 6

_Codex's review of `753de57`, pasted into the session by the user on 2026-11-16 (Codex's own copy was not pushed).
Kept here, abridged only in the "fake versus real" table and the reproduction snippets, so the response has its
baseline in the repository. Response: `review-response-round-6.md`._

Scope: checkout at `753de57`; `2bce6a6..753de57` contains recent fixes, while this review intentionally covers the full real execution paths requested. Labels refer to the next supervised Pod rehearsal unless stated otherwise; a production blocker can have lower urgency for the rehearsal. Evidence is distinguished below between CPU reproduction, code inference, and unmeasured GPU behavior. No end-to-end GPU run or A100 runtime is established.

## Priority reassessment for the next supervised rehearsal

The local 06a record reports that Colab r3 passed all 11 steps on an A100-SXM4-40GB, including both-arm attribution and interventions, edited-checkpoint logits equal to the gate, and a 135-file export and clean restore; 31.6 GB at 4,068 tokens and OOM at 8k and above. The r3 finite-difference checks failed the gate; the later diagnostic 4 (full-FP64 reference) passed for both arms on four rows (48 comparisons per arm, three adapters and three neurons, maxima 5e-5 and 1e-5, 27.4 GB peak). The Pod draft requests five adapters and three neurons, so that diagnostic does not cover the identical selection. 06a's driver takes a different route through shared backend and stage helpers; it does not establish the complete Pod qualification sweep, invariance and generation selection, chain sign-off, or 80 GB memory and timing.

**Priorities:**

- The skipped-minimum-budget invariance defect is real under its stated condition; relevance to the next run depends on the Pod's actual row lengths.
- Startup failures bypassing normal chain teardown remain consequential if a startup exception occurs. Tokenizer identity is a real missing provenance check, with no evidence the present tokenizer differs.
- The numeric pilot-filter validator and the production-cohort prompt for edit verification are production-only defects.
- GPU memory, qualification duration and numerical gates require measurement on the actual Pod. Do not change approved tolerances to obtain a pass.

**Next step:** a small targeted CPU integration check using the real backend/stage boundary and the actual rehearsal-shaped launch record, with tiny local model/data fixtures and only device and external I/O stubbed; include startup failure and the skipped-budget case. Then one bounded, continuously supervised Pod rehearsal without automatic restart, recording peak memory, elapsed time, sealed row lengths, stage outcomes, backup verification and actual stop state.

## Findings

1. **Blocking — startup failures can bypass chain teardown and delay Pod stop.** `attribution_chain.py:536-538,549,609-618,789-793`; `attribution_watchdog.sh:25-27`. `_start()` runs before the protected region; the CLI validates the volume and builds the real backend before `Chain.run()`. A failure there skips summary, backup and stop; the watchdog waits for the full deadline. Reproduced with a mock. **Fix:** cover post-watchdog startup failures, or have the launcher detect writer exit and trigger a bounded safe stop/backup. **Validate:** inject failures during backend construction and `_start`.
2. **Blocking, conditional — smallest skipped token budget is still used for invariance.** `attribution_stages.py:197-203,249-265,673-683`. The sweep can skip a budget below the longest calibration row and select a larger one, then pass the skipped minimum to `batch_invariance`, whose packer rejects the row. CPU reproduction with a 2,049-token row. **Fix:** compare only valid exercised budgets.
3. **Blocking — tokenizer identity is not bound to the sealed source.** `attribution_backend.py:57-63,81-96,140-148`. Binding verifies adapters, base files and the test parquet, not the tokenizer files/chat template against the sequences manifest. **Fix:** bind tokenizer assets and template to the sealed identity; fail on mismatch.
4. **Blocking for production — numeric pilot filter rejected.** `attribution_chain.py:154-174,191-193`; `launch-record-draft.yaml:81-83`; `attribution_report.py:617-620`. `stage_d.pilot_results.*` classifies nested filter values as text; `filters.step: 64` is rejected, and a string would fail the report's literal comparison. **Fix:** validate pilot paths separately from typed filter values.
5. **Blocking for production — edit verification consumes a blind cohort prompt.** `attribution_stages.py:510-515,546-549`; `attribution_backend.py:248-267`. Stage D passes `items[0]['prompt_ids']` from the production cohort, although the backend contract calls for a rehearsal prompt. **Fix:** always use a non-cohort rehearsal probe.
6. **Non-blocking for a supervised rehearsal; blocking for a claimed hard stage time bound — qualification loops do not check the cap between units.** `attribution_stages.py:175-207,249-267,269-291,673-683`. **Fix:** check the persisted budget between bounded GPU units.
7. **Non-blocking — monitor loads the model before preflight; multiple large copies can coexist.** `attribution_stages.py:86-99,175-190`; `attribution_backend.py:223-241,248-262,276-296`. `parameter_count()` loads and retains the untuned FP32 model before `preflight()`; arm switches can retain a prior arm; FP64 deep-copies the active model; edited Stage D loads another base. `_fd_check` allocates the reference before its cleanup `try`. **Fix:** derive the parameter count without a GPU load, manage model ownership, measure peaks.
8. **Non-blocking metadata gap — Stage D run config omits prompt template and seed** (`attribution_stages.py:527-530`; AGENTS.md).
9. **Question — integer-only numeric fields accept floats** (`attribution_chain.py:154-175`), e.g. `qualification.fd_rows: 1.5`, `inputs.max_new_tokens: 1.5`.

## Launch 2 and prior review context

At `753de57` the rehearsal YAML spells `19.5e+12`, which PyYAML parses as a float; `mistyped` on the actual draft reports no mismatches. Round 5 points 1–3 are largely avoided by a single supervised launch with no automatic restart, not universally fixed; point 4 (tokenizer binding) remains (finding 3). The 60 GB free-space preflight, the 80% allocated-memory rule and the approved tolerances are policy choices, not proven failures.

## Validation

Focused CPU tests passed (26 in backend + stages; 2 selected launch-record tests). No full-suite pass in the reviewer's environment (PyTorch library missing for the system interpreter).

## Fake versus real contract inventory (summary)

Every method on `Scripted`, `TinyBackend` and `Calls` was compared with the real `TransformersBackend` / `HFGenerator`. Shapes broadly match; the fakes do not exercise: real construction and source/tokenizer binding; preflight ordering; model residency (`parameter_count`, `untuned_generator`, `load`, `fp64_model`, edited Stage D); CUDA memory accounting (`device_memory`, `peak_memory`, `is_oom` are constants); real tokenization of rehearsal/cohort items; and real edit verification against gated logits (finding 5).

## Qualification: memory, thresholds and time

Nominal weights: ~6 GB per FP32 copy and ~12 GB per FP64 copy of a 1.5B model. Qualification holds untuned + current arm (+ transient FP64 copy, ~24 GB nominal), plus a possible extra arm during switches. `_probe_memory` measures peak allocated (not reserved) memory against total device bytes. Colab 06a measured 27.4 GB peak (FP32 attribution with the full-FP64 reference) and 4k fit / 8k OOM on 40 GB; neither determines the 80 GB sweep, invariance, generation or Stage D peak, nor whether qualification fits the 3,600 s cap (launch 2 had used 761 s when it failed).

**Smallest useful next smoke:** on CPU, the actual `Stages` entry point and actual `TransformersBackend` against a tiny random Qwen2+PEFT snapshot and tiny sealed fixtures, stubbing only CUDA accounting, volume and Hub I/O; cover preflight ordering, one FD arm, a short sweep with an OOM mock then a successful candidate, batch invariance, one generation candidate and one edited Stage D run.

# Response to review round 4 (implementation)

Disposition of `review-round-4.md` (Codex, reviewing `7be1df9`). All 11 points are adopted; I checked each against the code before adopting it. Every fix was made red/green on CPU: tiny random Qwen2 + PEFT 0.18.0, every external call mocked. At `1410e58`, 196 attribution and export tests pass. **No GPU was used.** Point 1 in particular needs the real GPU path in 06a. User decisions on 2026-11-14: point 9 option A (qualify only sizes actually exercised), point 8 two-phase summary.

1. **Adopt (`50d6245`).** Finite-difference gates are created on the input's device, and metric tensors go through `.detach().cpu()` before NumPy. The fixture oracle and the mutation helpers got the same treatment. CPU can't reproduce a device mismatch, so the regression test fails on the two bug patterns instead: `torch.full` without `device=`, and `.numpy()` on a CUDA-like tensor. The real check is 06a.
2. **Adopt (`3cf8537`).** Qualification is stored as completed only after the automatic sign-off succeeds. The sign-off records `qualification_sha256`. On every start, any completed qualification or later stage requires a sign-off by the rule, `accept`, whose hash matches the completed qualification; otherwise the run fails for investigation. Tests: the reviewer's failing-check-then-resume case, and a forged mismatched sign-off.
3. **Adopt (`aae09e8`).** `attribution_backend.bind_sources` runs before any work:
   - each arm's `adapter_model.safetensors` and `adapter_config.json` must equal that arm's *verified* step-64 inventory-v2 entry;
   - every pinned base-model file must match its hash;
   - the test parquet must equal the sequences-v2 manifest's.
   The result is recorded in the runtime. Tests: correct binding; swapped arm directories; other base weights; other test file; an unverified checkpoint; an arm absent from the inventory.
4. **Adopt (`3cf8537`).** `run_units` and `run_conditions` build the expected identity first and skip a sealed unit only if its stored identity is equal. A mismatch (plan, code, runtime or example IDs) or a corrupt unit is refused.
5. **Adopt (`f5dc3bb`, `3cf8537`).**
   - `restore_from_volume` runs at every start. It copies back each sealed unit and run directory the volume holds but local disk lacks, re-verifies each one, and never overwrites local sealed work. Test: crash in Stage C, delete the whole local root, resume to completion.
   - `edit_checkpoint` is retry-safe. It reuses a verified identical edit; an interrupted or different one is moved aside and rebuilt. The record is written last.
   - Stage D is sealed per (arm, condition, batch). The config is sealed first and must match; batches are skipped only when their sealed items match; the cap is checked between batches; only complete runs are analysed. Test: an interruption in Stage D followed by a resume regenerates exactly one batch fewer than the total.
   - Edited checkpoints, profiler staging and the rehearsal budget now live outside the export scope. The edit records (hashes) are sealed with Stage D, so exports are no longer made partial by them.
6. **Adopt (`3cf8537`).**
   - `Budget.close` charges the time since the last boundary however a stage ends.
   - A stage that ends over its cap is marked `cap-exceeded-at-completion`: before production it stops the chain; in production it skips to the report.
   - The total cap is checked before each stage starts.
   - Teardown time, including the export, is charged to `teardown`.
   - Stage D checks between batches (point 5).
   Tests: the reviewer's 200 s preflight under a 50 s cap; total-cap exhaustion before Stage D; restart persistence (existing).
7. **Adopt (`d217764`).** Each launch writes a fresh owner token before starting its chain and watchdog. The watchdog acts only while the file holds its own token, and re-checks it before signalling and again before copying or stopping. A superseded watchdog stands down. Test: the reviewer's scenario, run with the real script. The old watchdog exits "superseded", nothing is copied or stopped, and the replacement writer stays alive.
8. **Adopt with the user-approved two-phase summary (`3cf8537`).** Summaries live per attempt and phase under `attribution/summaries/<attempt>/{pre-export,final}`. The teardown order is:
   1. writers;
   2. sealed volume copy;
   3. **pre-export summary** sealed and copied to the volume (HF marked `pending`);
   4. HF export, which includes the pre-export summary;
   5. **final summary** with both outcomes, written to the volume, then a second small upload of only that summary (best effort, outcome in `stop_record.json`);
   6. stop request.
   `latest_summary` picks the newest attempt, then the latest phase, so a retry can never copy a stale summary. Tests: a fake-Hub clean restore through the real chain teardown returns the summary; a failed attempt followed by a successful retry leaves `attempt-002` as the latest, in state `complete`.
9. **Adopt, option A (`1410e58`).**
   - **Generation batch:** a candidate larger than the rehearsal item count is recorded `not-exercised` and can't be selected. With the proposed 20 items, 32 and 64 are therefore never qualified. An OOM fails that candidate and ends the sweep. Actual rows, padded tokens and peak memory are recorded.
   - **Token budget:** the probe replicates the longest calibration row to fill each candidate budget, and actual rows, width and padded tokens are recorded.
   - **Before production:** the selected budget is re-measured on the real worst-case discovery batches (most padded tokens; longest rows). An OOM or a peak over the memory fraction stops the chain before production.
10. **Adopt (`c9f7de8`).**
    - R is unavailable when the full-removal effect is below the approved resolution floor (`denominator-below-resolution-floor`). The floor is now a required argument, evaluated from qualification before `build`.
    - Primary 1's headline is the family-adjusted inference. It says `supported` only if the interval rule supports it *and* Holm rejects primary 1 in every family it belongs to. The raw interval outcome is kept as `outcome_unadjusted`, alongside the per-family Holm results and an adjustment note.
11. **Adopt (`f5dc3bb`, `50d6245`).** `stage_d_generator` takes the gate/checkpoint probe prompt from the caller (`probe_ids`) and no longer reads the cohort. The 06a driver passes a rehearsal item. In production the probe is the first Stage D item, after the freeze. The 06a driver test now uses a backend that raises on any discovery, held-out or cohort access, and asserts that every probe is a rehearsal prompt.

## Not established by this response

- No GPU, Colab or Pod run. Point 1's real verification is 06a.
- The watchdog's `timeout` dependency is GNU coreutils, present on the Linux Pod; macOS lacks it, as the review notes.
- The repository-wide suite was last run file by file at `7be1df9`, with 8 environment-only failures (missing IPython, shallow clone). It was not re-run here.

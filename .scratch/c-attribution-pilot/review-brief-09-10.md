# Brief: pre-GPU implementation review of C-attribution tickets 09–10

Branch `c-attribution-pilot` at `0be77cf` (pushed). Read-only review; no GPU, no Pod, no pushes.
Write one ranked findings file: `.scratch/c-attribution-pilot/review-implementation-09-10.md`
(P0/P1/P2/note; each with file:line, a concrete failure scenario, and a proposed test).

## What to review
- `pilot_eval/c_attribution_chain.py` — typed launch record (schema, seal, refusals), per-run volume subtree,
  stage runner, resume (a stage is skipped only if its outputs re-verify by hash), StopBeforeProduction vs
  analysis outcomes, teardown order (copy → pre-export summary → export → final summary → stop).
- `pilot_eval/c_attribution_stages.py` — the seven stage bodies; that each reads only sealed inputs and
  binds to the launch-record hash.
- `pilot_eval/c_attribution_backend.py` — local-files-only loading, adapter hash + live adapter check,
  preflight (disk, CUDA, HF_HOME off the volume, HF_HUB_OFFLINE, volume identity, private repo, catalog).
- `scripts/c_attribution_launch.sh`, `scripts/attribution_watchdog.sh` (deadline `none`, progress status).
- `tests/test_c_attribution_chain.py`, `test_c_attribution_launcher.py`,
  `c_attribution_realpath_fixture.py`, `test_c_attribution_realpath.py` — do they test what they claim?
- The fix pass `0be77cf` against `review-implementation-01-06.md` / `-response.md`.

## Questions I most want answered
1. Can any path spend GPU time on an unsealed, mismatched or stale input (record, bundle, adapter, base model,
   rows, freeze)?
2. Can resume silently reuse output produced under a different record or code commit?
3. Can an integrity failure be reported as an analysis outcome (or the reverse) and let the chain continue?
4. Can teardown lose results (stop before copy/export completes, or export a partial run as complete)?
5. Does the CPU real-path test exercise the same code the Pod runs, or does a stub hide a real-path difference?

## Known and out of scope
- Qualification check (b), sampled-batch replay, is recorded as not run.
- Rehearsal record values tagged "[for approval]" await the user's approval (ticket 11).
- 5 macOS-only test failures (old watchdog/launcher tests need `timeout`/`flock`).

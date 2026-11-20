# Response: C-attribution implementation review of tickets 06–10

Review: `review-implementation-09-10.md` (Codex, at `0be77cf`). All seven P1s and the P2 are accepted and
fixed in one pass, each with a test written first and seen failing. No tolerance or gate changed. CPU only:
nothing here is verified on a GPU or a Pod.

| # | Finding | Fix | Test |
|---|---|---|---|
| P1 | IEEE FP32 not set or recorded | The backend calls the existing `set_ieee_precision()` (CUDA matmul, cuDNN, mkldnn = `ieee`, matmul `highest`) when it is built, before any model work. `runtime()` records the readback; preflight reads it again and refuses any backend that is not `ieee` (an unreadable mixed-API state is a problem too). This restores what the earlier attribution and neuron-audit backends did. | `test_backend_sets_reads_back_and_records_ieee_precision` (starts from TF32 everywhere; a later change is refused) |
| P1 | FP64 reference kept alive | Stages drop every owning reference (`reference`, `model`, `inst`, `out`) before `backend.release()`, which now only collects and empties the cache; the FD call is in `try/finally`. | `test_the_fp64_reference_and_each_arms_model_are_freed_before_the_next_measurement` (weakrefs: no FP64 copy at the peak-memory reset, no earlier model at each load, none after attribution) |
| P1 | Base-model hash self-attesting | `PINNED_BASE_MODEL` fixes HF's own per-file identities for revision `989aa798` in the code: the LFS sha256 of `model.safetensors` (`dd924a11…`) and the git blob id of `config.json` (`f81ead14…`), from huggingface_hub's download metadata; the local files match both. `build` refuses files that differ from them; `check_base_model` checks the code's pins (never the bundle's) and refuses a bundle built under other pins. The CPU world substitutes its own pins through a backend probe. | `test_a_self_consistent_snapshot_cannot_be_labelled_the_pinned_revision`, `test_the_local_pinned_model_matches_hf_identities` (hashes the real 3 GB file), `test_the_pod_preflight_checks_the_base_model_against_the_codes_pins` |
| P1 | Work only on local disk until teardown; `done` after any end | Every unit boundary and every finished stage mirrors newly sealed runs to the volume (incremental: a run whose seal is already there is skipped) and records the result in `progress.json`. Teardown writes `last_attempt_end.json` for every end and `done` only for a complete run, so a failed or stopped attempt resumes under the same run ID. | `test_sealed_work_reaches_the_volume_at_each_boundary_not_only_at_teardown` (checks the volume from inside the stage, before any teardown), `test_a_handled_failure_ends_the_attempt_not_the_run_and_resumes` |
| P1 | Resume trusted old preflight and qualification | Preflight runs on every attempt (sealed per attempt) and returns an identity: runtime (versions, device name, precision readback, tokenizer), bundle hash, base-model hashes, code commit. Qualification stores the identity it ran under; a resume under a different one stops at preflight, naming the differing fields. **Decision taken:** refuse rather than requalify, because requalifying would mix units from two runtimes in one run; a new run ID requalifies. | `test_resume_reruns_preflight_and_refuses_qualification_from_another_runtime`; the real-path resume test passes with matching identities |
| P1 | Timed-out copy alive at the stop; coverage overstated | The teardown volume copy and the summary copies run in a spawned child process that is terminated, killed and reaped on timeout (`in_child`), so they have ended before anything later runs. The real exporter's Hub calls already run in child processes; they now get `export_attempt` as their timeout, and a still-running export thread is listed in `teardown_io_running`. Backups now report verified bytes and coverage separately: a backup is "complete" only if verified, every expected run is present, and the run itself completed. `copy_sealed` no longer counts as verified when it reported problems. | `test_teardown_ends_a_timed_out_copy_before_requesting_the_stop` (the copy's PID is dead when the stop runs), `test_an_empty_copy_with_missing_expected_work_is_not_a_complete_backup`, `test_a_failed_run_reports_which_stages_never_produced_outputs` |
| P1 | Pod ID not checked | `precheck` and the chain refuse a launch unless `RUNPOD_POD_ID` equals the record's `pod_id`. This is checked first, so the volume-identity failure still stops the right Pod. If the Pod is not confirmed, teardown does not run either stop command and says to stop the Pod by hand. The launcher's precheck already runs before it takes watchdog ownership. | `test_the_record_pod_id_must_be_the_pod_running_the_launch` |
| P2 | Missing results; arrays as strings | After selection, `bootstrap.json` gains `selected`: an interval for every selected read and its matched control (group, role, layer, index), on the same resample counts. The report includes the untuned row, paired with each arm's intact run, when `untuned` is in scope; otherwise it says why. Sealed JSON now writes NumPy arrays and scalars as JSON numbers (lossless) and refuses unknown objects instead of `str()`-ing them. | `test_numpy_values_round_trip_through_a_sealed_run_as_numbers`, `test_an_unknown_object_is_refused_rather_than_stringified`, the whole-chain test (numeric selected and per-layer intervals), `test_the_untuned_row_is_reported_alongside_when_in_scope` |

## Changed by design in existing tests
- The resume tests now expect `preflight` to run again on every attempt.
- The crash test expects "verified bytes, partial", not "verified backups: volume copy", for a failed run.
- Old tests that deleted `done` to simulate a lost Pod now tolerate its absence (failed runs no longer write it).

## Shared code
- `attribution_units._encode` (`default=str` → NumPy-aware, refusing unknown objects) is shared with the
  neuron-attribution pipeline; its tests were rerun.
- `attribution_chain.copy_sealed` gained an opt-in `skip_present`; default behaviour is unchanged.
- The neuron-attribution chain still uses the thread-based `call_with_timeout` around its copy, so it still has
  the timed-out-copy problem. It is not fixed here.

## For the user
- The rehearsal record's generation conditions (`intact`, `top10-adapters`, `global64-reads`) leave out
  `untuned`, so the rehearsal will not exercise the untuned row unless you add it.
- Not verified until the rehearsal: actual IEEE settings on the A100, peak memory after the release fix, a real
  copy/export timeout, and a real Pod stop.

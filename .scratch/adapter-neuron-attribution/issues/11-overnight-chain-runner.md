# 11: Overnight chain runner

**Status:** resolved (2026-11-13) — CPU implementation approved by the user 2026-11-13 (overnight requirement, spec §Overnight unattended chain). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 05

**What to build:** `pilot_eval/attribution_chain.py` (plus a launcher script) that runs preflight → qualification → base traces → attribution and freeze → Stage C → Stage D → report → volume copy → HF export → Pod stop unattended from a sealed launch record, reusing `neuron_audit_pod` patterns.

- [ ] Launch record: loads and verifies a sealed record (all fixed values, caps, Pod ID, stop command, code commit); refuses to start if any value is still `[proposed]` or null.
- [ ] Automatic sign-off after qualification by the record's rule; stops before production on any listed failure; never stops on analysis outcomes.
- [ ] Per-stage and total hard caps at unit boundaries with persisted elapsed budget; projections advisory only; safe-boundary stop with verified partial export.
- [ ] End of run on every path (success, failure, cap, exception), in order: stop and join writers; copy sealed, hash-verified units to the volume (timeout); HF export with readback (3 attempts, timeouts); finalize the sealed morning summary including both backup outcomes (never label unverified as verified); write it to the volume; request the Pod stop with both command forms (timeouts), recording requested vs externally verified. Do not reuse `may_stop_pod` for this decision (it requires a verified HF restore; the user's policy stops regardless).
- [ ] Independent watchdog (shell, outside the chain's Python), timed from the persisted remaining cap + margin: signals the writer, waits a bounded time, copies only sealed verified units (timeout), writes a watchdog summary, requests the stop (timeouts); cancelled cleanly on normal completion.
- [ ] Resume: only when this run's previous worker is verifiably gone and the volume identity and sealed units match the launch record; then take over its own marker. Live writer, foreign run ID or ambiguity → stop for investigation. The marker is advisory, not a distributed lock; never set `resume_own=True` blindly.
- [ ] Rehearsal mode validates against its own labelled two-arm expected set; production refuses missing arms.
- [ ] Tests (CPU, all external calls mocked): every exit path ends in writers joined → volume copy → export → sealed summary → stop requested; HF failure still stops; hung copy and hung stop command time out; neither backup verified is reported as such; killed-worker stale marker resumes, live writer and ambiguous marker stop; unavailable matched group is skipped, not fatal; longest-first packing does not stop a healthy run; qualification failure stops before production; analysis failure does not stop; overrun ends at a unit boundary; watchdog fires on a hung stage; stop-command fallback; launch record with a null value refused; resume skips sealed units.

## Comments

### 2026-11-13 — launch-record fields added (implementing agent)

The code needs these values, and the spec and candidate plan don't give them. The launch record must state each one, and the code refuses a record that leaves one out. I have not chosen any of them:

- `marker_stale_seconds`: how old a heartbeat must be before a dead worker's marker may be taken over.
- `timeouts_seconds` (`copy`, `export_attempt`, `stop_attempt`, `writer_join`); the spec requires each step to have a timeout but gives no values.
- `hourly_cost_usd`, for the summary's cost estimate.
- `qualification`: `calibration_minimum`, `fd_arms`, `fd_rows`, `fd_adapters` (spec: 5), `fd_neurons` (spec: 3), `head_chunk`, `memory_fraction` (spec: 0.80 [proposed]).
- `inputs.rehearsal_generation_ids`: the 20 test indices, pending the draw-rule question on ticket 01.
- `selection.random_seeds` and `statistics.null_seed`, pending the question on ticket 04.

## Answer

2026-11-13 — `pilot_eval/attribution_chain.py`, `pilot_eval/attribution_stages.py`, `pilot_eval/attribution_backend.py`, `scripts/attribution_watchdog.sh`, `scripts/attribution_overnight.sh`. Tests (CPU, every external call mocked): `tests/test_attribution_chain.py` (28), `tests/test_attribution_stages.py` (2), `tests/test_attribution_backend.py` (2).

- **Launch record.**
  - The record is sealed by a canonical-JSON SHA-256 in `launch_record.seal.json` and must match the current code commit. Any null, empty or `[proposed]` value, or a missing required key or stage cap, refuses the launch.
  - Production must expect all six arms. A rehearsal needs a `rehearsal*` label and validates against its own labelled arm set.
  - Stop commands are built from the record's templates with the Pod ID checked.
- **Stages.** `attribution_stages` builds the seven stages from tickets 01–05 and 10 behind a backend interface. `TransformersBackend` reuses the joint audit's loader, `ChatTokenizer`, IEEE precision and `check_live_adapter`. Stage state lives in sealed run directories and units, so a resume rebuilds it from disk.
  - Qualification runs on calibration rows and the rehearsal items only: arm routing, finite differences, batch invariance at two budgets, token-budget sweep (OOM fails only that candidate), generation batch qualification (longest prompts first).
  - Edited Stage D checkpoints are verified by hash and by bit-identical logits against the g = 0 gate.
  - An end-to-end CPU test runs the real stages, including the proposed FD and batch-invariance tolerances, through a full two-arm rehearsal chain on tiny Qwen2 arms. Only the untuned trace generator is scripted, because a random model cannot solve GSM8K.
- **Sign-off and stops.** The automatic sign-off is recorded as the rule (`AUTO_REVIEWER`), never a person. A failed check, review flag, missing arm, unmet minimum, no feasible budget or preflight problem stops before production. Analysis outcomes (gate failures, skipped conditions, inconclusive primaries) never stop the run.
- **Caps.** `Budget` is persisted on the volume, and `ctx.boundary()` is called between units. A cap ends that stage at the boundary; the chain skips to the report, then tears down. A resumed run continues the persisted elapsed time. Projections are logged as advisory, and a longest-first fixture runs to completion.
- **Teardown on every path** (success, pre-production stop, cap, exception):
  1. Stop and join writers.
  2. Copy only sealed, hash-verified units (`mirror_store`) and sealed run directories to the volume and re-verify (`copy_sealed`), under a timeout.
  3. HF export with fresh readback: 3 attempts, each with a timeout.
  4. Seal the morning summary with both backup outcomes. "NO VERIFIED BACKUP; possibly missing outputs" is written when neither verified, and nothing is labelled verified without verification.
  5. Copy the summary to the volume.
  6. Request the stop with each command form under a timeout. The record is `requested` vs `externally_verified: False`, plus a `stop_record.json`.
  `may_stop_pod` is not used. Tests cover HF failure, a hung copy, a hung stop, the fallback command and writer join order.
- **Resume** (`resume_decision`): take over only when the marker is this run's, on this host, its PID is not running and its heartbeat is older than `marker_stale_seconds`, and the volume identity matches. A live writer, foreign run ID, other host, fresh heartbeat or unreadable marker stops for investigation, and the chain then never tears down the other writer's run. Sealed units are skipped (`run_units`, `run_conditions`). A crash in Stage C followed by a rerun completes and reuses the sealed units.
- **Watchdog** (`scripts/attribution_watchdog.sh`) runs outside the chain's Python, launched in its own session by `scripts/attribution_overnight.sh`. Its deadline is the persisted remaining total cap plus a margin (`watchdog_deadline`). On firing it:
  1. sends SIGTERM to the writer, waits a bounded time, then SIGKILL;
  2. copies only sealed units via `attribution_chain watchdog-copy`, under a timeout;
  3. writes the watchdog summary;
  4. requests the stop with both forms, each under a timeout;
  5. rewrites the summary with the outcome.
  It exits cleanly when the chain writes its done file. Both behaviours are tested by running the real script.
- **Not covered on CPU.** The CUDA loader, NVML sampling inside stages, the real HF upload and a real Pod stop. These are 06a and the attended Pod rehearsal.

### 2026-11-13 — CPU wiring for 06b–09 (implementing agent)

- **Every stage runs under `attribution_gpu.StageMonitor`.** It records NVML busy/memory/power at 1 Hz (≥ 30 s needed for a summary), synchronized wall time, unpadded and padded tokens, padding fraction, an MFU estimate and a profiler trace of the stage's first unit only. Each attempt is sealed under `attribution/compute/<stage>-<ns>/`, and the report's compute appendix shows the latest attempt per stage.
  - The MFU numerator uses (forward, backward) passes per stage: attribution (2, 2), Stage C (2, 0), generation (1, 0), qualification (1, 2).
  - On the Pod, NVML needs `nvidia-ml-py` in the venv. Without it the record says "unavailable".
- **The report now includes:**
  - neuron concentration and cancellation at each frozen top `down_proj`, with the random-order null; the resolution floor is the record's rule evaluated on 06b's measured batch-invariance discrepancy;
  - a vs a₀ quoted raw from the init check's `rows.jsonl`, if `inputs.init_check_rows` is given;
  - Stage D outcomes per run (accuracy, mean tokens, capped, invalid, early-stop rate);
  - the item-level comparison of the full-arm runs with saved pilot results. A missing or mismatched pilot file is reported as unavailable, never fatal.
- **New required launch-record fields:**
  - `compute.fp32_peak_flops`: the measured device's dense FP32 peak, the MFU denominator.
  - `stage_d.early_stop_rule`, in the form `{kind: quantile-of-untuned-eos-tokens, quantile: q}`.
  - `stage_d.pilot_results`: saved pilot responses for gold-sft and online-grpo; required in production.

### 2026-11-13 — question for the user (implementing agent)

- **Early-stop definition.** The spec says the early-stop rate uses "the threshold recomputed from this study's untuned run" but never defines early stop, and no ADR or earlier pilot does. I implemented one candidate form: a response counts as an early stop when it ends with EOS before the q-th quantile of the untuned run's EOS-terminated response lengths; capped responses never count. q is left empty. Please confirm the form and give q, or give another definition.
- **Pilot result files.** Which saved pilot response files are the comparison references for gold-sft and online-grpo (step 64)? Each row needs `id` = `gsm8k:test:<index>` and a `correct` flag (or `score.correct`).

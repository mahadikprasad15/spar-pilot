# 04: Run resumable multi-arm audit with measured head option

**Status:** needs-triage — draft for adversarial review

**What to build:** After successful GPU qualification and once partial export is available, run the frozen full Phase 1A collection with progress, durable bounded work units and validated resume. Include only the previously measured and approved head scope.

**Blocked by:** 03b, 07.

- [ ] Use FP32 weights/forwards and reductions with TF32 and autocast off; verify each process's interpreter, packages, actual cache/offline load, runtime and frozen batch/mask hashes before work.
- [ ] Execute identical example order, membership and padding in every arm; show arm, unit, completed/total, timestamp and stale-progress warning. An SSH disconnect is not a failed job.
- [ ] Write and seal resumable immutable dense blocks and per-token read data; reject changed sources, corrupt units or unsafe concurrent writers. Revalidate sources once per session unless their immutable contract is proven; do not repeatedly rehash history.
- [ ] Enforce one Pod, one tmux session and one writer via operator runbook and advisory PID/host/heartbeat marker; stale or ambiguous marker requires investigation. Do not describe it as a reliable distributed lock. Validate output/temp storage operations and preserve failure evidence. Omitted or subset head coverage is explicit.

## Comments

### 2026-11-11 — runner built and CPU-tested; not yet run on a GPU (Claude)

- User decisions (2026-11-11):
  - sl-0 is a Runpod **Global Volume** (47.7 GB stored), with no fixed size;
  - per-arm upload and verification to the existing private HF dataset under `neuron-audit/`;
  - a reference-only final manifest;
  - stop the Pod only after the required final backup is verified;
  - Colab for bug-finding rehearsals, then the full run on the Pod.
- Storage facts checked 2026-11-11:
  - Runpod Global Volume docs: grows with data and is billed per GB used plus requests; no atomic rename, locks, hard links or chmod; listings cached about a minute; beta.
  - HF docs: free tier has 100 GB private storage, files under 200 GB recommended (500 GB hard), fewer than 100 files per commit.
  - Measured via the API: 15.84 GB private, current revisions only.
  - About 55 GB of FP32 arm packages gives about 71 GB total, within the plan's 85% rule; a second full run would not fit without PRO or pruning.
- `pilot_eval/neuron_audit_production.py`:
  - **Preconditions:** plan loads for 04 (wall cap recorded); the plan's frozen core equals the one 03b ran under (only cap/approval may differ); qualification complete, rehearsal-matched, with an *accept* sign-off bound to the current files; batch manifest hash; discovery/held-out rows only.
  - **Preflight:** runs in the worker before any work.
  - **Work:** arm by arm on local disk, one sealed unit per (arm, frozen batch); identities on the first 2 batches; every unit mirrored to the volume (`mirror_store`: copyfile, seal last, verified from explicit paths, torn copies redone).
  - **State:** progress and writer marker written directly (no rename) to local disk and the volume. A foreign marker refuses with "investigate"; there is no takeover.
  - **Per arm:** deterministic package → upload → fresh-download sha256 verification → local prune.
  - **Finish:** the final manifest references the packages and is read back. `stop_command` runs only if `may_stop_pod_final` holds and the run completed; otherwise the Pod is left running with an event.
  - **Failures:** a partial final manifest is attempted, and the Pod is never stopped.
  - **Overrun:** checkpoints every 10%, two consecutive forecasts above 2× projection or the cap stop the run, with a partial export. The forecast uses collection time only (export/mirror time would otherwise trigger false stops); the hard cap covers total wall time. **[proposed]**
- Tests (10): full run, crash and resume (the finished arm is not recollected), foreign marker, overrun stop, sign-off missing/rejected, plan-core change, rehearsal limits, and the Colab rehearsal flow (labels, `rehearsal-<label>` HF paths, selection refusal, production refusing a rehearsal qualification). 141 audit tests pass.
- **Not verified:** real GPU, real HF throughput for multi-GB packages, Global Volume behaviour (fsync, rename) beyond the documented limits.
- Colab rehearsal recipe:
  1. `python -c "from pilot_eval import neuron_audit_plan as p; p.make_rehearsal_plan('frozen_plan_v1.yaml','rehearsal_plan.yaml')"`.
  2. `python -m pilot_eval.neuron_audit_qualify run --rehearsal --plan rehearsal_plan.yaml --config <colab.json>`, with `run_id` `rehearsal-…`.
  3. `python -m pilot_eval.neuron_audit_qualify signoff …`.
  4. `python -m pilot_eval.neuron_audit_production --rehearsal --plan rehearsal_plan.yaml --config <colab.json>`, with `max_batches_per_arm` and `rehearsal_cap_hours` in the config.

### 2026-11-11 — Colab rehearsal r1 findings (Claude)

- Rehearsal 04 (1 batch/arm) completed: every arm and the 03b package verified on HF, final manifest complete. Rehearsals can now choose covering frozen batches (`rehearsal_batch_choice: cover`, `558e77e`). Lessons: `docs/agents/experiment-lessons.md`.

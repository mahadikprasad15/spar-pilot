# Sampling SFT: budgeted jobs and RunPod handoff

The CLI and `scripts/run_sampling_sft_job.py` are the primary execution path.
The notebook is an optional client of the same commands. Ticket 02 implements
lifecycle controls. Ticket 03 adds bounded sampling, score comparison and
throughput candidate workflows; their real GPU qualification is still pending.
Nothing here provisions a Pod, deletes a volume, or grants paid-job approval.

## One authoritative budget

Use the same canonical artifact root across attempts. Keep its budget directory
on the persistent storage used by subsequent stages; never allocate work from
disconnected copies of the ledger. A downloaded job bundle is an export, not a
second budget authority.

```sh
python -m pilot_eval sampling-sft-budget-init --config artifacts/plans/sampling-sft-v1/sampling-sft.plan.json --output-root artifacts
python -m pilot_eval sampling-sft-budget-status --output-root artifacts
```

The authoritative budget is `artifacts/runs/sampling-sft/budget/budget.json`.
It is an atomic snapshot with a checked transaction chain and a separate origin
identity. Missing or changed ledgers cannot be reset through initialization.
Amounts use integer microdollars internally. The fixed limits are $25 total,
$10 sampling, $6 training, and $9 storage/genuinely shared overhead.

GPU reservations are rate × (estimated hours × 1.5 + separately counted setup
hours). Estimated hours exclude the separately supplied setup time. Count idle,
debugging and reruns against the stage served. Use actual provider quotes before
launch, not the design's illustrative rates. Existing or proposed commitments
at 80% require an explicit review bound to the current budget and request hashes;
review never raises a hard cap. Reservations alone do not authorize paid work.

## CPU-only lifecycle smoke

This no-op produces a tiny local result and two simulated copy receipts. It
makes no provider call, loads no model and incurs no cloud charge. Its
`simulation` flag cannot pass as a live downloaded-copy receipt.

After preparing the source plan, save the following request in its plan directory:

```json
{
  "job_id": "lifecycle-demo",
  "kind": "noop",
  "stage": "sampling",
  "mode": "noop",
  "command": [],
  "outputs": ["runs/sampling-sft/sampling-sft-v1/lifecycle-demo/result.json"]
}
```

Run from the repository checkout:

```sh
python -m pilot_eval sampling-sft-job-prepare --config artifacts/plans/sampling-sft-v1/sampling-sft.plan.json --request artifacts/plans/sampling-sft-v1/demo-request.json --output-root artifacts
python -m pilot_eval sampling-sft-job-run --job artifacts/runs/sampling-sft/sampling-sft-v1/jobs/lifecycle-demo/job.plan.json --output-root artifacts
python -m pilot_eval sampling-sft-job-export --job artifacts/runs/sampling-sft/sampling-sft-v1/jobs/lifecycle-demo/job.plan.json --copy-root artifacts/exports/sampling-sft-demo/network --role network --receipt artifacts/exports/sampling-sft-demo/network-receipt.json --output-root artifacts
python -m pilot_eval sampling-sft-job-export --job artifacts/runs/sampling-sft/sampling-sft-v1/jobs/lifecycle-demo/job.plan.json --copy-root artifacts/exports/sampling-sft-demo/downloaded --role downloaded --receipt artifacts/exports/sampling-sft-demo/download-receipt.json --output-root artifacts
python scripts/run_sampling_sft_job.py --job artifacts/runs/sampling-sft/sampling-sft-v1/jobs/lifecycle-demo/job.plan.json --network-root artifacts/exports/sampling-sft-demo/network --download-receipt artifacts/exports/sampling-sft-demo/download-receipt.json --max-wait-seconds 0 --output-root artifacts
```

The final command reuses sealed work and ends with `noop-completed`; it does not
report a stopped Pod. Job preparation binds the request, source plan and
implementation hashes. Repeated compatible completed work is reused. A changed
request needs a new job ID, and new source/code identities need a new attempt.

## Paid-job boundary

Before the first spend, complete the source preflight and obtain the user's
explicit go-ahead. Reserve the reviewed estimate before renting/launching work;
Pod provisioning itself is outside this implementation. Training additionally
requires the later data-gate approval. Full generation, eligibility and training
remain later stages. Bounded engine commands are documented in
[sampling-sft-engine.md](sampling-sft-engine.md).

A live request has kind `gpu`, mode `runpod`, the served stage, actual quoted
`rate_usd_per_hour`, `estimated_hours`, `setup_hours`, the attached
`network_volume_id`, an explicit argv `command`, and declared output files in
this plan's canonical run tree. It may not use the shared-overhead line for GPU
work. Review command arguments; use environment variables for credentials.
Live commands are limited to reviewed Python module entrypoints for the project scientific CLI or pytest; direct shells, Python -c, cloud-management commands and lifecycle recursion are refused. This is not a sandbox for malicious code inside an approved module. Training launches remain blocked until Ticket09 integrates the post-gate approval contract.

The paid-job approval names reviewer/notes, `request_sha256`,
`source_audit_sha256`, and explicit true values for `paid_execution_approved` and
`preflight_reviewed`. An 80% review also binds `budget_sha256` and
`review_80_percent`. The software does not manufacture these approvals.

The worker checks its own `RUNPOD_POD_ID`, provider identity, attached volume,
and provider hourly price against the reserved rate before launching the child.
The source audit must be ready. The command runs without a shell, in a separate
owned process group with a bounded timeout, so inference children are not left
behind when the command ends. Startup failure, timeout and nonzero exit retain
logs and an explicit failed-work result rather than becoming successful data.
A recorded but unfinished launch is uncertain and never auto-relaunched.

The installed RunPod CLI schema/permissions and physical volume mount still need
real preflight verification. The adapter uses the documented
[`pod get` and `pod stop` commands](https://github.com/runpod/runpodctl#pod-management),
filters provider metadata, and implements no create/delete/terminate action.
An unsupported schema, missing rate or failed provider call blocks rather than
silently changing API or deleting a resource.

## Export and shutdown handshake

1. The worker seals an export manifest containing required payload hashes, its
   execution receipt and logs. A failed command can still export its diagnostic
   bundle; this does not turn failed scientific work into successful work.
2. The worker exports to the network-volume artifact root and verifies every
   file. If the artifact root is already on the verified network volume, a
   network receipt can verify it in place without another full copy. Verify the
   physical mount during real preflight; path names alone are not volume proof.
3. An external client downloads the bundle and runs the CPU receipt command on
   the actual downloaded files. Use the copied manifest's path, the local copy
   root, role `downloaded`, and a new receipt path. The client must have the same
   tested code revision. A live download receipt cannot be issued by the worker
   host; no-op receipts are marked simulated.
4. Transfer that receipt back to the worker/control path. Receipts are explicit
   operator handoffs, not authentication or magical remote file inspection. The
   worker validates bindings and rechecks the accessible network copy. Network
   and downloaded receipts cannot describe the same copy.
5. The run script waits only within its declared bound. Once both receipts are
   available, it persists stop intent to the network control record and stops
   its own Pod as the last external mutation. It never labels that call as
   confirmed shutdown. Missing receipts or stop failure remain blocked with
   reservations held and an immediate operator-action message. There is no
   automatic destructive fallback. The code cannot guarantee costs while a
   provider is unreachable or an operator leaves a blocked Pod running.
6. From the external controller, run `sampling-sft-job-confirm` against the
   copied job plan. It queries provider status and writes a shutdown receipt
   only for the same Pod reporting `EXITED`. It records whether local stop-intent
   evidence was available. This provider observation is distinct from billing.

Example client receipt, with paths chosen for the downloaded tree:

```sh
python -m pilot_eval sampling-sft-job-receipt --manifest DOWNLOADED_ROOT/runs/sampling-sft/PLAN/jobs/JOB/export-manifest.json --copy-root DOWNLOADED_ROOT --role downloaded --receipt DOWNLOADED_ROOT/downloaded-receipt.json --output-root DOWNLOADED_ROOT
```

`sampling-sft-job-finish` accepts the job plan and both receipt paths when doing
that handoff manually. `sampling-sft-job-execute` (or the script wrapper) runs
the integrated work/network-export/bounded-ack/stop path. No transport endpoint,
SSH host, cloud key or download location is guessed for the user.

## Reconcile actual charges

Use evidenced observations with stable event IDs, actual USD, reviewer and notes.
`sampling-sft-budget-observe` records cumulative interim GPU charges while
retaining the outstanding reservation. This is conservative double coverage of
uncertain remaining time, not a second actual charge.

`sampling-sft-budget-settle` replaces a GPU reservation with its final observed
cost. A launched RunPod job also needs its bound external `shutdown_receipt`
under the authoritative artifact root. Transfer that receipt to the primary
ledger; do not start a new budget in the downloaded copy. Final costs are never
invented from elapsed-time estimates. A real overrun is recorded and blocks
new work instead of rejecting the bill to make totals look green.

Storage requests name the volume, quoted monthly rate and expected months.
Partial storage observations include `covered_months`; they add actual charges
and retain the forecast for the remaining reserved lifetime. Extend storage
before retaining the volume beyond its reserved period. Repeating a request is
idempotent; add a distinct extension request ID when intentionally extending again.
Stopping a Pod does not release the storage reservation or imply a deleted volume. Multiple forecasts may be reserved, but another GPU launch is blocked while a prior launched GPU job has unreconciled cost. The ledger exposes those job IDs; closing the prior allocation does not mean deleting its data.

Deletion of network volumes, or termination of an unexported Pod, still requires
explicit approval and is not automated here. Actual provider billing and
multi-machine storage/locking behavior remain real-environment checks.

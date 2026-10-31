# Sampling SFT: full generation

Ticket 05 collects k0/k2/k5 for the original512 training problems. Its outputs
are trace attempts and automatic per-problem dispositions. Human eligibility
review, the comparative data gate and training follow in later tickets.

## Prepare on CPU

Use the saved [pilot](sampling-sft-pilot.md) configuration:

```sh
python -m pilot_eval sampling-sft-generation-prepare --pilot plans/sampling-pilot-v1/sampling-sft.pilot.json --name sampling-generation-v1 --output-root artifacts
python -m pilot_eval sampling-sft-generation-status --config plans/sampling-generation-v1/sampling-sft.generation.json --output-root artifacts
```

Preparation freezes all512 original source IDs, stored scorer-prompt IDs, gold
texts, proposer/algorithm identity, runtime and code hashes. The three arms share
one runtime. It creates no model or job. A plan prepared before the pilot report
exists remains unqualified; use a new name after adding that report, rather than
silently changing the earlier plan's parents.

## Review before Stage2

Real generation requires the actual Stage0.5 worker/profile/T3 selection, real
pilot kill/restart replay and the pilot alarm report. The collector revalidates
those products. Any failed replay, score check, source/runtime mismatch or unfinished-attempt
threshold blocks model loading. Under the explicitly recorded ADR 0022 policy,
initial correctness is descriptive and an authorized budget waiver is recorded
as waived; the historical pilot report is not rewritten. CPU fixtures cannot satisfy the real gates.

`generation-status` reports `plan_sha256` and `pilot_report_sha256`. Record a
review JSON with those values and the decision:

```json
{
  "generation_approved": true,
  "alarm_policy": "descriptive-correctness-v1",
  "waive_budget_limits": true,
  "plan_sha256": "COPY_FROM_STATUS",
  "pilot_report_sha256": "COPY_FROM_STATUS",
  "reviewer": "REVIEWER",
  "notes": "Measured pilot and Stage2 forecast reviewed."
}
```

```sh
python -m pilot_eval sampling-sft-generation-freeze --config plans/sampling-generation-v1/sampling-sft.generation.json --review artifacts/plans/sampling-generation-v1/review.json --output-root artifacts
```

This writes `generation.review.json`, bound to the exact plan and report, with
the observed correctness and the explicit waived budget state. An
unchanged decision is reused after later ledger updates; it does not create a
new reservation. Each paid job still needs Ticket02's current quote, conservative
reservation, required review and request-bound paid-job approval. The explicit user waiver
must also be bound to the current ledger and each job request; accounting and
the recorded caps remain visible. Unknown
prior job cost remains reserved until reconciled.

## Run through the budgeted job entrypoint

Use `scripts/run_sampling_sft_job.py` and the
[RunPod lifecycle](sampling-sft-runpod.md). The frozen request's child command
is an argv list such as:

```json
["python", "-m", "pilot_eval", "sampling-sft-generation-run",
 "--config", "plans/sampling-generation-v1/sampling-sft.generation.json",
 "--k", "0",
 "--job", "runs/sampling-sft/sampling-sft-v1/jobs/generation-k0/job.plan.json",
 "--output-root", "artifacts"]
```

Repeat with k2/k5 and their separately approved jobs. Config/job paths are
relative to the artifact root or absolute; the root itself is relative to the
checkout or absolute. Source and arm must match the approved job. Direct real
inference without its active reserved job is refused before engine import.

Keep the canonical artifact root on the verified persistent network volume.
The export inventory and downloaded copy must cover raw round/block evidence,
responses and metadata as well as the final tables. The job lifecycle exports
declared files; it does not discover arbitrary undeclared output folders.
The later verified-handoff ticket packages the experiment's complete products.
Pod release follows verified export; no volume deletion is implemented here.

## Attempt and retry model

Each arm starts all512 problems at data seed0. Each logical engine request also
records its problem, block, refinement step and deterministic request seed.
Generation/scoring batches retain fixed problem order. The proposer is
gold-conditioned, while the scorer uses the original training prompt.

One complete trace attempt ends as finished, capped, block-budget-exhausted or
engine-error. Only a finished response graded correct by unchanged flexible-v3
is automatically eligible. Strict scoring is retained alongside it.

After an attempt round finishes, retain every response and retry only the
ineligible problems:

| Attempt/retry index | Data seed | Work |
|---|---|---|
| 0 | 0 | Initial attempt |
| 1 | 1001 | First model-outcome retry |
| 2 | 1002 | Second retry |
| 3 | 1003 | Final retry |

The source formula is `data_seed + 1000 + retry_index` for one-based retries.
Every covered problem stops at its first eligible attempt. Wrong answers,
stop-only responses and capped/exhausted traces remain in the attempt table.
There is no gold fallback, duplication or fourth retry.

An empty/malformed engine answer instead replays the identical logical request
once, with the same seed. A second engine/protocol failure blocks the whole arm;
it does not consume a model retry or publish a completed disposition table.
Engine-error diagnostics and failed attempt rows are retained. Interruptions are
operational events, not invented terminal model outcomes.

## Durable layout and restoration

```text
artifacts/runs/sampling-sft/<source-plan>/generation/<name>/
  meta/run_manifest.json
  generation.review.json
  run_k0.json  run_k2.json  run_k5.json
  k0/  k2/  k5/
    writer.json                 # exists while owned; explicit recovery after a hard kill
    events.jsonl
    meta/status.json
    rounds/0/ ... rounds/3/
      env.<run>.json
      blocks/000/ ...
        scientific.jsonl       # requests, proposals, aligned scores and decisions
        timings.jsonl          # operational work counters
        state.json             # retained IDs/texts, next block and per-problem RNG
        complete.json          # hash chain, runtime and worker evidence
      partial/
      abandoned/<run>/
      attempts.jsonl
      complete.json
    attempts.jsonl
    dispositions.jsonl
    results.json
    complete.json
  results/results.json
  complete.json
```

Resume with the same plan and k in a newly approved bounded job. Only sealed
blocks are progress. A partial block moves to retained operational history and
is replayed; previous seals stay unchanged. Completed responses inside a saved
round are not generated again. After a round finishes, later retries exclude
covered problems. The model is loaded once per arm invocation and reused across
its retry rounds. If only finalization remains, saved states finish on CPU
without another model load.

One writer owns an arm. For a genuinely dead writer, use the recorded token:

```sh
python -m pilot_eval sampling-sft-generation-recover-writer --config plans/sampling-generation-v1/sampling-sft.generation.json --k 0 --owner-token RECORDED_TOKEN --review-notes "Recorded process confirmed stopped" --output-root artifacts
```

An active local writer or wrong token is rejected. Changed inputs, code,
runtime or sealed artifacts require investigation and a new attempt, not an
in-place continuation with different settings. `--stop-after-blocks` is available
for bounded diagnostic pauses; it does not change the scientific recipe.

## Read and verify the outputs

```sh
python -m pilot_eval sampling-sft-generation-verify --config plans/sampling-generation-v1/sampling-sft.generation.json --k 0 --output-root artifacts
python -m pilot_eval sampling-sft-generation-verify --config plans/sampling-generation-v1/sampling-sft.generation.json --output-root artifacts
```

Each completed arm has exactly512 dispositions. A covered row references one
finished eligible attempt; an uncovered row has exhausted all four bounded
attempts. Verification reconstructs the attempt rows from sealed states,
recomputes automatic scoring, checks first-eligible selection and reconciles
end-state counts, IDs and retained response-token totals. All three arms must
verify before the top-level collection is completed.

`response_tokens` counts retained attempt tokens, including actual stops; it is
not the total compute spent on rejected proposals and auxiliary scorer work.
Those requests and timings remain in block evidence. `run_k*.json` retains a
cumulative experiment-ledger snapshot at collection, with unknown active cost
still reserved. Status also reads the current authoritative ledger; it does not
invent final bills from elapsed time.

During a partial round, `provisional_covered` reports eligible finished states
already saved in blocks; `attempt_count` counts sealed round attempts. This
distinction keeps live progress visible without presenting an unfinished round
as a completed attempt table.

Collection completion is not the coverage/training gate. Dispositions are
automatic-only, human review is pending and `training_allowed` stays false.
Ticket06 handles separate human eligibility decisions and reviewed coverage;
Ticket07 computes the descriptive recipe comparisons. The temperature1 baseline,
NLL offset, evaluation and internals remain deferred.

## Sequential three-arm generation and full scientific inventory

The user selected k0, k2 and k5 in ADR 0023. Prepare one distinct lifecycle job
per arm with the reviewed generation config, provider volume/rate and a bounded
`estimated_hours`/`setup_hours` forecast. Each job must declare its arm's
`complete.json`, `results.json`, `attempts.jsonl` and `dispositions.jsonl` under
the canonical generation directory. Those declarations are a minimum, not a
complete raw-data export. The job lifecycle retains cost reservations until
actual billing is reconciled.

Prepare the three jobs without launching them, using explicit estimates in hours:

```sh
python -m pilot_eval sampling-sft-pipeline-prepare-generation --config plans/GENERATION/sampling-sft.generation.json --volume-id VOLUME_ID --rate ACTUAL_QUOTE --reviewer Prasad --hours-k0 ESTIMATE --hours-k2 ESTIMATE --hours-k5 ESTIMATE --setup-hours ESTIMATE --output-root artifacts
```

This saves a JSON array under the generation run tree with three prepared job paths:

```json
[
  {"arm": 0, "job": "runs/sampling-sft/SOURCE/jobs/JOB0/job.plan.json", "reviewer": "Prasad"},
  {"arm": 2, "job": "runs/sampling-sft/SOURCE/jobs/JOB2/job.plan.json", "reviewer": "Prasad"},
  {"arm": 5, "job": "runs/sampling-sft/SOURCE/jobs/JOB5/job.plan.json", "reviewer": "Prasad"}
]
```

```sh
python -m pilot_eval sampling-sft-pipeline-generation --config plans/GENERATION/sampling-sft.generation.json --jobs PATH_TO_ARRAY.json --output-root artifacts
```

The command runs the jobs in order, never concurrently, and binds each user's
budget waiver to the then-current ledger and request immediately before launch.
A completed successful job is reused; a failed job or uncertain recorded launch
stops for inspection and reconciliation. It never silently replaces a stale
approval or fabricates a successful job. The estimate sets a hard child timeout
at 1.5 times estimated hours, and must not be mistaken for an upper-bound
measurement. Each job retains its child log and normal lifecycle receipts.

After all three complete, the wrapper verifies the scientific arms and writes
`results/full-scientific-inventory.json` with hashes of the complete dynamic arm
trees, including block-level scientific records. This inventory is not itself an
external copy. Independently copy and verify the full inventory, plans and budget
ledger before any requested Pod stop. The ordinary job export covers only the
explicitly declared files and does not replace the full-tree backup.

After generation, prepare the eligibility packet with:

```sh
python -m pilot_eval sampling-sft-pipeline-review --config plans/GENERATION/sampling-sft.generation.json --name REVIEW --output-root artifacts
```

The review command freezes coverage only when actual required human labels are
complete. Finish actual human review for any invalid extraction. If coverage is blocked,
stop; for omissions or flagged cohort differences, record the required human
acknowledgments. Then perform the independent score/data gate, full T15/T16
readiness and selected k0/k2/k5 L4 training jobs in order, using the respective
runbooks. It does not fabricate labels, select a GPU or claim training occurred.

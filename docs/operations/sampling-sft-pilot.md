# Sampling SFT: fixed pilot and exact replay

Ticket 04 implements the 32-problem k5 pilot. It does not authorize a paid job
or full generation. CPU fixtures exercise recovery; real L40S qualification is
a separate recorded execution.

## Sequence

1. Complete the source audit, reserved engine checks and measured Stage0.5
   selection described in [sampling-sft-engine.md](sampling-sft-engine.md).
2. Prepare the pilot on CPU from the selected instrument plan.
3. Run the `uninterrupted` lane to completion inside an approved budgeted job.
4. Run the `resumed` lane with the controlled kill after completed block 2.
5. Confirm the child exited by SIGKILL. Inspect its recorded writer identity,
   release that ownership explicitly, then resume in a fresh process/job.
6. Compare scientific records and produce the CPU alarm/forecast report.

The pilot selects 32 ordinals with local seed0 from the original ordered512
training rows. Both lanes use those same problems, k5, the same immutable
runtime and the same ordered extension/refinement request groups.

```sh
python -m pilot_eval sampling-sft-pilot-prepare --instrument plans/CHOSEN_PROFILE/sampling-sft.instrument.json --name sampling-pilot-v1 --output-root artifacts
python -m pilot_eval sampling-sft-pilot-status --config plans/sampling-pilot-v1/sampling-sft.pilot.json --output-root artifacts
```

Config/job paths are relative to the artifact root or absolute. Settings/review
files supplied separately are relative to the checkout or absolute. Preparing
without a runtime-selection receipt is permitted for CPU fixtures; real
execution refuses it before model loading. CPU preparation is not qualification.

## Budgeted GPU jobs

Use the existing [job lifecycle](sampling-sft-runpod.md): frozen request, actual
quote, conservative reservation, explicit paid approval, export, downloaded-copy
receipt and own-Pod stop. Count setup and debugging against sampling. Nothing
here rents a Pod, changes a cap or deletes storage.

Each real child command uses `sampling-sft-pilot-run` with `--config`,
`--output-root`, its own `--job` plan and one of these argument sets:

| Job | Arguments | Expected outcome |
|---|---|---|
| Reference | `--lane uninterrupted` | Completed 32-item pilot |
| Fault injection | `--lane resumed --interrupt-after-block 2` | Child exit -9 after the second block seal |
| Restore | `--lane resumed` | Reuses completed blocks and computes missing work |

Reserve each job before launch. The controlled kill is deliberately a nonzero
job outcome: export its partial block evidence, events and writer metadata,
not a nonexistent final completion. The generic runner cleans up its owned
inference process group; it does not call an unfinished dataset complete.
Reconcile that job's actual cost before launching another GPU job. Resume keeps
the same pilot plan, lane and runtime, but uses a separately approved job ID.

`--stop-after-blocks 2` is a graceful diagnostic pause. It exercises restoration
but cannot stand in for the required kill/restart qualification. If the entire
pilot finishes before the injection point, the kill criterion remains unmet;
the software does not move the injection point or claim success automatically.

## Recovery and ownership

Pilot artifacts live under:

```text
artifacts/runs/sampling-sft/<source-plan>/pilot/<pilot-name>/
  plan-seal.json
  uninterrupted/  resumed/
    writer.json                 # only while owned; survives a hard kill
    events.jsonl                # starts, engine replays, interruptions, recovery
    env.<attempt>.json          # actual runtime, separately hash-bound to blocks
    meta/status.json
    blocks/000/                 # completed block only
      scientific.jsonl
      timings.jsonl
      state.json                # states, next block, per-problem RNG
      complete.json             # file hashes and previous-block seal
    partial/                    # not committed scientific progress
    abandoned/<attempt>/        # retained partial history after restoration
    results.json
    responses.jsonl
    complete.json
  replay.json
  report.json
```

After confirming the old process is dead, read its `writer.json` and run:

```sh
python -m pilot_eval sampling-sft-pilot-recover-writer --config plans/sampling-pilot-v1/sampling-sft.pilot.json --lane resumed --owner-token RECORDED_TOKEN --review-notes "Confirmed recorded worker exited; checked job receipt" --output-root artifacts
```

An active local PID or wrong token is rejected. Cross-host recovery relies on
the recorded identity and explicit operator review. Ownership is never silently
cleared. Changed source, runtime, environment, state or sealed payload fails
verification rather than recomputing over it.

Only completed blocks count as progress. An unfinished block is moved to
`abandoned/` as operational history and replayed from the previous sealed RNG
and state. Earlier sealed bytes are preserved. Cache clearing occurs at every
block boundary and process start; within-block caching remains enabled if the
selected runtime permits it. Input and cached-token counters retain the actual
prefill work. No unmeasured cache-saving counterfactual is claimed.

If the final block was sealed but finalization was interrupted, restoration
finishes from its saved texts/IDs on CPU without loading the model again.

## Exact scientific comparison

```sh
python -m pilot_eval sampling-sft-pilot-compare --config plans/sampling-pilot-v1/sampling-sft.pilot.json --output-root artifacts
```

The comparison includes ordered requests, response IDs, scores, strict acceptance
decisions, final texts, end states and automatic scoring. Times, PIDs, cache
counters and engine-replay diagnostics remain operational records. A failure
saves the first differing coordinate and blocks qualification without a numeric
tolerance. The report records the approved requalification order: disable
prefix caching, constrain scheduling, then enable supported deterministic modes.
These are reviewed new attempts, not automatic changes to a running experiment.

Real qualification additionally requires real-CUDA evidence for every sealed
block, a new attempt restored from block2, the recorded controlled kill, and the
external job execution receipt confirming exit -9. Graceful pause, CPU SIGKILL
tests or two uninterrupted copies cannot satisfy that real qualification.

## Pilot alarm and cost report

Supply an explicit forecast JSON with `rate_usd_per_hour`, `estimated_hours`,
`setup_hours` and `notes` explaining its measured pilot basis. The estimate
should cover k0/k2/k5, bounded retries and setup/debugging; the code does not
assume that k5 timing is an upper bound on the other arms' lengths. Optional
additional fields can record the calculation and retry assumptions.

```sh
python -m pilot_eval sampling-sft-pilot-report --config plans/sampling-pilot-v1/sampling-sft.pilot.json --forecast artifacts/plans/sampling-pilot-v1/stage2-forecast.json --output-root artifacts
```

The report shows all32 initial attempts, flexible-v3 correctness, strict scoring,
length median/p90, acceptance by step/cut decile, measured token/work timings,
memory evidence, replay status and the proposed buffered reservation. P90 uses
the inclusive linear quantile. Forecasts are estimates, not automatically
approved or reserved jobs. Current spent/reserved amounts come from the shared
ledger; unknown prior cost remains held there.

| Alarm | Rule at n32 |
|---|---|
| Unfinished | capped + exhausted >2%; one case triggers |
| Initial correctness | flexible-v3 correct <90%; 28 or fewer triggers |
| Budget | rate × (estimated hours ×1.5 + setup hours) exceeds remaining sampling/total headroom |

Every alarm includes its value, denominator/threshold and triggered flag.
Engine failure blocks the attempt separately. A replay failure remains visible
in the report. No prompt, J64 or cap1024 is changed automatically. Production
always remains disabled here; Ticket05 requires qualified evidence and the
subsequent reviewed budget/approval handoff.

## Inspect a saved replay mismatch on CPU

This standalone diagnostic reads both saved lanes and writes a new diagnostic
folder without changing the frozen pilot implementation or replay report. It
requires no model, GPU, network, or paid job. Run from a checkout that contains
`pilot_eval/sampling_sft_replay_diagnostic.py`:

```sh
python -m pilot_eval.sampling_sft_replay_diagnostic --pilot-dir artifacts/runs/sampling-sft/SOURCE/pilot/PILOT --name replay-inspection-1
```

The name must be new; the command refuses to overwrite an existing directory.
`diagnostics/NAME/summary.json` contains score counts and maximum absolute
error, decision and final-trace counts, per-block localization, source hashes,
and structural/integrity issues. `differences.jsonl` includes the corresponding
individual score, generation, decision and final-trace differences. Blocks 1–2
are before the recorded interruption boundary; later blocks are after it.
Requests are compared only if their full saved inputs match. Changed, missing or
duplicate requests are reported separately, so score counts cover aligned
requests only. An integrity failure invalidates conclusions until investigated.
This is descriptive evidence; it does not change or pass the exact replay gate.

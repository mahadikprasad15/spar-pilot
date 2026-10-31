# Sampling SFT qualification on the existing A40 Pod

This handoff follows the successful four-problem smoke. It runs no full512
sampling and no training. Previously sealed artifacts remain untouched; new
instrument code requires a fresh name such as a40-q-v1. Keep the old smoke's
pinned release and independent backup as its verification record. The corrected
IEEE runtime requires new settings, a new name, and fresh fixtures. See
[ADR 0021](../adr/0021-sampling-sft-ieee-runtime.md).

## Preparation and execution

In the Pod's activated Python environment, set:

```bash
export PATH="/root/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Create a fresh settings file from the earlier smoke settings. This reads the
installed source files without importing Triton, saves their fingerprints, and
does not modify the original settings:

```bash
python -m pilot_eval.sampling_sft_runtime \
  --output-root /workspace/artifacts \
  --input /workspace/artifacts/plans/sampling-sft-v1/engine-a40-smoke-v1.json \
  --output /workspace/artifacts/plans/sampling-sft-v1/engine-a40-ieee-v1.json
```

Define this convenience function in that same SSH shell. The volume ID below
belongs to the currently verified A40 setup; another deployment must supply its
own volume ID, settings and rate.

```bash
qualify() {
  python -m pilot_eval sampling-sft-qualification-stage \
    --config /workspace/artifacts/plans/sampling-sft-v1/sampling-sft.plan.json \
    --settings /workspace/artifacts/plans/sampling-sft-v1/engine-a40-ieee-v1.json \
    --output-root /workspace/artifacts --name a40-q-ieee-v1 \
    --volume-id cmuptiucb000007l5dnohe861 --rate 0.60 \
    --reviewer Prasad --stage "$1" --run
}
```

The0.60 rate is a conservative estimate above the observed0.59 quote, not a bill.
Default work estimate0.5h gives a45-minute child timeout with the existing1.5
buffer. `--estimated-hours` is configurable before preparing the stage. Actual
cost remains unsettled until evidenced. The per-job waiver removes cost-cap
blocking, not deadlines or scientific gates. Prior historical cost remains
incomplete; no zero-total-cost claim is made.

Run stages individually and inspect each result before proceeding:

| Command | Work and required evidence |
|---|---|
| `qualify fixtures` | Generate20 fixed scorer fixtures; one is deliberately forced to1024 tokens. Never training data. |
| `qualify refinement` | Four normal k2 attempts; log cuts, scores and accept/reject decisions. |
| `qualify scores` | vLLM raw log probabilities for the sealed20 traces. |
| `qualify oracle` | Independent HF FP32/eager probabilities for the identical IDs and positions. |
| `qualify profile` | Compare T3 first; one block on64 fixed problems at k2, batching comparison and worker memory evidence. |
| `qualify select` | CPU: requires real passing T3 and profile; selects only the supplied scheduler, no optimality claim. |
| `qualify pilot-reference` |32-problem k5 uninterrupted lane. |
| `qualify pilot-kill` | Same cohort/settings; intentional child SIGKILL after sealed block2. Expected returncode-9. |
| `qualify pilot-resume` | Check external-9 receipt and dead ownership, then resume missing work in a new process/job. |
| `qualify compare` | CPU: exact scientific equality plus real kill/restart evidence. |

The command follows the correct child log automatically. Do not add `-u` to the
saved child argv; PYTHONUNBUFFERED controls log flushing. No command rents,
restarts, deletes or stops the Pod. A model/protocol failure stops the stage and
retains logs. Re-running a completed job verifies/reuses it; failed/interrupted
jobs are not automatically relaunched under the same identity.

To inspect/prepare without inference, call the full command without `--run`.
Preparation still requires upstream fixture/evidence artifacts for dependent
stages. Select/compare are CPU operations; their evidence requirements remain.
The stage receipts/approvals live under the source run's qualification/name
folder; scientific data are in instrument/name-stage and pilot/name-pilot.
All ordinary job metadata, logs and manifests retain the existing jobs layout.

## Fixtures and numerical acceptance

The20 problem ordinals are fixed before capture as14,41,63,75,95,97,101,103,142,
143,150,209,223,256,265,266,288,310,317,323 (a subset of the seed0 pilot cohort).
The final request's min_tokens1024 is a test-only stop suppression control. The
other19 use min_tokens0. All response IDs come from saved generation outputs.
The shortest length(s) are short and longest length(s) long; these labels are
assigned before scoring. Degenerate lengths or no finished response block T3.
No traces are replaced after seeing score errors.

Both scorers use the saved original user/scorer prompt, without gold in its
context. The independent comparison retains per-token errors and worst positions.
Every trace must satisfy mean error<=0.001 nats/token, max per-token error<=0.01,
and exact ID/position alignment. Failure blocks selection; no tolerance changes.
Successful fixture generation or matching CPU fixture scores is not GPU
qualification. Stage0.5 selection is not exact-replay qualification.

## After qualification

Inspect the32-item pilot alarm report using the existing pilot-report command
and an explicit measured-basis forecast. The correctness/unfinished alarms and
production freeze/review are not bypassed by this handoff. A replay mismatch
also blocks full generation; follow the existing reviewed requalification order.

Before ending the GPU phase, back up the complete source-run tree, plans and
notebook controls to HF plus an independently downloaded, hash-verified copy.
The generic job export contains declared outputs only; include blocks, requests,
fixtures, pilot partial/abandoned history, `compiler-cache`, and the current ledger in the complete
phase backup. Keep credentials outside every archive. Only after verified
exports request own-Pod stop through the existing lifecycle and verify externally.
Do not delete the Global Volume or other account storage.

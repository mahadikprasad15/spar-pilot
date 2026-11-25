# Sparse rewiring: Pod rehearsal runbook (ticket 15)

One A100 Pod, one tmux window, one launch. The chain runs **preflight → qualification (automatic sign-off) →
gold references → rank 1 (16 steps, evaluated at 0, 12, 16) → rank 0 (same) → verified volume copy → HF export →
Pod stop**. It is a machinery and measurement run: the label is `rehearsal-pod-1`, its numbers are not results, and
cohort150 is never touched (the launcher refuses it). If any qualification check fails, the chain stops before the
references and before any training, copies what it has, and stops the Pod. That is a finding to send back, not
something to work around.

Record: `.scratch/sparse-rewiring-gsm8k/launch-record-rehearsal.yaml` (the production draft with only the run
length, checkpoints, cadences, label and run IDs changed; `tests/test_rewire_launch.py` checks that).

## 0. Once, on your Mac (the machine that holds the artifact root)

The bundle needs gold's plan and run files, the gold step-64 adapter, ticket 04's validation set, the tokenizer files
and the base model, all under one artifact root. Ticket 01 found the gold files in `~/spar/pilot1/artifacts`.

```bash
cd ~/spar-pilot && git fetch && git checkout claude/project-thread-7tozx7 && git pull
ROOT=~/spar/pilot1/artifacts
# ticket 04's frozen validation set (committed by ticket 01 in PR #18 under the repo's artifacts/):
mkdir -p $ROOT/validation && cp -R artifacts/validation/sparse-rewiring-ticket04-v1 $ROOT/validation/
# the pinned tokenizer files, generation config and base model, if $ROOT/inputs/models/<rev>/ is missing them:
REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct', revision='$REV', \
  allow_patterns=['config.json','model.safetensors','generation_config.json','tokenizer.json', \
  'tokenizer_config.json','vocab.json','merges.txt'], local_dir='$ROOT/inputs/models/$REV')"
# build (hashes every file against ticket 01's values and the base model against HF's pins), upload with
# readback to the private dataset, write the receipt:
python -m pilot_eval.rewire_bundle export --artifact-root $ROOT --label rehearsal-pod-1 \
  --receipt .scratch/sparse-rewiring-gsm8k/bundle-receipt-rehearsal-pod-1.json
git add .scratch/sparse-rewiring-gsm8k/bundle-receipt-rehearsal-pod-1.json
git commit -m "Sparse rewiring: rehearsal input bundle receipt" && git push
```

`export` refuses a missing file, a file whose hash differs from ticket 01's, a validation set that is not ticket 04's
(`ids_sha256 5ea47bc5…`), uncommitted `catalog/` changes, and an existing receipt.

## 1. Pod

- 1× **A100-SXM4-80GB**, Global Volume **sl-0** attached (it holds `/workspace/neuron-audit/.neuron-audit-volume.json`).
- **120 GB container disk** (the default 50 GB is too small; preflight refuses it).
- Pod secret **`HF_TOKEN`** with write access to `Prasadmahadik/spar-sampling-sft-artifacts`. Never echo it.

## 2. Environment

```bash
export REPO=/root/spar-pilot WORK=/root/sparse VOL=/workspace/neuron-audit PY=/root/venv/bin/python
tmux new -s sparse
git clone --branch claude/project-thread-7tozx7 https://github.com/mahadikprasad15/spar-pilot.git $REPO
python3 -m venv /root/venv
$PY -m pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q transformers==4.57.6 peft==0.18.0 accelerate==1.12.0 huggingface-hub==0.36.0 \
    tokenizers==0.22.1 safetensors pyyaml pyarrow jinja2 nvidia-ml-py numpy
export HF_HOME=/root/hf-home PYTHONUNBUFFERED=1 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
source /root/venv/bin/activate && cd $REPO && mkdir -p $WORK
$PY -m pilot_eval.neuron_audit_pod facts --volume_dir $VOL --work $WORK
```

Check: the device is an A100-80GB, `$VOL` is on a different mount from the container disk, no `disk_problems`.

## 3. Inputs (the only online steps)

```bash
HF_HUB_OFFLINE=0 $PY -m pilot_eval.rewire_bundle restore \
    --receipt .scratch/sparse-rewiring-gsm8k/bundle-receipt-rehearsal-pod-1.json --clean-root $WORK/inputs
#   -> prints verified_files and bundle_manifest; keep the bundle_manifest path for step 4
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct', \
    revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', allow_patterns=['config.json', 'model.safetensors'], \
    local_dir='$WORK/model')"
```

## 4. Seal the record

```bash
$PY -m pilot_eval.rewire_launch seal --draft .scratch/sparse-rewiring-gsm8k/launch-record-rehearsal.yaml \
    --out $WORK/launch_record.json --artifact-root $WORK/inputs --bundle <bundle_manifest from step 3> \
    --model-dir $WORK/model --work-dir $WORK
```

`seal` fills in the code commit of this checkout and the Pod ID (`RUNPOD_POD_ID`) and refuses anything empty or
mistyped.

## 5. Launch

```bash
set -o pipefail
export HF_HUB_OFFLINE=1
PY=$PY bash scripts/rewire_launch.sh $WORK/launch_record.json $WORK/local $VOL 2>&1 | tee $WORK/launch-1.console.log
# add --stop-pod as a 4th argument only if you want the Pod stopped after a complete verified backup
```

The launcher takes a lock, runs `precheck`, starts the chain and the independent watchdog (no time caps: it acts
only if the chain dies without finishing, then copies sealed work). Without `--stop-pod` (the default since
2026-11-25) nothing stops the Pod: the chain copies, exports and verifies, then prints the stop command for you.
For production, seal `launch-record-draft.yaml` instead of the rehearsal YAML; cohort150 is then scored at step
1,024 only. Progress is in
`$VOL/c-attribution/sparse-rewiring-rehearsal-pod-1/c-attribution/chain/progress.json`.

- **If it prints `NO RUN STARTED`**, nothing is running and the Pod is still billing: fix the named problem and
  relaunch, or `runpodctl stop pod $RUNPOD_POD_ID`.
- **If it prints `STOP WITHHELD`**, no backup holds all the work, so the Pod was left running. The reason and the
  next step are in `chain/stop_withheld.json`.
- **A relaunch on the same Pod** (after a crash) re-verifies and skips completed stages; an interrupted arm
  restarts from step 0 (D24). Wait until `precheck` prints ok (a dead writer's heartbeat must be 900 s old).

## 6. Afterwards (stop the Pod yourself unless you passed --stop-pod)

Results are on sl-0 under `$VOL/c-attribution/sparse-rewiring-rehearsal-pod-1/` and in the HF dataset. On a cheap
CPU Pod with sl-0 attached (or from the HF export), send back:

```bash
R=$VOL/c-attribution/sparse-rewiring-rehearsal-pod-1/c-attribution
cat $R/chain/stop_record.json $R/chain/last_attempt_end.json
cat $R/summaries/attempt-001/final/run_summary.md
cat $R/sparse-rewiring/qualification/summary/qualification.json | head -c 30000
cat $R/sparse-rewiring/references/summary/references.json
```

`qualification.json` holds every check and the measurements that replace the arithmetic estimates in
`decision-log.md`: peak memory per rank on the longest row, seconds per step, one validation CE pass, one 16-question
greedy pass, and the extrapolated production hours.

## 7. Rank-0 read-LR probe (decision 2026-11-25)

The rehearsal showed rank 0 at read LR 3e-4 falling from 103/128 to 36/128 (step 12) and 12/128 (step 16).
`pilot_eval.rewire_lr_probe` trains rank 0 alone under the rehearsal's frozen plan with only the read LR changed,
and reports validation CE at steps 12 and 16 plus 128-question greedy accuracy at step 16. No chain, no sign-off,
no cohort150. About 12 minutes per LR on an A100. Rank-0 checkpoints stay on the container disk and are deleted
once evaluated; results go to the volume.

A fresh A100 Pod with sl-0 and 120 GB disk, then sections 2 and 3 as before (same receipt), then:

```bash
export HF_HUB_OFFLINE=1
$PY -m pilot_eval.rewire_launch seal --draft .scratch/sparse-rewiring-gsm8k/launch-record-rehearsal.yaml \
    --out $WORK/probe_record.json --artifact-root $WORK/inputs \
    --bundle $WORK/inputs/bundles/c-attribution/sparse-rewiring-rehearsal-pod-1/bundle-f4c4112263f23ae4.json \
    --model-dir $WORK/model --work-dir $WORK
$PY -m pilot_eval.rewire_lr_probe --record $WORK/probe_record.json --volume $VOL --work $WORK/lr-probe \
    --out $VOL/c-attribution/sparse-rewiring-lr-probe-1 --lrs 1e-4 3e-5 1e-5 2>&1 | tee $WORK/lr-probe.log
```

It prints one `RESULT` line per LR. Probes do not stop the Pod (researcher, 2026-11-25): stop it yourself after reading the results. The numbers are in
`$VOL/c-attribution/sparse-rewiring-lr-probe-1/lr-probe.json`.

## 8. Rank-0 64-step probe at 3e-5 (decision 2026-11-25)

The 16-step probe could not show pruning: gates start open (logα 1.0) and none can close before about step 34. This
runs rank 0 at 3e-5 over production's own first 64 steps (`launch-record-lrprobe64.yaml` is the production draft
with only mode, label and run IDs changed): checkpoints 16, 32 and 64 with validation CE and open-gate counts, and
128-question greedy accuracy at 64. About 30–35 minutes on an A100. Same Pod and sections 2–3 as before, then:

```bash
export HF_HUB_OFFLINE=1
$PY -m pilot_eval.rewire_launch seal --draft .scratch/sparse-rewiring-gsm8k/launch-record-lrprobe64.yaml \
    --out $WORK/probe64_record.json --artifact-root $WORK/inputs \
    --bundle $WORK/inputs/bundles/c-attribution/sparse-rewiring-rehearsal-pod-1/bundle-f4c4112263f23ae4.json \
    --model-dir $WORK/model --work-dir $WORK
$PY -m pilot_eval.rewire_lr_probe --record $WORK/probe64_record.json --volume $VOL --work $WORK/lr-probe-64 \
    --out $VOL/c-attribution/sparse-rewiring-lr-probe-64 --lrs 3e-5 --cut-steps 64 --checkpoints 16 32 \
    2>&1 | tee $WORK/lr-probe-64.log
```

Probes and rehearsal-style checks are launched without `--stop-pod` (researcher, 2026-11-25): the researcher stops
the Pod after reading the results. Results reach the volume after each LR either way.

## 9. Production report and catalog (ticket 18)

After the production chain finishes (`state complete` in the final summary), on the same Pod or any Pod with sl-0:

```bash
R=$VOL/c-attribution/sparse-rewiring-production
$PY -m pilot_eval.rewire_report build --root $R          # seals $R/c-attribution/sparse-rewiring/report/report.json
cat $R/c-attribution/sparse-rewiring/report.md            # send this back
$PY -m pilot_eval.rewire_report register --root $R --location sl-0 \
    --prefix c-attribution/sparse-rewiring-production --catalog catalog
$PY -m pilot_eval.catalog verify --root sl-0=$VOL         # the new entries must all pass
git add catalog && git commit -m "Catalog: sparse rewiring production arms, responses, prior uses" && git push
```

`build` refuses any missing or hash-mismatched unit. It lists the ticket 10–13 analyses it did not find under
`c-attribution/sparse-rewiring/analysis/` instead of inventing them; rerun `build` after they are sealed there.


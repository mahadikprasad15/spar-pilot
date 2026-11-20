# C-attribution: attended Pod rehearsal runbook (ticket 11)

This is one Pod and one tmux window. You make one deliberate interruption and one resume, and the run ends with
a real Pod stop by the chain. It is a machinery and measurement run: its report is labelled REHEARSAL and its
numbers are not results. The steps follow `.scratch/adapter-neuron-attribution/pod-runbook.md`, with every Pod
lesson in `docs/agents/experiment-lessons.md` applied.

**Before you start (done locally, already committed):** the input bundle is exported and its receipt is at
`.scratch/c-attribution-pilot/bundle-receipt-inputs-v1.json`. The approved record is
`.scratch/c-attribution-pilot/launch-record-rehearsal.yaml`.

```bash
export REPO=/root/spar-pilot WORK=/root/c-attr VOL=/workspace/neuron-audit PY=/root/venv/bin/python
```

## 0. Pod

- 1× **A100-SXM4-80GB**, with Global Volume **sl-0** attached. It already holds `$VOL/.neuron-audit-volume.json`.
- **120 GB container disk.** The 50 GB default is too small (lesson pod1).
- A Pod secret **`HF_TOKEN`** with write access to `Prasadmahadik/spar-sampling-sft-artifacts`. Never echo it.

## 1. Environment

```bash
tmux new -s cattr
git clone --branch c-attribution-pilot https://github.com/mahadikprasad15/spar-pilot.git $REPO
python3 -m venv /root/venv
# Exactly these versions. torch from the cu128 index brings the CUDA runtime, cuDNN and the nvidia-* libraries,
# which are part of the resume fingerprint.
$PY -m pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q transformers==4.57.6 peft==0.18.0 accelerate==1.12.0 huggingface-hub==0.36.0 \
    tokenizers==0.22.1 safetensors pyyaml pyarrow jinja2 nvidia-ml-py
# HF's cache must be on the container disk: downloads through the sl-0 volume came back empty (lesson pod-1).
export HF_HOME=/root/hf-home PYTHONUNBUFFERED=1 PYTHONPATH=$REPO
source /root/venv/bin/activate && cd $REPO && mkdir -p $WORK
```

## 2. Facts

```bash
$PY -m pilot_eval.neuron_audit_pod facts --volume_dir $VOL --work $WORK
```

Check three things:
- The device is an A100-80GB.
- `$VOL` is on a different mount from the container disk.
- There are no `disk_problems`.

## 3. Inputs

These are the only online steps. The run itself is offline.

```bash
# the bundle: responses, configs, adapters, tokenizer files, generation config, hot-layer table, cohort
HF_HUB_OFFLINE=0 $PY -m pilot_eval.c_attribution_bundle restore \
    --receipt .scratch/c-attribution-pilot/bundle-receipt-inputs-v1.json --clean-root $WORK/inputs
#   -> prints verified_files and bundle_manifest; keep the bundle_manifest path for step 4
# the base model at the pinned revision (preflight checks both files against HF's own IDs)
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct', \
    revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', allow_patterns=['config.json', 'model.safetensors'], \
    local_dir='$WORK/model')"
```

## 4. Seal the approved record

```bash
$PY -m pilot_eval.c_attribution_chain seal --draft .scratch/c-attribution-pilot/launch-record-rehearsal.yaml \
    --out $WORK/launch_record.json --artifact-root $WORK/inputs --bundle <bundle_manifest from step 3> \
    --model-dir $WORK/model --work-dir $WORK
```

`seal` fills in the code commit and the Pod ID (`RUNPOD_POD_ID`). It refuses anything empty or wrongly typed.

## 5. First launch, with a dry stop

The rehearsal must survive one interruption, but a stop at that point would end it. For this launch only, a
stand-in `runpodctl` records the stop instead of doing it.

```bash
mkdir -p /root/dry-stop && printf '#!/bin/sh\necho "DRY STOP $*" >> /root/dry-stop/log\n' > /root/dry-stop/runpodctl
chmod +x /root/dry-stop/runpodctl
PATH=/root/dry-stop:$PATH PY=$PY bash scripts/c_attribution_launch.sh $WORK/launch_record.json $WORK/local $VOL \
    2>&1 | tee $WORK/launch-1.console.log
```

- **Order:**
  1. preflight;
  2. qualification, with automatic sign-off or a stop;
  3. rows;
  4. attribution: 2 units per arm;
  5. aggregate, split, bootstrap and freeze;
  6. generation: 4 conditions × 8 questions per arm, untuned once;
  7. report;
  8. volume copy, HF export, final summary, then the stop.
- **Watch for qualification.** If any check fails, the chain stops. That is a finding, not something to work
  around: send the logs.
- **If this launch ends on its own** (a stop, a failure or completion), the dry stop means **the Pod is still
  running**. Collect step 8's files, then stop it by hand: `runpodctl stop pod $RUNPOD_POD_ID`.

**Interrupt during attribution.** This simulates a lost Pod. When `$VOL/c-attribution/c-rehearsal-pod-1/c-attribution/chain/progress.json`
shows `stage: attribution` and `units_done: 1`, do this in a second tmux window:

```bash
pkill -9 -f 'pilot_eval.c_attribution_chain run'
```

The watchdog sees the writer die without finishing. It copies the sealed work and records a stop in
`/root/dry-stop/log`, but nothing actually stops.

## 6. Resume from an empty local disk

```bash
rm -rf $WORK/local
# The dead writer's heartbeat must be older than 900 s before a takeover is allowed. Repeat until this prints ok:
$PY -m pilot_eval.c_attribution_chain precheck --record $WORK/launch_record.json --local $WORK/local --volume $VOL
# Then relaunch WITHOUT the dry stop: this launch ends with the real Pod stop.
PY=$PY bash scripts/c_attribution_launch.sh $WORK/launch_record.json $WORK/local $VOL 2>&1 | tee $WORK/launch-2.console.log
```

In the final summary, expect `resume: take-over` and `restored_from_volume` listing the first unit. Preflight runs
again and must match the qualified setup.
Completed stages are re-verified and skipped, and attribution continues from unit 2. At the end the Pod stops
itself.

- **If it prints `STOP WITHHELD`**, the Pod is still running because no backup holds all the work. The reason
  and the next step are in `chain/stop_withheld.json`.
- **If a resume is refused for the setup**, the message names the package to install or the GPU to rent.

## 7. Optional: rebind onto a second Pod (about 10 minutes more)

Do this instead of step 6, to rehearse a replacement Pod:
1. Stop this Pod by hand after step 5.
2. On a new Pod, run steps 0–3.
3. Rebind the record to the new Pod:
   `$PY -m pilot_eval.c_attribution_chain rebind --record <old launch_record.json> --out $WORK/launch_record.json`.
   Copy the old record from `$VOL/c-attribution/c-rehearsal-pod-1/c-attribution/launch/*/launch_record.json` on
   the volume.
4. Run step 6 with the new record.

## 8. Afterwards (the Pod is stopped)

Results are on sl-0 under `$VOL/c-attribution/c-rehearsal-pod-1/` and in the HF dataset (label
`rehearsal-pod-1`). On a cheap CPU Pod with sl-0 attached, collect these and send them back:

```bash
R=$VOL/c-attribution/c-rehearsal-pod-1
cat $R/c-attribution/chain/stop_record.json $R/c-attribution/chain/last_attempt_end.json
cat $R/c-attribution/summaries/attempt-002/final/run_summary.md
cat $R/c-attribution/qualification/qualification.json | head -c 20000
cat $R/c-attribution/qualification/numerics.json
cat $R/c-attribution/report/report.md
tail -100 $R/c-attribution/chain/console.log
cat /root/dry-stop/log 2>/dev/null   # only on the first Pod
```

Check that the summary lists every stage and that both backups are "verified, complete" before trusting anything.
The measurements in ticket 11's list then go into the production record, by your decision.

## 9. Locally, after the run (ticket 13)

1. Restore the export from HF into the local artifact root, for example `artifacts/imports/c-attribution-<label>/`.
2. Make the figures:
   `python -m pilot_eval.c_attribution_report figures --report <restored>/c-attribution/report/report.json --out <restored>/figures`.
3. Register the run:
   `python -m pilot_eval.c_attribution_report register --root <restored> --label <label> --location local --prefix imports/c-attribution-<label>`.
4. Run `python -m pilot_eval.catalog verify` and commit the catalog only when it reports OK.


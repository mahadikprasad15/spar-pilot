# Adapter attribution — supervised Pod rehearsal runbook

One Pod, one tmux window, **one launch, no automatic restart** (review round 5). This is a test of the overnight
chain on the real Pod, not production. It ends by stopping the Pod itself. Reuses the joint audit's Pod tooling
(`.scratch/joint-neuron-adapter-audit/pod-runbook.md`).

```bash
export REPO=/root/spar-pilot  WORK=/root/attr  VOL=/workspace/neuron-audit  PY=/root/venv/bin/python
```

## 0. Pod

- 1× **A100-SXM4-80GB**; attach Global Volume **sl-0** (it already holds the identity file, `$VOL/.neuron-audit-volume.json`).
- Container disk **120 GB**.
- Pod secret **`HF_TOKEN`** with write access to `Prasadmahadik/spar-sampling-sft-artifacts`. Never echo it.

## 1. Environment

```bash
tmux new -s attr
git clone --branch claude/bold-gauss-5qkzwi https://github.com/mahadikprasad15/spar-pilot.git $REPO
python3 -m venv /root/venv
$PY -m pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q transformers==4.57.6 peft==0.18.0 accelerate==1.12.0 huggingface-hub==0.36.0 \
    tokenizers==0.22.1 safetensors pyyaml pyarrow jinja2 nvidia-ml-py
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH=$REPO
# HF's cache must be on the container disk: through the sl-0 volume (/workspace, geesefs) downloads came back
# empty without error (Pod rehearsal pod-1, docs/agents/experiment-lessons.md).
export HF_HOME=/root/hf-home
cd $REPO && mkdir -p $WORK
```

## 2. Facts, then paste them back

```bash
$PY -m pilot_eval.neuron_audit_pod facts --volume_dir $VOL --work $WORK
```

Check: the device is an A100-80GB, `$VOL` is on a different mount from the container disk, and there are no
`disk_problems`.

## 3. Inputs: prep snapshot, pinned base model, pinned GSM8K files

```bash
$PY -m pilot_eval.neuron_audit_pod prepare --work $WORK          # writes $WORK/inputs.json
HF_HUB_OFFLINE=0 $PY -m pilot_eval.attribution_backend fetch-gsm8k --prep $WORK/prep \
    --sequences_dir $WORK/prep/runs/joint-neuron-adapter-audit/sequences/sequences-v2
```

Both verify hashes: the model files against inventory-v2, and the GSM8K files against the sequences-v2 manifest.

## 4. Seal the approved launch record

```bash
cp .scratch/adapter-neuron-attribution/launch-record-pod-rehearsal.yaml $WORK/draft.yaml
# set hourly_cost_usd in $WORK/draft.yaml (the only [user] value left), then:
$PY -m pilot_eval.attribution_chain seal --draft $WORK/draft.yaml --out $WORK/launch_record.json \
    --inputs $WORK/inputs.json --work $WORK
```

`seal` refuses while any value is empty.

## 5. Launch once (supervised)

The launcher calls plain `python`, so the venv must be active. Any earlier attempt's chain state on sl-0 is moved
aside first: the chain's volume paths are fixed, and a new record needs a new run ID (lessons, pod-1).

```bash
source /root/venv/bin/activate && which python
[ -d $VOL/attribution ] && mv $VOL/attribution $VOL/attribution-<previous run_id>
bash scripts/attribution_overnight.sh $WORK/launch_record.json $WORK/local $VOL 2>&1 | tee $WORK/launch.console.log
```

- **Order:** preflight → qualification (automatic sign-off or stop) → base traces → attribution and freeze →
  Stage C → Stage D → report → volume copy → HF export → final summary → **Pod stop**.
- **The console is also written to** `$VOL/attribution/chain/console.log`, which survives the stop.
- **If anything fails, it still tears down and stops the Pod.** Don't relaunch; send the logs instead.

## 6. Afterwards (the Pod is stopped)

Results are on sl-0 (`$VOL/attribution/…`) and in the HF dataset (label `rehearsal-pod-2`). To read them, start a
cheap CPU Pod with sl-0 attached and send back:

```bash
cat $VOL/attribution/chain/stop_record.json
ls $VOL/attribution/summaries/*/
cat $VOL/attribution/summaries/attempt-001/final/run_summary.md
tail -100 $VOL/attribution/chain/console.log
```

Check that the summary lists every stage and both backups as verified before trusting them.

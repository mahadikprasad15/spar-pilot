# Joint neuron/adapter audit — Pod runbook

One Pod session, one tmux window, one writer. Every step is resumable; nothing here deletes the Global Volume
or any HF data. Commands assume these names (set them once per shell):

```bash
export REPO=/root/spar-pilot          # code checkout (container disk)
export WORK=/root/audit               # local working dir (container disk; ephemeral)
export VOL=/workspace/neuron-audit    # audit dir ON THE sl-0 MOUNT (adjust to the Pod's mount path)
export PY=/root/venv/bin/python
```

## 0. Pod

- GPU: 1x A100-SXM4-80GB (the plan's `runtime.expected_device`).
- Attach Global Volume **sl-0**. Note its mount path; `VOL` must sit inside it.
- Container disk: **120 GB** (Runpod's default 50 GB is too small; `facts` reports `disk_problems` if so). At least 100 GB (the full run needs about 50 GB free for the working set; venv, model
  cache and packages add about 20 GB).
- Pod secret: `HF_TOKEN` with write access to `Prasadmahadik/spar-sampling-sft-artifacts`. Never echo it.
- `RUNPOD_POD_ID` is set by Runpod; `runpodctl` inside the Pod is used only by the final guarded stop.

## 1. Environment

```bash
tmux new -s audit
git clone --branch neuron-audit-phase1 https://github.com/mahadikprasad15/spar-pilot.git $REPO
git -C $REPO checkout <code_commit>            # the commit recorded in the plan (step 5)
python3.12 -m venv /root/venv
$PY -m pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q transformers==4.57.6 peft==0.18.0 accelerate==1.12.0 huggingface-hub==0.36.0 \
    tokenizers==0.22.1 safetensors pyyaml pyarrow jinja2 matplotlib
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH=$REPO
cd $REPO
```

## 2. Pod facts (changes nothing) → paste back

```bash
mkdir -p $VOL $WORK
$PY -m pilot_eval.neuron_audit_pod facts --volume_dir $VOL --work $WORK
```

Prints and saves `$WORK/pod-facts.json`: interpreter and prefix, package versions (full torch build string),
device, runtime fingerprint, volume mount/device, container free disk, code commit, and the suggested plan
values. **Paste the JSON back**; Claude fills the Pod-specific plan fields from it.

## 3. Volume identity (once per volume)

Only after `facts` shows `$VOL` on a different mount and device than the container disk:

```bash
$PY -m pilot_eval.neuron_audit_preflight init-volume --volume_dir $VOL --volume_id sl-0 --local_dir $WORK
```

It refuses if the directory is on the container disk; it never overwrites an existing identity file.

## 4. Inputs and the small Pod rehearsal → paste back

```bash
$PY -m pilot_eval.neuron_audit_pod prepare --work $WORK          # prep snapshot + pinned model, hash-checked
$PY -m pilot_eval.neuron_audit_pod rehearsal --work $WORK --volume_dir $VOL --tag pod1
```

The rehearsal uses the real volume and runtime with small sizes (`make_rehearsal_plan(small=True)`: 2+2 FP64
rows, 4/4 calibration rows, budgets 4096/8192, 1 h 03b ceiling), then:
03b → rehearsal sign-off → 04 with 2 batches per arm, interrupted after 3 units → **local working dir deleted**
→ 04 resumed from the volume (must log `units-restored-from-volume`) → clean-root `restore-final` from HF.
Outputs are labelled `rehearsal-pod-pod1` and can never feed production. It never stops the Pod.
**Paste back** the `PASTE THIS BACK` line and `$WORK/pod-rehearsal-pod1.json`.

## 5. Approved plan, review record and code commit

The approved plan (`frozen_plan_v1.approved.yaml`) and the background review record (`sequences-v2-review.json`)
are committed on the branch *after* the code commit they name. Fetch both from the branch tip, then check out
the plan's `runtime.code_commit` (detached) so preflight sees exactly that HEAD with no tracked changes:

```bash
B=origin/neuron-audit-phase1; S=.scratch/joint-neuron-adapter-audit
git -C $REPO fetch -q origin && git -C $REPO show $B:$S/frozen_plan_v1.approved.yaml > $WORK/plan.yaml
git -C $REPO show $B:$S/sequences-v2-review.json > $WORK/prep/runs/joint-neuron-adapter-audit/sequences/sequences-v2/review/review.json
git -C $REPO checkout -q --detach $(grep -m1 'code_commit:' $WORK/plan.yaml | awk '{print $2}')
$PY -m pilot_eval.neuron_audit_plan --plan $WORK/plan.yaml --stage 04 --artifact_root $WORK/prep
```

The last command must report no problems (it checks approval, every required field, the review decision and
that the review record matches the sealed sample).

## 6. 03b → 04 unattended (one command; ends with the Pod stopped)

```bash
$PY -m pilot_eval.neuron_audit_pod chain --work $WORK --volume_dir $VOL --tag v1 --plan $WORK/plan.yaml 2>&1 | tee $WORK/chain-v1.console.log | grep -E '"kind"|"stage"|CHAIN|Error|Traceback'
```

Pre-registered rules (plan `review_flags.automatic_signoff`, `operations.*`, user 2026-11-12):
- 03b runs under the 5 GPU-hour ceiling (cumulative across restarts).
- 03b exits 0, status `qualified`, no review flag → a sign-off labelled `automatic-rule (...)` is recorded and
  04 starts with the fixed 6 h cap. 04 stops the Pod after the verified final backup, or after a failure/overrun
  once the partial backup verifies (`stop_after_verified_partial_backup`).
- Anything else (failure, ceiling, `partial`, any review flag) → `CHAIN STOPPED BEFORE 04`: the 03b run dir is
  copied to `$VOL/runs/joint-neuron-adapter-audit/qualification-stopped/q-v1` and re-verified by hash, uploaded
  to HF when possible, and the Pod stops once the volume copy verified. A human then reviews; resume 03b/04 on a
  new Pod with the same tag.
- Report: `$WORK/chain-v1.json` (03b and 04 return codes, decision, sign-off, backups).

## 7. Resume on a new Pod

Repeat steps 1–3 (step 3 finds the existing identity file), `prepare`, step 5, then step 6 with the same tag.
Sealed 03b stages are reused and its clock continues; mirrored 04 units come back from the volume and verified
arms are skipped. A foreign writer marker stops the run for investigation; `--resume_own` only after checking.
After a chain stop before 04, a human records the 03b sign-off
(`$PY -m pilot_eval.neuron_audit_qualify signoff --run_dir <q-v1 dir> --reviewer <name> --decision accept|reject`)
and runs 04 directly:
`$PY -m pilot_eval.neuron_audit_production --plan $WORK/plan.yaml --config $WORK/production-v1.json`.

## 8. What 04 does

- Uploads the 03b outputs as a verified qualification package, then arm by arm: sealed units mirrored to `$VOL`,
  each arm packaged with exactly its frozen batch list and verified by fresh download, local units pruned.
- On completion: reference-only final manifest read back, then the Pod stops (`runpodctl pod stop $RUNPOD_POD_ID`).

## 9. Off the Pod: clean restore, 05 and 06 (CPU)

```bash
python -m pilot_eval.neuron_audit_export restore-final --progress $VOL/runs/joint-neuron-adapter-audit/production/p-v1/meta/progress.json \
    --clean_root ./restored-v1 --require_complete
```

Without the volume, write `{"repo":…, "revision":…, "path":…, "sha256":…}` from the final manifest record to
a file and pass `--final_record <file>`. The receipt lists `store_roots` (one per arm) and
`restored_extras.qualification_package` (the 03b run directory). Then:

```bash
python -m pilot_eval.neuron_audit_selection --plan plan.yaml --stores <store_roots…> \
    --sequences_run prep/runs/joint-neuron-adapter-audit/sequences/sequences-v2 \
    --qualification_dir restored-v1/qualification/q-v1 --output_root out --run_id s-v1 --layers 0-27
python -m pilot_eval.neuron_audit_report --plan plan.yaml --selection_dir out/runs/joint-neuron-adapter-audit/selection/s-v1 \
    --stores <store_roots…> --sequences_run prep/runs/joint-neuron-adapter-audit/sequences/sequences-v2 \
    --qualification_dir restored-v1/qualification/q-v1 --output_root out --run_id report-v1 --layers 0-27
```

Finally commit the approved `plan.yaml` (as `.scratch/joint-neuron-adapter-audit/frozen_plan_v1.approved.yaml`)
with the run receipts.

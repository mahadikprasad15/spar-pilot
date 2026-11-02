# Sampling SFT Stage 0 overnight run (L4)

One wrapper measures the three trained Sampling SFT adapters (k0, k2, k5) with the
original Stage 0 instruments, on the same L4 Pod and `sl-0` volume that trained them:

1. **Batch agreement**: on the fixed first 16 cohort questions, for every arm and both
   batch shapes (16 and the tail 6), batched-vs-singleton FP32 logits within the
   unchanged 1e-3 gate, and batched greedy replay matching fresh singleton answers
   exactly (or at a measured near tie). An arm that fails generates one question at a time.
2. **Behaviour**: greedy answers to the 150 held-out questions, scored with
   gsm8k-flexible-v3 and strict, in the original Stage 0 batch membership (9 x 16 + 6).
3. **Fixed-context KL and effective writes**: the original 150 GSM8K + 150 FineWeb
   sequences (pinned by file hash), singleton FP32, FP64 output-head KL (variant || untuned),
   untuned and shared Gold checkpoint-0 zero controls.
4. **Geometry (CPU)**: base-weight SVD and principal-subspace overlap of each arm's
   factors at steps 0, 8, 16, 32, 64; update sizes, trajectory and cross-arm cosines.
5. **Export**: one sealed bundle to the private HF repo, verified by a fresh download
   file by file; then, with `--stop-pod`, the Pod is stopped.

Each measurement runs as a child process with a time limit and its own log. A crashed
child is relaunched once and resumes from sealed units; an integrity failure or timeout
is not retried. The wrapper re-verifies each product on CPU before marking it complete.
Rerunning the same command after any stop continues where it left off. A Pod restart does
not block resuming; a different GPU, driver or package set does.

Old Gold SFT and GRPO results are not regenerated. The original Stage 0 measured them with
the same instruments on a Colab L4; any join should disclose the environment recorded in
each product (`meta/runtime.json`).

## 1. Restore the fixed contexts (once)

The Stage 0 fixed-context payload is not on `sl-0`. From the Mac (needs the user's
go-ahead; uploads 7 MB of public-dataset text to the private repo):

```bash
cd ~/spar/pilot1/artifacts && hf upload Prasadmahadik/spar-sampling-sft-artifacts plans/pilot3-inputs-v1/activation.prepared.json plans/pilot3-inputs-v1/activation.prepared.json --repo-type dataset --commit-message "Stage 0 fixed-context inputs"
```

```bash
cd ~/spar/pilot1/artifacts && hf upload Prasadmahadik/spar-sampling-sft-artifacts runs/pilot-3/Qwen--Qwen2.5-1.5B-Instruct/gsm8k-fineweb/heldout150-control150-seed42/float32/pilot3-inputs-v1/inputs/items.json runs/pilot-3/Qwen--Qwen2.5-1.5B-Instruct/gsm8k-fineweb/heldout150-control150-seed42/float32/pilot3-inputs-v1/inputs/items.json --repo-type dataset --commit-message "Stage 0 fixed-context inputs"
```

On the Pod, copy them into the artifact root at the same paths (the precheck checks the hash):

```bash
cd /root/spar-pilot-training && /root/training-venv/bin/python -c 'import shutil;from pathlib import Path;from huggingface_hub import hf_hub_download;r=Path("/workspace/artifacts");[(r/p).parent.mkdir(parents=True,exist_ok=True) or shutil.copyfile(hf_hub_download("Prasadmahadik/spar-sampling-sft-artifacts",p,repo_type="dataset"),r/p) for p in ["plans/pilot3-inputs-v1/activation.prepared.json","runs/pilot-3/Qwen--Qwen2.5-1.5B-Instruct/gsm8k-fineweb/heldout150-control150-seed42/float32/pilot3-inputs-v1/inputs/items.json"]]'
```

## 2. Precheck (read-only except CPU tests)

```bash
cd /root/spar-pilot-training && git fetch origin sampling-stage0-l4 && git switch sampling-stage0-l4
```

```bash
cd /root/spar-pilot-training && scripts/sampling_stage0_precheck.sh
```

Proceed only if it exits successfully and shows: the reviewed commit; an L4; the three
arms verified; `fixed inputs are the Stage 0 payload: True`; no existing overnight state
(or the state you expect to resume); all CPU tests passing. `same as Stage 0` for the base
hash is recorded either way; `False` means the base differs from the Stage 0 base and
must be reported with any comparison.

## 3. Launch (detached; survives SSH drops)

```bash
tmux new -d -s stage0 'cd /root/spar-pilot-training && /root/training-venv/bin/python -m pilot_eval.sampling_stage0_overnight --root /workspace/artifacts --training-jobs TRAINING_JOBS --inputs plans/pilot3-inputs-v1/activation.prepared.json --name l4-stage0-v1 --volume-id cmuptiucb000007l5dnohe861 --stop-pod 2>&1 | tee -a /workspace/artifacts/runs/stage0/overnight-l4-stage0-v1.console.log'
```

Replace `TRAINING_JOBS` with the `training jobs:` path the precheck prints. Default child
limits are 2 h (agreement), 4 h (behaviour), 3 h (KL), 2 h (geometry); override with
`--hours-<stage>`.

## 4. Status (any time; takes no lock, writes nothing)

```bash
cd /root/spar-pilot-training && /root/training-venv/bin/python -m pilot_eval.sampling_stage0_overnight status --root /workspace/artifacts --name l4-stage0-v1
```

Exit codes: 0 exported and verified; 1 stopped (see `status.json` and
`failure-traceback.txt`; rerun the launch command to resume); 2 another wrapper holds this
run. Never start a second wrapper to "check"; use the status command.

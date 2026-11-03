# Sampling Stage 0 principal-cut overnight run

This is a separate run over the existing sealed Sampling Stage 0 geometry and behaviour plans. It uses each endpoint's rank-one adapter and the top singular vectors of the **base** weight matrix covering 50% of squared singular-value energy. Per arm it evaluates original, principal output and input removal, three random-output controls (seeds 1000–1002), and output/input per-module magnitude controls. With k0, k2 and k5 and 150 questions this is 24 variants and 3,600 fresh singleton responses. There is no inherited batch agreement. The run can take longer than overnight; no completion-time promise is implied.

Use the checked-out reviewed revision, the original RunPod persistent volume and the already provisioned, pinned A100 environment from `sampling-stage0-overnight.md`. The source geometry plan, original behaviour plan, model snapshot, and sealed training checkpoints must all be present under the same artifact root. The geometry source is reverified, and the base SVD is recomputed once per module; its principal bases, edited factors and checks are sealed for resume. This extension does not modify existing training code or source artifacts. The private HF preflight creates a small JSON commit before the build, and the export is accepted only after complete response verification and archive/index readback at an exact revision.

After the reviewed branch is available on the Pod and the shell is in that checkout:

```bash
export OUTPUT_ROOT=/workspace/artifacts
export GEOMETRY_PLAN="$OUTPUT_ROOT/plans/a100-stage0-v1/sampling-stage0.geometry.json"
export RUN_NAME=a100-stage0-cuts-v1
export HF_REPO=Prasadmahadik/spar-sampling-sft-artifacts
export BUILD_SECONDS=86400
export EVAL_SECONDS=86400
export TRANSFER_SECONDS=3600
# HF_TOKEN must already be present in the shell environment or configured with hf auth.
tmux new-session -d -s sampling-cuts 'bash scripts/sampling_stage0_cuts_overnight.sh'
```

Inspect progress and logs:

```bash
RUN_DIR="$OUTPUT_ROOT/runs/stage0/$RUN_NAME/principal-cuts"
python -m pilot_eval.sampling_stage0_cut_workflow status --run-dir "$RUN_DIR"
tail -f "$RUN_DIR/logs/overnight.log"
tmux attach -t sampling-cuts
```

Detach with `Ctrl-b`, then `d`. On failure, inspect the status and log, fix the cause without changing frozen sources or implementation, and rerun the same script with the same environment and name. Completed cut modules and response records are checked and skipped. A runtime hardware/software difference, source drift, integrity failure, or modified seal stops resume; do not delete evidence to bypass it. A deadline stops the child and leaves partial artifacts; increase the relevant limit only after reviewing the status. The script does not stop or terminate the Pod.

Successful completion records the exact private Hub revision and archive/index hashes in `meta/hf-upload.json`. The local response file contains raw tokens, extracted scores, cap information and prompts; automatic scoring has not received human review. Real A100 execution and adapter switching require a live smoke check: the CPU fixtures alone do not establish GPU correctness, speed or memory use.

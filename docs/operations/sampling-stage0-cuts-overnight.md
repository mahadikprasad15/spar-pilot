# Sampling Stage 0 principal-cut overnight run

A separate run over the existing sealed Sampling Stage 0 geometry and behaviour plans. It uses each endpoint's rank-one adapter and the top singular vectors of the **base** weight matrix covering 50% of squared singular-value energy (the original Stage 0 edit maths, unchanged). Per arm it evaluates the original, principal output and input removal, three random-output controls (seeds 1000-1002), and output/input per-module magnitude controls: 24 variants x 150 questions = 3,600 responses.

## Stages

1. **prepare / preflight**: freeze the plan; prove private HF write and exact-revision readback.
2. **build (CPU)**: base SVD per module, edited factors and 24 PEFT adapters, all sealed and resumable.
3. **batch agreement**: the original Stage 0 batch16 protocol (ADR 0014). Every variant gets batched-vs-singleton FP32 answer-position logits and the cached decode step at the 16 and tail-6 shapes, within the unchanged 1e-3 gate. The first and last arm's principal output cuts (as V1/V5 were for Gold/GRPO) get batched greedy replay against fresh singleton answers: exact, or a measured near tie. A variant batches only if its numeric units pass and its arm's replay passes; otherwise it generates one question at a time.
4. **evaluation**: ordered 16-question groups (tail 6), identical across variants, sealed per variant and group. Requires the agreement's hardware/software environment.
5. **verify**: every unit, summary, adapter and agreement binding re-checked on CPU; `results/consistency.json` compares each arm's fresh *original* answers with the a100-stage0-v1 behaviour answers (same adapters, decoder and batch membership; exact matches are expected in the same environment).
6. **export**: verified tree to the private HF repo, read back from an empty cache. Base singular vectors (`bases.npz`, about 1.6 GB) stay sealed on the volume and are not uploaded; they are deterministic from the hashed base weights.

Timing on the A100 measured on 2026-11-02: one question at a time 9.4 tokens/s (about 34 s per answer); batch 16 about 4.9 s per answer. Expect roughly 25 min build, 30-40 min agreement, about 5 h evaluation. A variant that falls back to one at a time adds about 1.4 h. The script does not stop the Pod.

## Launch

Use the pinned A100 environment, the original volume and a detached checkout of the reviewed commit.

```bash
tmux new-session -d -s sampling-cuts 'cd /root/spar-pilot-stage0 && export PYTHON=/root/stage0-venv/bin/python OUTPUT_ROOT=/workspace/artifacts GEOMETRY_PLAN=/workspace/artifacts/plans/a100-stage0-v1/sampling-stage0.geometry.json RUN_NAME=a100-stage0-cuts-v1 HF_REPO=Prasadmahadik/spar-sampling-sft-artifacts HF_HUB_CACHE=/root/hf-stage0-cache && bash scripts/sampling_stage0_cuts_overnight.sh'
```

`PYTHON` must be the pinned venv interpreter; the script refuses to start otherwise and checks the pinned package versions first. Optional: `GPU` (default A100), `RETRIES` (default 1), `BUILD_SECONDS`/`AGREEMENT_SECONDS` (default 4 h), `EVAL_SECONDS` (default 12 h), `TRANSFER_SECONDS` (default 1 h).

Each crashed stage is relaunched once and resumes from sealed units. An integrity failure (base or adapter changed, including during an adapter switch) is recorded in `meta/integrity-failure.json`, is never retried, and blocks reuse until investigated.

## Watch

```bash
tail -n 30 -F /workspace/artifacts/runs/stage0/a100-stage0-cuts-v1/principal-cuts/logs/overnight.log
```

```bash
/root/stage0-venv/bin/python -m pilot_eval.sampling_stage0_cut_workflow status --run-dir /workspace/artifacts/runs/stage0/a100-stage0-cuts-v1/principal-cuts
```

Status shows the stage, each variant's batch size once the agreement finishes, progress, the consistency report and the export receipt. After any stop, inspect status and log, then rerun the same command: completed work is checked and skipped. Do not change the checkout mid-run: the plan freezes implementation hashes.

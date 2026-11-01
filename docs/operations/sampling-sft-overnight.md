# Sampling SFT overnight run (L4)

One wrapper runs: independent L4 gate scoring → report → recording the user's pre-authorized
three-arm selection → k0, k2, k5 training (each from Gold checkpoint-0, 64 original steps) →
one sealed HF export → a fresh-download, per-file check → optional own-Pod stop.

Each finished stage leaves a `*.complete.json` pointer under
`runs/sampling-sft/overnight/<name>/`. A rerun skips those stages. An uncertain or failed
job (`launch.json` without a successful work seal) is never relaunched: the wrapper stops
and records `stopped` with the stage, error and `failure-traceback.txt`. A failure never
stops the Pod; completed work stays on the Global Volume.

## Before launch (on the Pod, read-only except CPU tests)

```bash
cd /root/spar-pilot-training && scripts/sampling_sft_overnight_precheck.sh
```

Proceed only if:
- the commit is the reviewed one
- the score job has no `launch.json` or `execution.json`
- `command[0]` is the training venv's Python
- the CPU tests pass. `test_sampling_sft_training_production_path.py` must **pass**, not skip.

Then set `--hours-k*` so that 1.5× the value is comfortably above the expected arm time;
the child is killed at that bound.

## Launch (detached; survives SSH drops)

```bash
tmux new -d -s overnight 'cd /root/spar-pilot-training && /root/training-venv/bin/python -m pilot_eval.sampling_sft_overnight --root /workspace/artifacts --gate plans/l4-gate-v2/sampling-sft.gate.json --score-job runs/sampling-sft/sampling-sft-v1/jobs/l4-score-v2/job.plan.json --data plans/l4-training-data-v1/sampling-sft.training-data.json --name l4-night-v1 --volume-id cmuptiucb000007l5dnohe861 --rate RATE --reviewer "Prasad Mahadik" --hours-k0 2 --hours-k2 2 --hours-k5 2 --setup-hours 0.5 --stop-pod 2>&1 | tee -a /workspace/artifacts/runs/sampling-sft/overnight-l4-night-v1.console.log'
```

## Status (any time; takes no lock, writes nothing)

```bash
cd /root/spar-pilot-training && /root/training-venv/bin/python -m pilot_eval.sampling_sft_overnight status --root /workspace/artifacts --name l4-night-v1
```

```bash
tmux attach -t overnight
```

Exit codes: 0 exported and verified; 1 stopped on failure (see `status.json`); 2 another
wrapper already holds this run (its status is left untouched). Never start a second
wrapper to "check" — use the status command.

## What "verified export" means here

Two checks pass. First, the HF server's copy of the archive is read back and its hash
matches. Second, a fresh download into an empty cache matches every file's SHA256 in the
sealed manifest. Both checks run on the worker Pod itself, not on an external host.

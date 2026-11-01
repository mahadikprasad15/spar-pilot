#!/usr/bin/env bash
# Read-only Pod checks before the unattended Sampling SFT run, plus CPU-only tests.
# Launches nothing, reserves nothing, prints no credentials.
set -euo pipefail
PY=${PY:-/root/training-venv/bin/python}
ROOT=${ROOT:-/workspace/artifacts}
JOB=${JOB:-runs/sampling-sft/sampling-sft-v1/jobs/l4-score-v2}
cd "$(dirname "$0")/.."

echo "== Code and interpreter"
git log --oneline -1
git status --short
"$PY" -c 'import sys,importlib.metadata as m;print(sys.executable,sys.prefix);print({n:m.version(n) for n in ["torch","transformers","peft","trl","huggingface-hub"]})'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-MISSING}"; df -hT /workspace | tail -1

echo "== Score job (must be prepared and never launched)"
ls "$ROOT/$JOB"
for f in launch.json execution.json work-complete.json; do
  if [ -e "$ROOT/$JOB/$f" ]; then
    echo "PRESENT: $f  <-- inspect; do not relaunch" >&2
    exit 1
  fi
done
"$PY" - "$ROOT/$JOB" <<'EOF'
import json,sys,pathlib
d=pathlib.Path(sys.argv[1]);job=json.loads((d/'job.plan.json').read_text());r=job['request']
print('command[0]:',r['command'][0]);print('estimated_hours:',r['estimated_hours'],'(child timeout = 1.5x)')
events=json.loads((d/'state.json').read_text())['events'];print('last event:',events[-1]['kind'] if events else None)
EOF

echo "== Budget ledger"
"$PY" -c "import json;from pilot_eval.sampling_sft_budget import status;s=status('$ROOT');print(json.dumps({k:s[k] for k in ['committed_usd','cap_exceeded','review_required']}))"

echo "== Production training loop with real PEFT (must pass, not skip)"
production_output=$(SAMPLING_SFT_CPU_PYTHON="$PY" HF_HUB_OFFLINE=1 "$PY" -m pytest -q -p no:cacheprovider \
  tests/test_sampling_sft_training_production_path.py) || { echo "$production_output"; exit 1; }
echo "$production_output"
if ! echo "$production_output" | grep -Eq '(^|[ ,])1 passed([ ,]|$)' || echo "$production_output" | grep -Eq '[0-9]+ skipped'; then
  echo 'Production-path test did not pass without skips.' >&2
  exit 1
fi
echo "== Focused wrapper and training tests"
SAMPLING_SFT_CPU_PYTHON="$PY" HF_HUB_OFFLINE=1 "$PY" -m pytest -q -p no:cacheprovider \
  tests/test_sampling_sft_overnight.py tests/test_sampling_sft_training.py

#!/usr/bin/env bash
# Read-only Pod checks before the unattended sampling Stage 0 run, plus CPU-only tests.
# Launches nothing, writes no artifacts, prints no credentials.
set -euo pipefail
PY=${PY:-/root/training-venv/bin/python}
ROOT=${ROOT:-/workspace/artifacts}
NAME=${NAME:-l4-stage0-v1}
TRAIN_NAME=${TRAIN_NAME:-l4-night-v1-train}
INPUTS=${INPUTS:-plans/pilot3-inputs-v1/activation.prepared.json}
cd "$(dirname "$0")/.."

echo "== Code and interpreter"
git log --oneline -1
git status --short
"$PY" -c 'import sys,importlib.metadata as m;print(sys.executable);print({n:m.version(n) for n in ["torch","transformers","peft","trl","huggingface-hub"]})'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-MISSING}"; df -hT /workspace /tmp | tail -2
echo "artifact mount (must be the network volume at /workspace):"; grep " /workspace " /proc/mounts || echo "NO /workspace MOUNT"

echo "== Training record and sources (CPU verification of the three trained arms)"
JOBS=$(find "$ROOT/runs/sampling-sft" -path "*/pipeline/$TRAIN_NAME/training-jobs.json" 2>/dev/null | head -2)
[ "$(echo "$JOBS" | grep -c .)" = 1 ] || { echo "Expected exactly one $TRAIN_NAME/training-jobs.json; found: $JOBS" >&2; exit 1; }
echo "training jobs: ${JOBS#$ROOT/}"
"$PY" - "$ROOT" "${JOBS#$ROOT/}" "$INPUTS" <<'EOF'
import json,sys
from pathlib import Path
from pilot_eval.sampling_stage0_overnight import sources_manifest
from pilot_eval.sampling_stage0_evaluation import sources
from pilot_eval.sampling_stage0_drift import STAGE0_ITEMS_FILE_SHA256
from pilot_eval.sampling_stage0_geometry import STAGE0_BASE_SHA256
from pilot_eval.training import file_hash
import tempfile
root=Path(sys.argv[1]);manifest=sources_manifest(root,sys.argv[2])
print('gold config:',manifest['gold_config']);print('model dir:',manifest['weights_dir'])
with tempfile.TemporaryDirectory(dir=root) as temp: # sources() wants an in-root manifest; removed on exit
    path=Path(temp)/'sources.json';path.write_text(json.dumps(manifest))
    source,items=sources(path,root)
print('arms verified:',sorted(source['variants']),'| cohort:',len(items),'| shared checkpoint-0:',source['zero_adapter']['adapter_sha256'][:12])
print('base sha256:',source['model_identity']['base_sha256'][:12],'| same as Stage 0:',source['model_identity']['base_sha256']==STAGE0_BASE_SHA256)
prepared=json.loads((root/sys.argv[3]).read_text());items_path=root/prepared['items_path']
if not items_path.is_file():sys.exit('Fixed-context items missing: restore them from HF first.')
print('fixed inputs are the Stage 0 payload:',file_hash(items_path)==STAGE0_ITEMS_FILE_SHA256)
if file_hash(items_path)!=STAGE0_ITEMS_FILE_SHA256:sys.exit(1)
EOF

echo "== Existing overnight state (read-only)"
"$PY" -m pilot_eval.sampling_stage0_overnight status --root "$ROOT" --name "$NAME" | head -40

echo "== Focused CPU tests"
"$PY" -m pytest -q -p no:cacheprovider tests/test_sampling_stage0_evaluation.py tests/test_sampling_stage0_agreement.py \
  tests/test_sampling_stage0_drift.py tests/test_sampling_stage0_geometry.py tests/test_sampling_stage0_overnight.py

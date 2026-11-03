#!/usr/bin/env bash
# Run from the checked-out repository on the allocated machine. See runbook.
set -Eeuo pipefail
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the existing artifact root}"
: "${GEOMETRY_PLAN:?Set GEOMETRY_PLAN to the existing source geometry plan}"
: "${RUN_NAME:?Set RUN_NAME to a new safe run name}"
: "${HF_REPO:?Set HF_REPO to the private dataset repository}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe RUN_NAME' >&2; exit 2; }
BUILD_SECONDS=${BUILD_SECONDS:-86400}
EVAL_SECONDS=${EVAL_SECONDS:-86400}
TRANSFER_SECONDS=${TRANSFER_SECONDS:-3600}
RUN_DIR="$OUTPUT_ROOT/runs/stage0/$RUN_NAME/principal-cuts"
PLAN="$OUTPUT_ROOT/plans/$RUN_NAME/sampling-stage0.cuts.json"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/meta"
exec > >(tee -a "$RUN_DIR/logs/overnight.log") 2>&1
stage=prepare
write_status() {
  STAGE_VALUE="$stage" STATE_VALUE="$1" EXIT_VALUE="$2" STATUS_FILE="$RUN_DIR/meta/overnight-status.json" python - <<'PY'
import datetime,json,os,pathlib
p=pathlib.Path(os.environ['STATUS_FILE'])
tmp=p.with_suffix('.json.tmp')
tmp.write_text(json.dumps(dict(stage=os.environ['STAGE_VALUE'],state=os.environ['STATE_VALUE'],
    exit_code=int(os.environ['EXIT_VALUE']),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
tmp.replace(p)
PY
}
on_error() { code=$?; trap - ERR; write_status failed "$code"; echo "Stage $stage failed (exit $code)."; exit "$code"; }
trap on_error ERR
write_status running 0
python -m pilot_eval.sampling_stage0_cut_workflow prepare --geometry "$GEOMETRY_PLAN" --output-root "$OUTPUT_ROOT" --name "$RUN_NAME"
stage=preflight; write_status running 0
timeout "$TRANSFER_SECONDS" python -m pilot_eval.sampling_stage0_cut_workflow preflight --plan "$PLAN" --output-root "$OUTPUT_ROOT" --repo "$HF_REPO"
stage=build; write_status running 0
python -m pilot_eval.sampling_stage0_cut_workflow build --plan "$PLAN" --output-root "$OUTPUT_ROOT" --deadline-seconds "$BUILD_SECONDS"
stage=evaluation; write_status running 0
python -m pilot_eval.sampling_stage0_cut_workflow run --plan "$PLAN" --output-root "$OUTPUT_ROOT" --repo "$HF_REPO" --deadline-seconds "$EVAL_SECONDS"
stage=verify; write_status running 0
python -m pilot_eval.sampling_stage0_cut_workflow verify --run-dir "$RUN_DIR"
stage=export; write_status running 0
timeout "$TRANSFER_SECONDS" python -m pilot_eval.sampling_stage0_cut_workflow export --run-dir "$RUN_DIR" --output-root "$OUTPUT_ROOT" --repo "$HF_REPO"
stage=complete; write_status complete 0
echo "Verified cut evaluation and export complete: $RUN_DIR"

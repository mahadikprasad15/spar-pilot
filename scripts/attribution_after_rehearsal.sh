#!/usr/bin/env bash
# After a rehearsal on this Pod: if it completed, launch the production run unattended; otherwise stop the Pod.
#
#   bash scripts/attribution_after_rehearsal.sh            (needs HF_TOKEN and RUNPOD_POD_ID in the environment)
#
# User decision 2026-11-16: keep the Pod between a passing rehearsal and the production run (GPUs are hard to get),
# and stop it at the end of the production run. During the rehearsal `runpodctl` is held aside
# (/usr/bin/runpodctl.held) so the rehearsal's own teardown cannot stop the Pod; this script puts it back.
# Every path ends with the Pod stopped: a failed rehearsal, a failed hand-over, or the end of production (the
# chain stops it; the final stop here is then a no-op).
set -u
REPO=${REPO:-/root/spar-pilot} WORK=${WORK:-/root/attr} VOL=${VOL:-/workspace/neuron-audit}
BRANCH=${BRANCH:-claude/bold-gauss-5qkzwi}
REHEARSAL_RECORD=${REHEARSAL_RECORD:-$WORK/launch_record.json}
RUNPODCTL=${RUNPODCTL:-/usr/bin/runpodctl}
POLL=${POLL:-30}
log() { echo "$(date -u +%FT%TZ) after-rehearsal: $*" | tee -a "$VOL/after_rehearsal.log"; }
restore_stop() { if [ -e "$RUNPODCTL.held" ] && [ ! -e "$RUNPODCTL" ]; then mv "$RUNPODCTL.held" "$RUNPODCTL"; fi; }
stop_pod() {
  log "stopping the Pod: $1"
  restore_stop
  "$RUNPODCTL" stop pod "$RUNPOD_POD_ID" || "$RUNPODCTL" pod stop "$RUNPOD_POD_ID"
  exit 1
}
: "${HF_TOKEN:?export HF_TOKEN first (the production export needs it)}" "${RUNPOD_POD_ID:?RUNPOD_POD_ID is not set}"
done_file="$VOL/attribution/chain/done"

log "waiting for the rehearsal to finish ($done_file)"
while [ ! -e "$done_file" ]; do sleep "$POLL"; done
while pgrep -f "attribution_chain run --record $REHEARSAL_RECORD" >/dev/null; do sleep 5; done
state=$(cat "$done_file")
log "rehearsal finished: $state"
[ "$state" = complete ] || stop_pod "the rehearsal did not complete ($state); production not started"

mv "$VOL/attribution" "$VOL/attribution-rehearsal" || stop_pod "could not move the rehearsal's folder aside"
restore_stop
cd "$REPO" && git pull -q --ff-only origin "$BRANCH" || stop_pod "git pull failed"
log "code at $(git rev-parse --short HEAD)"
# shellcheck disable=SC1091
source /root/venv/bin/activate || stop_pod "no venv"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH="$REPO" \
  HF_HOME=${HF_HOME:-/root/hf-home}
cp .scratch/adapter-neuron-attribution/launch-record-production.yaml "$WORK/production-draft.yaml" \
  || stop_pod "no production draft"
python -m pilot_eval.attribution_chain seal --draft "$WORK/production-draft.yaml" \
  --out "$WORK/production_record.json" --inputs "$WORK/inputs.json" --work "$WORK" \
  || stop_pod "seal refused the production record"

log "launching production"
bash scripts/attribution_overnight.sh "$WORK/production_record.json" "$WORK/production-local" "$VOL" 2>&1 \
  | tee "$WORK/production.console.log"
log "production launcher exited (${PIPESTATUS[0]})"
sleep 60
stop_pod "end of the production launch (a no-op if the chain already stopped the Pod)"

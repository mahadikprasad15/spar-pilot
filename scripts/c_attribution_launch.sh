#!/usr/bin/env bash
# Launch the C-attribution chain (ticket 09) with its independent watchdog, deadline disabled.
#
#   scripts/c_attribution_launch.sh <sealed launch_record.json> <local work dir> <volume dir>
#
# One launch per run ID. Volume state lives under <volume>/c-attribution/<run-id>/. There are no time caps
# (user, 2026-11-18): the watchdog never stops a live writer for elapsed time; it acts only when the writer
# exits without writing the done file (a crash, including at startup), then copies sealed outputs and requests
# the Pod stop with both command forms. There is no automatic rehearsal -> production handover.
set -euo pipefail
record=$1 local_dir=$2 volume=$3
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=${PY:-python}

# Anything that fails before the chain and its watchdog exist leaves the Pod running with nothing to stop it.
# Say so loudly; never stop here (a refused second launch must not stop a run that is already going).
not_started() {
  {
    echo
    echo "=================================================================================="
    echo "NO RUN STARTED: $1"
    echo "This Pod is still running and billing. Fix the problem and relaunch, or stop it with:"
    echo "    runpodctl stop pod ${RUNPOD_POD_ID:-<pod id>}"
    echo "=================================================================================="
  } >&2
  exit 1
}

[ -f "$record" ] || not_started "no launch record at $record"
mkdir -p "$local_dir" || not_started "cannot create $local_dir"
# Ownership is taken only after every check passes: the lock refuses a second launch on this Pod (inherited by
# the chain and the watchdog, so it is held while either runs); precheck refuses a finished run ID and a live,
# foreign or ambiguous writer marker on the volume.
exec 9>"$local_dir/launch.lock"
flock -n 9 || not_started "another launch on this Pod holds $local_dir/launch.lock (is a run already going in tmux?)"
$PY -m pilot_eval.c_attribution_chain precheck --record "$record" --local "$local_dir" --volume "$volume" \
  || not_started "precheck refused the launch record or found another writer (message above)"
paths=$($PY -m pilot_eval.c_attribution_chain paths --record "$record" --local "$local_dir" --volume "$volume") \
  || not_started "could not read the run paths from the record"
info=$($PY -c 'import json,sys; p=json.loads(sys.argv[1]); s=[c.replace(" ", "\x1f") for c in p["stop"]]; print(p["chain_dir"], s[0], s[1], p["copy_timeout"], p["stop_timeout"])' "$paths") \
  || not_started "could not parse the run paths"
read -r chain_dir stop1 stop2 copy_t stop_t <<<"$info"
mkdir -p "$chain_dir" || not_started "cannot create $chain_dir"

owner="$chain_dir/watchdog_owner"
token="$(hostname)-$$-$(date +%s)-$RANDOM"
printf '%s\n' "$token" > "$owner"

$PY -m pilot_eval.c_attribution_chain run --record "$record" --local "$local_dir" --volume "$volume" \
  > >(tee -a "$local_dir/chain.stdout.log" "$chain_dir/console.log") 2>&1 &   # the volume copy survives the Pod stop
chain_pid=$!

WATCHDOG_COPY="$PY -m pilot_eval.c_attribution_chain watchdog-copy --record '$record' --local '$local_dir' --volume '$volume'" \
WATCHDOG_STOP_1="${stop1//$'\x1f'/ }" WATCHDOG_STOP_2="${stop2//$'\x1f'/ }" \
WATCHDOG_SUMMARY="$chain_dir/watchdog_summary.json" WATCHDOG_PROGRESS="$chain_dir/progress.json" \
WATCHDOG_HANDLED="$chain_dir/last_attempt_end.json" WATCHDOG_STOP_REQUIRES_COPY=1 \
  setsid nohup bash scripts/attribution_watchdog.sh none "$chain_pid" "$chain_dir/done" \
  120 "$copy_t" "$stop_t" "$owner" "$token" >> "$chain_dir/watchdog.log" 2>&1 < /dev/null &
echo "chain pid $chain_pid; watchdog deadline disabled (no time caps); volume state $chain_dir"
wait "$chain_pid"

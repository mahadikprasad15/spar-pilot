#!/usr/bin/env bash
# Launch the attribution overnight chain (ticket 11) with its independent watchdog.
#
#   scripts/attribution_overnight.sh <approved launch_record.json> <local work dir> <volume dir>
#
# The launch record must be the user-approved, sealed record (its .seal.json beside it) for this commit.
# The watchdog runs in its own session (setsid), outside the chain's Python, timed from the *persisted*
# remaining total cap + margin; the chain cancels it by writing <volume>/attribution/chain/done.
set -euo pipefail
record=$1 local_dir=$2 volume=$3
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8

# Review round 7 (2): anything that fails before the chain and its watchdog exist leaves the Pod running with
# nothing to stop it. Say so loudly; never stop automatically here (a refused second launch must not stop a run
# that is already going, and a setup mistake should not cost the container disk).
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

mkdir -p "$local_dir" "$volume/attribution/chain" || not_started "cannot create $local_dir or $volume/attribution/chain"
# Review round 7 (1): ownership is taken only after every check below passes. A lock on the container disk
# refuses a second launch on this Pod (it is inherited by the chain and its watchdog, so it is held while either
# runs); precheck refuses a live, foreign or ambiguous writer marker on the volume (another Pod).
exec 9>"$local_dir/launch.lock"
flock -n 9 || not_started "another launch on this Pod holds $local_dir/launch.lock (is a run already going in tmux?)"
python -m pilot_eval.attribution_chain precheck --record "$record" --local "$local_dir" --volume "$volume" \
  || not_started "precheck refused the launch record or found another writer (message above)"
deadline=$(python -m pilot_eval.attribution_chain deadline --record "$record" --local "$local_dir" --volume "$volume") \
  || not_started "could not compute the watchdog deadline (message above)"
info=$(python - "$record" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
cmds = [c.format(id=r['pod_id']).replace(' ', '\x1f') for c in r['pod_stop_commands']]
print(r['pod_id'], cmds[0], cmds[-1], r['timeouts_seconds']['copy'], r['timeouts_seconds']['stop_attempt'])
PY
) || not_started "could not read the Pod id and stop commands from the record"
read -r pod stop1 stop2 copy_t stop_t <<<"$info"

# Every check passed: take ownership. Any watchdog left by an earlier (finished or killed) attempt sees a new
# token and stands down.
owner="$volume/attribution/chain/watchdog_owner"
token="$(hostname)-$$-$(date +%s%N)"
printf '%s\n' "$token" > "$owner"
rm -f "$volume/attribution/chain/done"

python -m pilot_eval.attribution_chain run --record "$record" --local "$local_dir" --volume "$volume" \
  > >(tee -a "$local_dir/chain.stdout.log" "$volume/attribution/chain/console.log") 2>&1 &   # the volume copy survives the Pod stop
chain_pid=$!

WATCHDOG_COPY="python -m pilot_eval.attribution_chain watchdog-copy --local '$local_dir' --volume '$volume'" \
WATCHDOG_STOP_1="${stop1//$'\x1f'/ }" WATCHDOG_STOP_2="${stop2//$'\x1f'/ }" \
WATCHDOG_SUMMARY="$volume/attribution/chain/watchdog_summary.json" \
  setsid nohup bash scripts/attribution_watchdog.sh "$deadline" "$chain_pid" "$volume/attribution/chain/done" \
  120 "$copy_t" "$stop_t" "$owner" "$token" >> "$volume/attribution/chain/watchdog.log" 2>&1 < /dev/null &
echo "chain pid $chain_pid; watchdog deadline $(date -u -d "@${deadline%.*}" +%FT%TZ) for Pod $pod"
wait "$chain_pid"

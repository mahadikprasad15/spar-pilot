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
mkdir -p "$local_dir" "$volume/attribution/chain"
rm -f "$volume/attribution/chain/done"

deadline=$(python -m pilot_eval.attribution_chain deadline --record "$record" --local "$local_dir" --volume "$volume")
read -r pod stop1 stop2 copy_t stop_t <<<"$(python - "$record" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
cmds = [c.format(id=r['pod_id']).replace(' ', '\x1f') for c in r['pod_stop_commands']]
print(r['pod_id'], cmds[0], cmds[-1], r['timeouts_seconds']['copy'], r['timeouts_seconds']['stop_attempt'])
PY
)"

python -m pilot_eval.attribution_chain run --record "$record" --local "$local_dir" --volume "$volume" \
  > >(tee -a "$local_dir/chain.stdout.log") 2>&1 &
chain_pid=$!

WATCHDOG_COPY="python -m pilot_eval.attribution_chain watchdog-copy --local '$local_dir' --volume '$volume'" \
WATCHDOG_STOP_1="${stop1//$'\x1f'/ }" WATCHDOG_STOP_2="${stop2//$'\x1f'/ }" \
WATCHDOG_SUMMARY="$volume/attribution/chain/watchdog_summary.json" \
  setsid nohup bash scripts/attribution_watchdog.sh "$deadline" "$chain_pid" "$volume/attribution/chain/done" \
  120 "$copy_t" "$stop_t" > "$volume/attribution/chain/watchdog.log" 2>&1 < /dev/null &
echo "chain pid $chain_pid; watchdog deadline $(date -u -d "@${deadline%.*}" +%FT%TZ) for Pod $pod"
wait "$chain_pid"

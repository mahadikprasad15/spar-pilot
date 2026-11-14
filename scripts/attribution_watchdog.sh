#!/usr/bin/env bash
# Independent watchdog for the attribution overnight chain (ticket 11); runs outside the chain's Python.
#
#   attribution_watchdog.sh <deadline_epoch> <writer_pid> <done_file> <wait_s> <copy_timeout_s> <stop_timeout_s> \
#       <owner_file> <owner_token>
#
# Ownership (review round 4, 7): every launch writes a fresh token to <owner_file> before starting its chain and
# watchdog. A watchdog acts only while the file still holds *its* token; a watchdog from an earlier (killed)
# attempt sees another token and exits without signalling, copying or stopping anything.
#
# Environment: WATCHDOG_COPY (command copying only sealed, hash-verified units to the volume, e.g.
#   "python -m pilot_eval.attribution_chain watchdog-copy --local L --volume V"), WATCHDOG_STOP_1 and
#   WATCHDOG_STOP_2 (the two Pod stop command forms), WATCHDOG_SUMMARY (watchdog summary JSON path).
# Exits quietly when the done file appears (normal completion). At the deadline: SIGTERM the writer, wait a
# bounded time, SIGKILL if still alive, copy sealed units (timeout), write the summary, request the stop with
# each form (timeout each), rewrite the summary with the outcome. A stop is only ever *requested* from here.
set -u
deadline=$1 pid=$2 done_file=$3 wait_s=$4 copy_t=$5 stop_t=$6 owner_file=${7:?owner file required} token=${8:?token required}
summary=${WATCHDOG_SUMMARY:?WATCHDOG_SUMMARY is required}

still_owner() { [ "$(cat "$owner_file" 2>/dev/null)" = "$token" ]; }

while :; do
  if ! still_owner; then echo "watchdog superseded by a newer attempt; standing down"; exit 0; fi
  if [ -e "$done_file" ]; then echo "watchdog cancelled: chain completed ($(cat "$done_file"))"; exit 0; fi
  if awk -v now="$(date +%s.%N)" -v d="$deadline" 'BEGIN { exit !(now >= d) }'; then break; fi
  sleep 1
done

if ! still_owner; then echo "watchdog superseded by a newer attempt; standing down"; exit 0; fi
echo "watchdog fired at $(date -u +%FT%TZ); signalling writer $pid"
kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 "$wait_s"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
writer_alive=false
if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; writer_alive=true; fi

if ! still_owner; then echo "watchdog superseded while waiting for the writer; not copying or stopping"; exit 0; fi
timeout "$copy_t" bash -c "${WATCHDOG_COPY:?WATCHDOG_COPY is required}"; copy_rc=$?

write_summary() {
  printf '{"fired": true, "fired_unix": %s, "writer_needed_kill": %s, "copy_returncode": %s, "stop": "%s", "stop_returncodes": [%s], "externally_verified": false}\n' \
    "$(date +%s)" "$writer_alive" "$copy_rc" "$1" "$2" > "$summary"
}
write_summary "pending" ""

stop="not-requested" codes=""
for cmd in "${WATCHDOG_STOP_1:?}" "${WATCHDOG_STOP_2:?}"; do
  timeout "$stop_t" bash -c "$cmd"; rc=$?
  codes="${codes:+$codes, }$rc"
  if [ "$rc" -eq 0 ]; then stop="requested"; break; fi
done
write_summary "$stop" "$codes"
echo "watchdog finished: copy rc $copy_rc, stop $stop"
exit 0

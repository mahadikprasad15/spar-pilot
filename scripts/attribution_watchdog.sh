#!/usr/bin/env bash
# Independent watchdog for the attribution overnight chain (ticket 11); runs outside the chain's Python.
#
#   attribution_watchdog.sh <deadline_epoch> <writer_pid> <done_file> <wait_s> <copy_timeout_s> <stop_timeout_s>
#
# Environment: WATCHDOG_COPY (command copying only sealed, hash-verified units to the volume, e.g.
#   "python -m pilot_eval.attribution_chain watchdog-copy --local L --volume V"), WATCHDOG_STOP_1 and
#   WATCHDOG_STOP_2 (the two Pod stop command forms), WATCHDOG_SUMMARY (watchdog summary JSON path).
# Exits quietly when the done file appears (normal completion). At the deadline: SIGTERM the writer, wait a
# bounded time, SIGKILL if still alive, copy sealed units (timeout), write the summary, request the stop with
# each form (timeout each), rewrite the summary with the outcome. A stop is only ever *requested* from here.
set -u
deadline=$1 pid=$2 done_file=$3 wait_s=$4 copy_t=$5 stop_t=$6
summary=${WATCHDOG_SUMMARY:?WATCHDOG_SUMMARY is required}

while :; do
  if [ -e "$done_file" ]; then echo "watchdog cancelled: chain completed ($(cat "$done_file"))"; exit 0; fi
  if awk -v now="$(date +%s.%N)" -v d="$deadline" 'BEGIN { exit !(now >= d) }'; then break; fi
  sleep 1
done

echo "watchdog fired at $(date -u +%FT%TZ); signalling writer $pid"
kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 "$wait_s"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
writer_alive=false
if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; writer_alive=true; fi

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

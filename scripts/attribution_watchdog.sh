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
# Deadline: an epoch, or the literal `none` (C-attribution pilot, user 2026-11-18: no time caps). With `none` a live
# writer is never stopped because time has passed; the watchdog acts only once the writer has exited without the
# done file, so no chain copy can be in flight when it copies (its own copy is bounded by <copy_timeout_s>).
# Optional: WATCHDOG_PROGRESS (the chain's progress JSON) and WATCHDOG_STATUS_EVERY (seconds, default 300) print a
# status line with the writer's last progress update, so an alive but stalled writer can be recognized.
# Exits quietly when the done file appears (normal completion). At the deadline, or as soon as the writer
# has exited without the done file: SIGTERM the writer, wait a
# bounded time, SIGKILL if still alive, copy sealed units (timeout), write the summary, request the stop with
# each form (timeout each), rewrite the summary with the outcome. A stop is only ever *requested* from here.
set -u
deadline=$1 pid=$2 done_file=$3 wait_s=$4 copy_t=$5 stop_t=$6 owner_file=${7:?owner file required} token=${8:?token required}
summary=${WATCHDOG_SUMMARY:?WATCHDOG_SUMMARY is required}

still_owner() { [ "$(cat "$owner_file" 2>/dev/null)" = "$token" ]; }
case "$deadline" in none) ;; ''|*[!0-9.]*) echo "watchdog: deadline must be an epoch or 'none', not '$deadline'"; exit 2;; esac
status_every=${WATCHDOG_STATUS_EVERY:-300}
last_status=0
status() {
  local now age
  now=$(date +%s)
  [ $((now - last_status)) -ge "$status_every" ] || return 0
  last_status=$now
  if [ -n "${WATCHDOG_PROGRESS:-}" ] && [ -f "$WATCHDOG_PROGRESS" ]; then
    age=$(python3 -c 'import json,sys,time; p=json.load(open(sys.argv[1])); print("%.0f s ago (stage %s, units %s)" % (time.time()-p["last_update_unix"], p.get("stage"), p.get("units_done")))' "$WATCHDOG_PROGRESS" 2>/dev/null || echo "unreadable")
    echo "$(date -u +%FT%TZ) watchdog: writer $pid alive; last progress update $age"
  else
    echo "$(date -u +%FT%TZ) watchdog: writer $pid alive; no progress record yet"
  fi
}

# Review round 6 (1): the writer can also die before reaching its own teardown (a startup failure in the CLI,
# backend construction or the chain's start). It then never writes the done file, so act at once instead of
# waiting for the deadline. The done file is checked again after the writer is gone (it is written before exit).
reason=deadline
while :; do
  if ! still_owner; then echo "watchdog superseded by a newer attempt; standing down"; exit 0; fi
  if [ -e "$done_file" ]; then echo "watchdog cancelled: chain completed ($(cat "$done_file"))"; exit 0; fi
  if ! kill -0 "$pid" 2>/dev/null; then
    sleep 2
    if [ -e "$done_file" ]; then echo "watchdog cancelled: chain completed ($(cat "$done_file"))"; exit 0; fi
    # Optional (C-attribution): the runner records each attempt it tore down itself, with its PID. That writer
    # already copied, exported and requested (or deliberately withheld) the stop: not a crash, so stand down.
    if [ -n "${WATCHDOG_HANDLED:-}" ] && [ -f "$WATCHDOG_HANDLED" ] && python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if str(r.get("writer_pid")) == sys.argv[2] else 1)' "$WATCHDOG_HANDLED" "$pid" 2>/dev/null; then
      echo "watchdog standing down: attempt handled by the runner ($(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("state %s, stop %s" % (r.get("state"), r.get("stop")))' "$WATCHDOG_HANDLED"))"
      exit 0
    fi
    reason=writer-exited-without-completing; break
  fi
  if [ "$deadline" != none ] && awk -v now="$(date +%s.%N)" -v d="$deadline" 'BEGIN { exit !(now >= d) }'; then break; fi
  status
  sleep 1
done

if ! still_owner; then echo "watchdog superseded by a newer attempt; standing down"; exit 0; fi
echo "watchdog fired at $(date -u +%FT%TZ) ($reason); signalling writer $pid"
kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 "$wait_s"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
writer_alive=false
if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; writer_alive=true; fi

if ! still_owner; then echo "watchdog superseded while waiting for the writer; not copying or stopping"; exit 0; fi
timeout "$copy_t" bash -c "${WATCHDOG_COPY:?WATCHDOG_COPY is required}"; copy_rc=$?

write_summary() {
  printf '{"fired": true, "reason": "%s", "fired_unix": %s, "writer_needed_kill": %s, "copy_returncode": %s, "stop": "%s", "stop_returncodes": [%s], "externally_verified": false}\n' \
    "$reason" "$(date +%s)" "$writer_alive" "$copy_rc" "$1" "$2" > "$summary"
}
write_summary "pending" ""

# Optional (C-attribution): a failed rescue copy means the volume may lack local work; stopping the Pod would
# delete it, so leave the Pod running and say so.
if [ "${WATCHDOG_STOP_REQUIRES_COPY:-0}" = 1 ] && [ "$copy_rc" -ne 0 ]; then
  write_summary "withheld" ""
  echo "STOP WITHHELD: the rescue copy failed (rc $copy_rc); the Pod is left running so its local work can be backed up. Stop it by hand: ${WATCHDOG_STOP_1:?}"
  exit 0
fi

stop="not-requested" codes=""
for cmd in "${WATCHDOG_STOP_1:?}" "${WATCHDOG_STOP_2:?}"; do
  timeout "$stop_t" bash -c "$cmd"; rc=$?
  codes="${codes:+$codes, }$rc"
  if [ "$rc" -eq 0 ]; then stop="requested"; break; fi
done
write_summary "$stop" "$codes"
echo "watchdog finished: copy rc $copy_rc, stop $stop"
exit 0

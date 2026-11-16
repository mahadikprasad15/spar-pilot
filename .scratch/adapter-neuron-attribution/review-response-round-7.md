# Response to review round 7

Baseline: `review-round-7.md` (Codex, on `516dbd8`). Both findings confirmed by reading
`scripts/attribution_overnight.sh`, and both fixed in `710d2af`, test-first: the new
`tests/test_attribution_launcher.py` runs the **real launcher and watchdog** with two worker processes and a mock
`runpodctl` on PATH. Three of its four tests failed on `516dbd8`; all four pass now. 222 attribution tests pass.

## P1: adopt

The launcher now takes watchdog ownership **last**, after every check passes:

1. `flock -n` on `<local>/launch.lock` (container disk). The lock's file descriptor is inherited by the chain and
   its watchdog, so it is held while either runs: a second launch on the same Pod is refused before it touches
   anything. This closes the concurrent-claim race on one Pod; `flock` on the FUSE volume is not relied on.
2. New `attribution_chain precheck`: validates the sealed record (seal, types, commit) and applies `resume_decision`
   to the volume's writer marker, so a live, foreign-run or ambiguous writer (including one on another Pod
   sharing sl-0) is refused before ownership transfer.
3. Deadline and stop commands are computed; only then the owner token is written and the chain and watchdog start.

A launch refused at 1–3 never writes the owner token and never starts a watchdog, so it can neither disarm nor stop
the live run. A writer that took ownership and then failed (the round-6 case) still makes its watchdog copy and
request the stop within seconds.

Tests: `test_r7_1_a_second_launch_never_disarms_or_stops_a_live_writer` (live writer process holding the marker;
owner file unchanged, no watchdog, no stop request, writer still alive);
`test_r7_1_a_second_launch_on_the_same_pod_is_refused_by_the_local_lock`;
`test_r7_1_an_owned_startup_failure_still_stops_the_pod_promptly` (the real CLI refuses a volume without its identity
file after ownership; the watchdog records `writer-exited-without-completing` and the mock `runpodctl stop pod`).

Residual (unchanged, ticket 11): cross-Pod claims rely on the marker's heartbeat rule, an advisory single-host
check; round-5 items 1–3 stay open before unattended use.

## P2: adopt (the high-visibility option, no automatic stop)

Any failure before the chain and watchdog exist (directories, lock, precheck, deadline, record fields) prints a
boxed `NO RUN STARTED: <reason>` with `runpodctl stop pod <id>` and exits 1, without stopping anything. Reason for
not stopping automatically: at that point no run exists to protect data for, a refused second launch must never
stop a running one, and on a supervised launch the person is at the terminal; a setup typo should not cost the
container disk. Test: `test_r7_2_a_bad_record_starts_nothing_and_says_the_pod_is_still_running` (missing record:
exit 1, message, no owner file, no watchdog, no stop).

## Not changed

Qualification cap checks and the real-backend CPU integration test remain open, as in round 6.

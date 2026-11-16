# Attribution pipeline review, round 7

_Codex's review of `516dbd8` (baseline `753de57`), pasted into the session by the user on 2026-11-16 (Codex's copy
was not pushed). Response: `review-response-round-7.md`._

Scope: consequential next-run and unattended-run defects. Findings are local CPU reproductions, not observations from a real Pod.

## P1: a refused second launch can immediately request a stop under the live first writer

`scripts/attribution_overnight.sh:15-19,30-38` changes the shared watchdog owner before the new worker establishes that no other writer is active. `pilot_eval/attribution_chain.py:283-301,485-490,550-552` refuses the new worker when a live writer holds the marker; its exception occurs before normal teardown. The new writer-exit branch in `scripts/attribution_watchdog.sh:29-35,40-60` then treats that refusal as an owned writer failure and requests Pod stop even though the old writer is still live. The ownership takeover was already a round-5 issue; commit `43f2a80` makes this failure request a stop promptly rather than merely waiting for the deadline. Relevant if a user reconnects and accidentally launches again while a tmux worker continues. Reproduced locally with the real `resume_decision` and watchdog script and mock stop commands.

Fix the ownership/start protocol so a rejected or ambiguous competing attempt never disarms an active writer's watchdog or starts a watchdog that can stop its Pod. Distinguish a writer that acquired ownership and then failed from a process refused before ownership; validate record and marker before transfer, and address concurrent-claim races. Add a regression at the launcher/watchdog boundary with two worker processes and mock stop commands, plus a genuine owned-startup-failure case.

## P2: failures before the writer and watchdog start still leave the Pod running

`scripts/attribution_overnight.sh:9,15-21,30-38` evaluates `deadline` under `set -e` before creating either the worker or the watchdog. A missing, stale, unsealed or wrong-commit launch record, or a failed deadline calculation, exits there; no watchdog requests a stop and no run summary is written. A local run with a missing launch-record path returned 1, created the owner file, and created neither `watchdog.log` nor `watchdog_summary.json`. Not the cause of either previous Pod failure. Handle and surface pre-writer launcher errors with a bounded fallback or an explicit high-visibility manual-stop instruction. Only stop automatically when the Pod ID is validated and a separate live writer has been ruled out; never stop an existing run to handle a refused second launch. Test this path and ensure it does not steal or disarm an existing owner's watchdog.

## Other dispositions

- No further consequential defect in the round-6 fixes (tokenizer binding, fitting budgets, pilot filter typing, rehearsal prompt, safetensors parameter count). Local sequences-v2 and inventory-v2 tokenizer hashes agree with the local tokenizer files; Pod copies not reverified.
- Qualification cap checks and the real-backend CPU integration smoke remain open, as do round-5 resume/backup and watchdog ownership issues before unattended use.
- Focused `-k r6_` tests: 11 passed, 1 failed only because macOS lacks GNU `timeout` (passed with a shim). No A100 end-to-end launch tested.
- Recommendation: fix P1, handle P2 with the owned-run guard, then the real-backend CPU integration check and one supervised A100 rehearsal. No automatic restart or duplicate launch; verify the actual Pod status after any stop request.

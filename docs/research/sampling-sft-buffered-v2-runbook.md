# Buffered v2 development handoff

These commands are preparation guidance for review, not a release or a reason
to interrupt the running v1 workload. The local branch has not been deployed
or qualified on real CUDA. Keep the current v1 checkout and source artifacts
unchanged. Run a fresh, separately named checkout only after review and release.

The saved Pod pilot points to
`/workspace/artifacts/plans/a40-q-ieee-nocache-detsched-v1-profile/sampling-sft.instrument.json`.
Its runtime selection and historical pilot report exist. The v2 pilot uses that
same selected instrument and validates its existing source/T3 evidence; it
does not regenerate score fixtures. Its own real-CUDA reference, controlled
SIGKILL after block two, resume, and exact comparison are still required. V2
generation also demands exact scientific equality between v1 and v2 reference
pilots. An exact difference blocks it; no tolerance is relaxed.

After an explicitly reviewed release, with the same qualified environment and
`--output-root /workspace/artifacts`, prepare a unique v2 pilot:

```sh
python -m pilot_eval sampling-sft-buffered-pilot-prepare \
  --instrument /workspace/artifacts/plans/a40-q-ieee-nocache-detsched-v1-profile/sampling-sft.instrument.json \
  --name buffered-replay-NEW --output-root /workspace/artifacts
```

For each of `pilot-reference`, `pilot-kill`, and `pilot-resume`, prepare a
distinct job with `sampling-sft-buffered-job-prepare --config PILOT --operation
STAGE --name UNIQUE --volume-id VOLUME --rate RATE --hours HOURS --output-root
/workspace/artifacts`; run it with `sampling-sft-buffered-job-run --config PILOT
--operation STAGE --job JOB --reviewer Prasad --output-root /workspace/artifacts`.
The resume command additionally needs `--kill-job KILL_JOB`. Jobs execute only
when the run command is issued; they are paid GPU work. The controlled kill is
expected to exit by SIGKILL. Inspect failed or uncertain launches and reconcile
before retrying. Then run `sampling-sft-buffered-pilot-compare --config PILOT
--output-root /workspace/artifacts` and inspect `gpu_qualified`.

Prepare a v2 generation plan with `sampling-sft-buffered-generation-prepare
--pilot OLD_PILOT --writer-pilot PILOT --name NEW --output-root
/workspace/artifacts`. Freeze a separate review bound to that plan and original
pilot report. After exact v2 replay and cross-writer evidence pass, prepare a
`generation-k5` buffered job with the same job commands. The existing k5 partial
blocks cannot be silently moved: their immutable seals bind the old plan.

After a newly generated v2 k5 is complete, `sampling-sft-mixed-generation-prepare`
can bind verified v1 k0/k2 and v2 k5 without rewriting them. The mixed review,
gate, and stored-ID data commands require the original coverage and scoring
rules. The mixed training path requires its own reviewed selection and T15/T16;
it does not authorize training merely by preparing data. This remains an
experimental implementation pending parent code review and real GPU replay.
Its current mixed handoff remains partial until comprehensive dynamic inventory,
independent external backup, lifecycle reconciliation, and actual cost
settlement are demonstrated. Do not stop a Pod on this assessment alone.

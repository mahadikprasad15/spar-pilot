# Response to review round 5

Disposition of `review-round-5.md` (Codex, reviewing `3cf8537..f5add6e`). Scope agreed by the user and Codex on 2026-11-14: make the small direct fixes, run the supervised Colab rehearsal, and fix only what it reveals before deciding on an overnight launch. Real GPU execution, memory and runtime are the most useful unknowns now.

- **5. Adopted (`1443169`).** The generation sweep starts with nothing selected. Batch size 1 becomes the selection only once its reference run succeeds. If batch 1 does not run, qualification stops before production with "batch size 1 did not run". The test is Codex's all-OOM mock.
- **6. Adopted (`1443169`).** A raw `contradicted` outcome is labelled the *unadjusted interval classification*, with a note that the Holm-adjusted p tests the support conjunction, not contradiction. Only `supported` is headed "after Holm".
- **1, 2, 3. Deferred by agreement.** These matter for a costly run left fully unattended that relies on automatic recovery. They are recorded as open in `issues/11-overnight-chain-runner.md` and must be resolved or explicitly accepted before any unattended overnight launch.
  - 1: the per-stage cap is not re-checked when a completed stage is skipped on retry.
  - 2: the expected-artifact set is not persisted or reconciled across attempts.
  - 3: watchdog ownership changes before the record is validated, and ownership is not re-checked just before the stop.
- **4. Handled operationally for 06a.** The tokenizer directory is passed explicitly, from the inventory's pinned tokenizer revision (notebook cell 6). The code-level binding to the sequences-v2 manifest's tokenizer hashes is still open on ticket 11.

The 06a Colab rehearsal does not use the chain runner, launcher or watchdog (`attribution_rehearsal`). Points 1–3 therefore cannot affect it. The first chain run, the Pod rehearsal, is to be treated as a test (see `issues/06a-colab-rehearsal.md`).

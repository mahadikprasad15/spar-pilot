# Three-arm sampling and training: staged pipeline

This release implements the user's explicit k0/k2/k5 selection (ADR 0023).
It runs jobs serially and resumes from verified outputs, but cannot make human
labels, silently acknowledge omissions, invent time estimates or move a Pod.
Run from the pinned release with the persistent artifact root. The earlier pilot
and generation plan remain bound to their existing source hashes.

1. **Prepare generation jobs** with `sampling-sft-pipeline-prepare-generation`,
   supplying the frozen generation plan, actual Pod volume/rate, reviewer and
   explicit per-arm estimated hours and setup. Run the resulting job array with
   `sampling-sft-pipeline-generation`. Each child timeout is 1.5× its estimate;
   a timeout can leave resumable sealed blocks and is not a successful run.
2. **Review:** run `sampling-sft-pipeline-review`. If it reports pending human
   labels, open its printed review page and complete actual human review; resume
   the command. It will not synthesize human decisions. If any arm has 1–5
   uncovered items or another acknowledgment flag, pause for the required
   review. More than five uncovered in any arm blocks training.
3. **Independent gate:** prepare with `sampling-sft-pipeline-prepare-gate`
   using the approved word-policy file. Create its bounded HF scorer job with
   `sampling-sft-pipeline-prepare-gate-job`, providing actual quote, volume and
   explicit time estimate. Run with `sampling-sft-pipeline-score-gate`; then
   `sampling-sft-pipeline-report-gate`. Supply a measured per-arm L4 training
   forecast to `sampling-sft-pipeline-select-gate`. It records the user's
   three-arm decision and waived budget only if no coverage flags need human
   acknowledgment; otherwise it returns the precise review boundary.
4. **Training data:** `sampling-sft-pipeline-data` exports the verified selected
   data and runs full CPU T15/T16 with the pinned trainer interpreter. It reuses
   existing passing evidence and never substitutes kernel-only readiness.
5. **Training:** on a verified compatible L4, prepare the three independent
   training plans/jobs using `sampling-sft-pipeline-prepare-training` with the
   shared selection and explicit per-arm rate/time estimates. Run the resulting
   array with `sampling-sft-pipeline-training`; the existing training backend
   checks L4/FP32, selection, readiness and required checkpoints. It verifies
   each completed 64-step result before claiming all three are complete.

The pipeline writes full SHA256 inventories for the generation tree, the dynamic
HF scoring tree and the three training trees. They are local evidence, not a
backup. Copy each complete tree, plans, source artifacts and budget ledger to an
independent host and verify all hashes before requesting any Pod stop. Static
lifecycle manifests list only declared outputs and are not a complete substitute.
No deletion, Pod stop or termination is performed by these commands. The current
machine-critical warning should be considered before starting a long run.

## Reconciled generation continuation and independent verification

A failed sealed job is not retried automatically. Inspect its execution/child
log, verify the prior worker is dead, recover any saved writer through the
existing public command, and settle its actual observed cost in the shared ledger.
Then prepare a distinct job with a fresh name and bounded estimate:

```sh
python -m pilot_eval sampling-sft-pipeline-replace-generation --config plans/GENERATION/sampling-sft.generation.json --arm ARM --prior-job runs/sampling-sft/SOURCE/jobs/OLD/job.plan.json --name NEW_JOB --estimated-hours ESTIMATE --output-root artifacts
python -m pilot_eval sampling-sft-pipeline-replace-entry --output-root artifacts --entries OLD_JOB_ARRAY.json --arm ARM --new-job runs/sampling-sft/SOURCE/jobs/NEW_JOB/job.plan.json --evidence runs/sampling-sft/SOURCE/generation/GENERATION/pipeline/replacements/NEW_JOB/prior-evidence.json --target NEW_JOB_ARRAY.json
python -m pilot_eval sampling-sft-pipeline-generation --config plans/GENERATION/sampling-sft.generation.json --jobs NEW_JOB_ARRAY.json --output-root artifacts
```

Only a sealed failed lifecycle result with settled cost and verified partial arm
state is accepted. A recorded but unfinished launch remains uncertain and must
be investigated. Prior arrays, jobs, logs and blocks are retained. Every completed
arm gets an inventory before the next arm starts, so an interruption later does
not prevent independent verification of the finished arm.

Copy an inventory and every listed relative file to an independent host, then
from that host's checkout run:

```sh
python -m pilot_eval sampling-sft-pipeline-verify-copy --inventory COPIED_INVENTORY.json --copy-root COPIED_ARTIFACT_ROOT --receipt NEW_RECEIPT.json --source-host SOURCE_HOST_FROM_INVENTORY
```

This checks every size and SHA256 and records the external host. The inventory
covers scientific trees, not the entire plans/source/budget ledger; copy and
verify those separately. The receipt is client-side evidence, not automatic
permission to stop or terminate the Pod.

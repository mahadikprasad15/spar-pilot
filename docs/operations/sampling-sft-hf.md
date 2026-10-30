# Sampling SFT: RunPod and private HF persistence

Dataset: `Prasadmahadik/spar-sampling-sft-artifacts` (private).
The live 2026-10-30 preflight found public visibility and applied the user's
requested private setting before any upload. Transfers reject public repos.

The network volume remains the active artifact root and authoritative budget
ledger. HF is an archive/transport, not a second running ledger or a lock server.
Use one pinned code release on the worker and external controller. Credentials
come from `HF_TOKEN`/the Hub credential store; do not put tokens in JSON or logs.
A repo-scoped write credential is needed for worker uploads and controller acks.
Do not upgrade the model environment to install a newer HF CLI; qualify the
transport with the existing pinned SDK, or use a separate transfer environment.

## Worker

The existing job plan/paid approval and verified volume mount are prerequisites.
The plan's declared output inventory must include all required raw attempts,
scientific block records, source configs, coverage and training/recovery files;
this transport uploads exactly the sealed job inventory. It does not recursively
upload every file on the machine or discover missing scientific dependencies.
Do not include credentials or model caches in output inventories.

```sh
python scripts/run_sampling_sft_job.py \
  --job "$JOB_PLAN" --output-root "$ARTIFACT_ROOT" \
  --approval "$PAID_APPROVAL" --network-root "$NETWORK_ARTIFACT_ROOT" \
  --download-receipt "$ARTIFACT_ROOT/notebook/downloaded-JOB.json" \
  --max-wait-seconds 600 \
  --hf-repo Prasadmahadik/spar-sampling-sft-artifacts
```

Choose a job-specific receipt name. Wait must fit the existing reserved job
bound; the actual launch age is checked while waiting. Work, export, upload,
setup and waiting all count toward the stage budget. Transfer errors preserve
sealed work and prevent shutdown; they require prompt operator attention.
Provider/network outages cannot be treated as confirmed stops or free time.

The runner executes work, verifies the network copy, uploads the sealed archive,
prints its HF receipt, and waits for a separately issued downloaded-copy receipt.
It receives that acknowledgement from HF and passes it through the existing
job/manifest/host validation before requesting its own Pod stop. An upload or ack
failure cannot authorize stop/delete. External provider confirmation and actual
billing reconciliation are still separate existing commands.

Upload only, for already sealed work or retrying a failed transfer:

```sh
python -m pilot_eval.sampling_sft_hf upload \
  --manifest "$EXPORT_MANIFEST" --output-root "$ARTIFACT_ROOT"
```

This writes `hf-upload.json` alongside the job manifest. A repeated upload uses
that exact revision and verifies readback rather than rerunning the workload.
An interrupted upload without a receipt may need retransmission of the bundle;
its content-addressed scientific inventory remains unchanged.

## External controller (laptop or separate CPU host)

Copy the small `hf-upload.json` from the worker/logged receipt to your controller.
This file contains no credentials. The download must run outside the worker for
a real job; same-machine test probes are explicitly marked simulated.

```sh
python -m pilot_eval.sampling_sft_hf restore \
  --upload-receipt "$UPLOAD_RECEIPT" \
  --copy-root "$LOCAL_ARTIFACT_ROOT" \
  --receipt "$LOCAL_ARTIFACT_ROOT/notebook/downloaded-JOB.json" \
  --acknowledge
```

Restore downloads the archive at its exact commit, checks archive and per-file
hashes, rejects links/path escapes/conflicting local files, and restores original
artifact-relative paths. It saves the normal downloaded-copy receipt plus HF
provenance. `--acknowledge` uploads only that receipt to HF; the worker can then
complete its existing stop handshake. Keep the controller ready before starting
GPU work to avoid billing idle time while waiting for acknowledgement.

## Archive layout

```
exports/<manifest-sha256>/bundle.tar.gz
exports/<manifest-sha256>/index.json
acknowledgements/<manifest-sha256>.json
```

Every upload receipt records the exact archive commit, archive hash, manifest
hash, job identity and canonical manifest path. Later repository changes do not
change an existing archived revision. Stop-intent/provider confirmation records
created after sealing are preserved on the network volume; archive them later
as a separately sealed control product if required. Do not mutate the scientific
bundle after its acknowledgement.

## Evidence and remaining preflight

CPU transport/lifecycle tests use fake HF/cloud adapters. A real tiny synthetic
upload/download/ack probe was also completed on 2026-10-30, saved under
`artifacts/runs/sampling-sft/persistence-smoke-2026-10-30-v1/`. It proves private
HF access and exact-revision hash restoration on this laptop. It does not prove
RunPod mounting, supported stop API behavior, independent-machine receipt flow,
worker numerical settings, score agreement, exact sampling replay or throughput.
Those remain real preflight checks before full paid generation. No GPU was
provisioned by this implementation step.

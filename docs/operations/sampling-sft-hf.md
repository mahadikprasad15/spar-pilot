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
  --max-wait-seconds "$ACK_WAIT_SECONDS" \
  --hf-repo Prasadmahadik/spar-sampling-sft-artifacts
```

Choose a job-specific receipt name. The acknowledgement clock starts after
network export and HF upload. It is capped by the remaining overall job
allocation, measured from the recorded launch time; it never resets that budget.
Choose ACK_WAIT_SECONDS using measured transfer/hash rates and bundle size,
and include that allowance in the reservation before launch. Work, export, upload,
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

## Transfer resilience and sizing

Temporary acknowledgement fetch failures (network timeouts/resets, HTTP429 and
HTTP5xx) are logged as `hf-ack-retry` and polled again with2/4/8/16/30-second
bounded backoff. Missing receipts are also polled. Authentication/permission,
malformed acknowledgement and hash failures remain fatal. Each real transfer
call is limited by the remaining allocation; expiry is distinct from an ordinary
network timeout and cannot be retried. Nothing automatically enlarges a cap.

Estimate the acknowledgement allowance using measured laptop throughput:

```sh
python -m pilot_eval.sampling_sft_hf forecast \
  --upload-receipt "$PROFILE_UPLOAD_RECEIPT" \
  --download-bytes-per-second "$MEASURED_DOWNLOAD_RATE" \
  --verify-bytes-per-second "$MEASURED_HASH_RATE" \
  --overhead-seconds 60
```

The forecast includes compressed download size, unpacked hash/read size, a1.5
allowance factor and explicit overhead. It recommends a wait, not spending
approval. Before the full bundle exists, scale measured profile sizes to the
planned run conservatively and reserve the allowance. Recheck actual bundle
size after upload; if existing budget headroom is insufficient, report and
reconcile rather than silently extending the paid wait.

Before bundling, the worker checks space for conservative tar/gzip overhead and
both archive construction and readback. The real Hub cache filesystem is also
checked. Restore checks destination/cache capacity before download and actual
unpacked member sizes before extraction. These checks reduce predictable disk
failures; concurrent processes can still consume space afterward.

Use a fine-grained HF token restricted to this private dataset, supplied as a
RunPod secret/environment variable. Anyone with Pod administrative access may
inspect its environment; do not use a broad personal token. Revoke the dedicated
token when the experiment is finished. Self-hashed receipts provide integrity
and job binding, not cryptographic identity; trust the controller and repo writers.

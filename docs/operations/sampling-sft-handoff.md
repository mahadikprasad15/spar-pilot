# Sampling SFT: final handoff and fresh-session operation

Ticket10 assembles a verified inventory and a research-status snapshot. It does
not collect new responses, train adapters, evaluate them or operate a Pod.
The handoff is an index of the canonical artifact tree, not another copy of the
model files. JSON contains the structured evidence; `report.md` is the readable
view. Partial execution is a supported result.

## Release and restoration

The first setup cell of `notebooks/sampling-sft.ipynb` contains `SOURCE_RELEASE`,
an exact Git code commit. Use that release in a fresh checkout before restoring
saved plans. The notebook is an optional CPU client of the same CLI commands as
the RunPod runbook. Its documentation/pin commit can follow the code commit.

```sh
python -m pilot_eval sampling-sft-handoff-release --commit SOURCE_RELEASE
```

This checks the local sampler/training execution sources against that commit.
It performs no fetch, checkout, package install or network request. Command
interfaces are covered separately by the public-workflow tests. Keep the
artifact root separate from the checkout so reconnects restore saved products.
For Colab CPU, mount Drive and select the existing artifact root first.

For an existing snapshot, only the release check and CPU verification below are
needed. Do not rerun generation, score collection or training to view it. Existing
stage verifiers can refresh disposable status projections; the handoff excludes
those projections from its scientific inventory.

## Workflow map

| Stage | Compute | Saved output | Resume/use |
|---|---|---|---|
| Source preparation/audit | CPU | Original config,512/150 IDs, gold order/masks, checkpoint identity | Restore the saved source plan |
| Engine scoring/profile and fixed32 pilot | Approved L40S job | Runtime selection, T3 and exact restart/replay evidence | Same frozen runtime; failures block |
| k0/k2/k5 generation | Approved L40S job | Raw IDs/text, attempts, steps, events,512 dispositions per arm | Sealed block boundaries |
| Intended-answer eligibility review | CPU + human | Separate labels, frozen coverage, explicit missing slots | Restore saved revision |
| Independent gate scoring | Approved model-work job | Aligned HF FP32 per-token scores | Sealed problem units |
| Gate statistics and selection | CPU + explicit review | Plain NLL/length/format/leakage, paired intervals, k0/k5 selection | No baseline/offset or automatic merging |
| Training-data export and full T15/T16 | CPU in pinned trainer environment | Exact stored IDs/order/init, short-batch evidence | Kernel-only checks are insufficient |
| Endpoint training | Approved L4 FP32 job |64-step logs, adapters0/8/16/32/64, optional full recovery | Verified checkpoint; otherwise new approved restart |
| Handoff | CPU | Hashed inventory, status JSON and Markdown | Verify without importing model libraries |

The sampler's setup/profile stages are distinct from the deferred **Stage0
geometry research**. Installing a pinned environment, renting hardware or
launching work still requires the existing approvals and budget reservations.
The preparation/status cells in the notebook never authorize a paid job.

See the stage runbooks for [lifecycle](sampling-sft-runpod.md),
[pilot](sampling-sft-pilot.md), [generation](sampling-sft-generation.md),
[review](sampling-sft-review.md), [gate](sampling-sft-gate.md),
[training data](sampling-sft-training-data.md) and [training](sampling-sft-training.md).

## Select evidence explicitly

Start with just `source_plan` for a source-only partial report. Add later plans
when their products exist. Omit stages that have not been prepared; do not invent
paths or edit completion flags.

```json
{
  "source_plan": "plans/sampling-sft-v1/sampling-sft.plan.json",
  "generation_plan": "plans/sampling-generation-v1/sampling-sft.generation.json",
  "review_plan": "plans/sampling-review-v1/sampling-sft.review.json",
  "gate_plan": "plans/sampling-gate-v1/sampling-sft.gate.json",
  "training_data_plan": "plans/sampling-training-data-v1/sampling-sft.training-data.json",
  "training_selection": "runs/sampling-sft/sampling-sft-v1/gate/sampling-gate-v1/training-selections/SELECTION_HASH.json",
  "training_plans": {
    "0": "plans/sampling-k0-attempt1/sampling-sft.training.json",
    "5": "plans/sampling-k5-attempt1/sampling-sft.training.json"
  },
  "jobs": [],
  "software_evidence": []
}
```

Paths are relative to the artifact root or absolute within it. All supplied
stages must belong to the same source/review/data chain. Only k0/k5 training is
accepted. The recorded post-gate selection determines which arms are required;
without that selection the default is both endpoint arms.

Every lifecycle plan in this experiment's `jobs/` tree is included automatically,
including failed and prepared attempts. A scientific event referencing a missing
job plan blocks capture. Use `jobs` entries when receipts are saved at nonstandard
paths: `{"plan":".../job.plan.json","network_receipt":"...json",
"download_receipt":"...json"}`. No client/provider is contacted.

`software_evidence` may list recorded `software-validated` JSON receipts with
`source_files` hashes. The report names their ticket/scope. Matching code and CPU
tests do not establish real GPU qualification or scientific completion.

## Build and inspect

```sh
python -m pilot_eval sampling-sft-handoff-prepare --selection handoff-selection.json --name sampling-handoff-v1 --release-commit SOURCE_RELEASE --output-root artifacts
python -m pilot_eval sampling-sft-handoff-build --config plans/sampling-handoff-v1/sampling-sft.handoff.json --output-root artifacts
python -m pilot_eval sampling-sft-handoff-verify --config plans/sampling-handoff-v1/sampling-sft.handoff.json --output-root artifacts
python -m pilot_eval sampling-sft-handoff-status --config plans/sampling-handoff-v1/sampling-sft.handoff.json --output-root artifacts
```

The command prints concise stage flags and missing evidence. The files are under
`artifacts/runs/sampling-sft/<source-plan>/handoffs/<name>/`:

- `results/report.json`: source, qualification, generation, human review, gate,
  readiness, training, budget and lifecycle evidence, separately labelled.
- `results/inventory.json`: paths, SHA256 and byte counts; explicitly absent
  optional observations are retained separately, not manufactured as files.
- `report.md`: readable status, selected outputs, limitations and deferred work.
- `complete.json`: the handoff product's integrity seal. Its existence means the
  **snapshot was built**, not that the experiment is complete.

`state=completed` requires all selected real experiment stages, full readiness,
verified training, export/shutdown evidence and reconciled GPU charges. CPU
fixtures remain partial even when their software workflow finished. A stop
request alone is not external confirmation that a Pod stopped. Copy receipts are
retained client hash confirmations, not authentication or a fresh check of live
cloud state; source payload hashes are rechecked locally.

Evaluation, internals, Stage0 research, the512x8 baseline/offset, k2 training,
k10, k5+5 polish, self-correct SFT and extra seeds remain deferred/separately
approved. Completed here never includes those follow-ups.

## Changes, copies and interruptions

A missing required selected checkpoint or altered parent is an error, not a
successful partial report. A genuinely unexecuted stage is pending. Active or
unrecovered upstream writer ownership blocks capture. Inspect/recover the owner
through that stage's public command before retrying; do not remove its marker.

Snapshots are immutable. When evidence or the budget ledger advances, create a
new named handoff. Rebuilding an unchanged handoff only verifies it. Keep source
files, raw history and approvals in their canonical locations.

For a portable verification bundle, copy every `inventory.files` entry preserving
its relative path, plus the handoff's `complete.json`, `results/inventory.json`,
`results/report.json` and `report.md`. The inventory includes its own saved plan
and input/manifest controls. Verify with the copied artifact root on CPU. A
minimal verification bundle is not a replacement for the full execution tree,
model cache and runtime needed to resume jobs.

Existing lifecycle commands verify both network and downloaded copies before
own-Pod stop; an external controller confirms shutdown and actual billing is
settled separately. Handoff commands do not stop, terminate or delete resources.
Volume deletion and termination of unexported work still need explicit approval.

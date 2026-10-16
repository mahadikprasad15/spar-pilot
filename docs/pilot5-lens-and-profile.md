# Pilot 5: community lens status and bounded endpoint costs

## What this release runs

The research question is whether SFT and GRPO leave task-related, readable activation changes on unrelated text. Production will compare paired activations on identical inputs and read their mean difference through qualified lenses. This release resolves an external lens risk and measures a small endpoint workload. It does not run that production experiment.

```text
Existing 200-document corpus + audited adapter sources
                 │
        ┌────────┴──────────────────┐
        ▼                           ▼
Pinned community lens          Four fixed diagnostic documents
CPU inspection                 2 legacy + 2 new, 128 tokens each
        │                           │
provenance status              base / SFT64 / GRPO64 / random64
        │                      each at batch 1 and batch 4
        │                           │
        │                      GPU profile → 8 durable units
        │                           │
        └──────────────┬────────────┘
                       ▼
             CPU verification and review
                       │
          gates + scope + budget + predictions
                       │
           production freeze remains pending
```

## Shared notebook sequence

| Cell | Runtime | Work and output |
| --- | --- | --- |
| 17 | CPU | Download one pinned 122 MiB lens plus card/manifest; verify checksum; inspect restricted tensor archive; save matrix inventory and an explicit provenance status. No Torch/model inference. |
| 18 | CPU | Verify used source seals and endpoint payloads, reverify the existing random construction, reuse the four diagnostic inputs and pin model metadata. Save `traces.profile.json`, workload and storage estimate. |
| 19 | GPU | Load one FP32 model and three existing endpoint adapters. Execute the frozen eight units; capture all 29 sites; measure forward/copy, hash, reduction/moment, serialization and memory costs. Save raw profile arrays and evidence after each complete unit. |
| 20 | CPU | Reverify parent hashes, runtime, all unit payloads and completion. Return `profile_complete: true`, `production_ready: false`. This is the GPU-release boundary for this diagnostic. |

Fresh CPU: **1–2 → 17–18**. Fresh GPU: **1–2 → 4 → 19–20**. Do not rerun 14–16 if their saved evidence is available. Do not use Run all. Switching the setup pin requires a fresh Python runtime; source data and completed diagnostic evidence remain in Drive.

## Exact approved workload

Four documents × four conditions × two batch arrangements = **32 sequence forwards**. There are **20 model calls**, because each condition takes four calls at batch 1 and one at batch 4. Every call captures embedding output plus all 28 decoder-block outputs together. It is not 29 separate model forwards per document.

The profile contains endpoints only. This does **not** approve endpoints instead of trajectories for production. It does not generate answers, train adapters, decode learned token lists or fit a new lens.

Raw FP32 profile arrays at width 1,536 occupy **696 MiB before compression** across eight units. These diagnostic arrays permit CPU comparison of batching effects; they are distinct from the proposed production retention banks. Compression and Drive save time are measured. Forward time includes hook transfer to CPU; the report deliberately does not claim a separately measured transfer time. GPU memory figures are cumulative peaks since model load.

Full frozen-base hashes are measured around every unit (16 reads on an uninterrupted run). Those integrity reads can take longer than the tiny forward workload. The notebook prints heartbeat messages while commands are active. No end-to-end ETA has been measured yet, and this four-document profile cannot guarantee 200-document production time.

## Community lens finding

Artifact: `qbz506/prabodha-lenses`, revision `bc6aec86a96e7e75ce286dfb8f3e59aaaca50cce`, file `qwen2.5-1.5b-instruct.pt`, SHA256 `6a23a7460139a0d3b73fdcc3a204bdee57475223bdfa72a284e6c7c60ed3a9cc`.

Actual inspection finds 27 finite FP16 matrices, each 1,536 × 1,536, with keys 0–26, and `n_prompts=16`. Block 14 is available. The manifest’s steering site 17 is not the only matrix in the artifact. Upstream row-vector transport is `v @ J.T`; model final norm and unembedding are separate operations.

**Status: insufficient provenance; not qualified.** The exact fitting base-model revision is not recorded. The card describes band-targeted/band-exit lenses and also a final-layer target; the serialized archive does not identify the fitted target/norm convention. Matching dimensions and a plausible filename do not settle those facts. No base-model qualification run is launched while that gate is unresolved.

Primary sources: [pinned card](https://huggingface.co/qbz506/prabodha-lenses/blob/bc6aec86a96e7e75ce286dfb8f3e59aaaca50cce/README.md), [pinned manifest](https://huggingface.co/qbz506/prabodha-lenses/blob/bc6aec86a96e7e75ce286dfb8f3e59aaaca50cce/lenses_manifest.json), [upstream implementation](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py).

The separately approved endpoint **cost diagnostic** can proceed without the lens. This does not authorize production with an unqualified lens, a logit-only primary claim, or fitting our own lens. Those choices remain with Prasad.

## Saved objects and recovery

- Inspection: `runs/pilot-5/<inspection-name>/lens-inspection/{config.json,results/results.json,complete.json}`; source card/manifest hashes and lens SHA are bound.
- Profile plan: `plans/<profile-name>/traces.profile.json`.
- Profile data: `runs/pilot-5/<profile-name>/profile/units/b{1,4}-{condition}/` with `arrays.npz`, `evidence.json`, `complete.json`.
- Whole-profile result: `profile/results/results.json`; runtime and manifest under `profile/meta/`.

All paths above are relative to the chosen artifact root. In Colab this is `/content/drive/MyDrive/SPAR/spar-pilot/artifacts`.

A unit contains four documents. Its arrays publish as accepted evidence only after the post-workload base hash matches and its payload seal is written. An interrupted unsealed unit repeats; sealed units verify and are reused. Resume requires the same recorded runtime; a different GPU or dependency stack needs a separately named profile. A wholly completed run is verified on CPU before any GPU check. It never reloads the model just to reuse a result.

The random control is reused, not regenerated: preparation requires its existing completion seal, verifies seed 42 and one realization, source GRPO hashes and saved construction, and rechecks the existing rank-one factors. Its per-module target and realized norms remain in the bound source evidence; this is one random reference, not a random-null distribution.

## What remains after cell 20

1. Review observed nonzero batching/conversion errors. No tolerance is selected automatically.
2. Complete the declared nonzero precision/ranking/bootstrap qualification; the current cost profile does not establish it.
3. Resolve community lens provenance/qualification or approve a separately budgeted alternative.
4. Review endpoint versus trajectory production scope, storage, budget and predictions.
5. Only then implement/run the production freeze/capture workflow in Ticket 07.

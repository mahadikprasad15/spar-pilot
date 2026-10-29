# Sampling SFT: eligibility review and coverage

Ticket 06 is CPU-only. It reads completed k0/k2/k5 collections, preserves their
automatic scores and responses, and publishes separate human eligibility and
coverage products. It starts no inference, cloud job or training.

## Prepare and inspect the workload

```sh
python -m pilot_eval sampling-sft-review-prepare --generation plans/sampling-generation-v1/sampling-sft.generation.json --name sampling-review-v1 --output-root artifacts
python -m pilot_eval sampling-sft-review-status --config plans/sampling-review-v1/sampling-sft.review.json --output-root artifacts
```

The packet contains every finished attempt whose flexible-v3 extraction status
is invalid, across the three arms and all retained retries. It includes the
question and exact response, but not gold answers or scorer correctness. The
reviewer identifies the response's intended answer; they are not asked to solve
the question or validate its reasoning. Valid but wrong extracted answers remain
ineligible. Capped/exhausted attempts are never rescued through human labels.

Open the `reviewer/review.html` path printed by status. The workload counter
shows reviewed and pending entries. There is no mandatory review if this packet
is empty. This is a local review aid, not an access-control boundary; raw source
files remain available separately.

## Label and save

Use numeric, absent or ambiguous. Numeric means a clearly identifiable intended
answer, whether correct or incorrect. Transcribe one number without units and
paste an exact supporting excerpt from the response. Written-out numbers may be
transcribed numerically when the excerpt supports them. Do not insert a number
from gold or repair the response. Absent/ambiguous labels carry no numeric value.

The page caches edits locally and downloads all entries, including pending ones.
If downloads are unavailable, use its copyable JSON backup. The CLI rejects
malformed numeric entries, so preserve the backup and finish those entries;
it does not discard them silently. Saving a partial review is supported.

```sh
python -m pilot_eval sampling-sft-review-import --config plans/sampling-review-v1/sampling-sft.review.json --labels /absolute/path/sampling-sft-labels.json --origin human --confirm-human --output-root artifacts
python -m pilot_eval sampling-sft-review-export --config plans/sampling-review-v1/sampling-sft.review.json --destination /absolute/path/saved-revision-1.json --output-root artifacts
```

Import the exported saved revision into the page before continuing. Each draft
is bound to the packet and a `base_revision`. An older download cannot overwrite
newer saved labels. Export to a new filename if an existing file contains edits;
the CLI refuses to overwrite a different draft. Read status after reconnecting;
no earlier notebook variables or browser cache are required.

`--confirm-human` is the operator's explicit attestation that these are human
decisions. It is not automated authentication of who typed them. Keep agent
suggestions separate: use export/import with `--origin agent`, without
`--confirm-human`. Suggestions have their own revision history and cannot
satisfy required review or change eligibility. Promoting a suggestion requires
actual human review and a separate confirmed human import.

The supported draft schema is:

```json
{
  "packet_sha256": "COPIED_FROM_EXPORT",
  "base_revision": 0,
  "reviewer": "REVIEWER",
  "labels": [
    {
      "response_id": "COPIED_FROM_EXPORT",
      "status": "numeric",
      "answer": "12",
      "evidence": "The final answer is 12.",
      "note": "Transcribed from the response."
    }
  ]
}
```

Keep every packet entry, including pending rows. Do not add text, token IDs,
gold answers or correctness decisions to the draft schema. The backend compares
confirmed numeric transcriptions with the saved gold separately, after answer
identification. Exact excerpt validation establishes provenance; determining
whether that excerpt really supports the transcription is the human task.

## Freeze reviewed coverage

```sh
python -m pilot_eval sampling-sft-review-freeze --config plans/sampling-review-v1/sampling-sft.review.json --output-root artifacts
python -m pilot_eval sampling-sft-review-verify --config plans/sampling-review-v1/sampling-sft.review.json --output-root artifacts
```

Freeze refuses any pending required human label. It preserves the original
automatic verdicts, then computes eligibility separately. A human override is
possible only for a finished extraction failure whose confirmed intended answer
equals gold. It never edits the text or token sequence. If review makes an
earlier retained attempt eligible, selection uses that earlier attempt; later
retries remain saved historical evidence, not deleted compute.

Coverage is published per arm with512 dispositions and the original64 batches
of eight slots in gold training order. Missing slots retain source/problem IDs
and reasons. They are not filled by gold, duplicates or incorrect traces.

| Uncovered problems per arm | Coverage outcome |
|---|---|
| 0 | Coverage passes only |
| 1–5 | Mark omissions; T16 and later training approval still required |
| More than5 | Coverage blocks training |

T16 belongs to the training-export ticket: verify batches3–7 under the original
supervised-token loss denominator. Clustered missing slots are retained in the
original batches; they are not redistributed. This ticket records the
requirement and does not claim to have run a trainer test.

The result includes common covered IDs for three-arm gate comparisons, covered
union, uncovered union/intersection, and each pair's uncovered-set symmetric
difference. A difference above5 is flagged before training approval. Each arm
keeps its own covered training slots. These flags do not silently change cohorts.

Frozen coverage is immutable. Further corrections require a separately named
review product. Changed generation sources, stale drafts, unknown labels or
altered reviewer packet/history fail verification. `training_allowed` remains
false even when coverage passes: the data gate, budget, T15/T16 and explicit
training approval remain later requirements.

## Saved products

```text
artifacts/runs/sampling-sft/<source-plan>/eligibility/<name>/
  meta/run_manifest.json
  meta/status.json
  reviewer/packet.json
  reviewer/review.html
  reviewer/page-seal.json
  drafts/human/000001.json
  drafts/human/000001.complete.json
  drafts/agent/...
  results/coverage.json
  results/k0/batches.json       # also k2/k5
  results/k0/dispositions.json  # also k2/k5
  complete.json
```

Revision snapshots carry origin, reviewer, human-confirmation status and the
previous revision hash. Automatic source products are read-only. The final
coverage seal binds the configuration and exact human revision. Retain these
raw JSON products alongside the review page; browser storage is not the record.

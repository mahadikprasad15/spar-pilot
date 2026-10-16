# Pilot 5: bounded JLens qualification from saved profile data

## The missing link

Ticket 04 supplied capture arithmetic and a real zero-adapter diagnostic. Ticket 05 supplied inspection of the community archive. Ticket 06 supplied the endpoint cost profile. Those deliveries did **not** execute the actual community matrix with the real output head on saved nonzero adapter differences. `qualified: false` was correct, but the implementation handoff did not distinguish this missing runner from a pending scientific review. The ticket remains partial until measured evidence is reviewed.

The new workflow fills that bounded numerical gap. It does not claim to reproduce the lens-fitting experiment, measure its approximation fidelity, or certify every layer and position. It does not replace the capture qualification receipt.

## Inputs → work → outputs

```text
Four saved profile documents, at batches 1 and 4
  base + SFT64 + GRPO64 + random64
          │
          ├─ block14, content positions 0–4
          │       adapted − base, then within-document mean
          │
Pinned J14 + pinned final RMS norm + pinned output head
          │
          ├─ transport v @ J14.T, RMS norm, vocabulary projection
          ├─ independent FP64 oracle on fixed probes and 32 head rows
          └─ nonzero retention sensitivity, full vocabulary and both signs
                  │
          FP16 / FP32 conversion before pooling
          whole-document resampling, saved seed and draw indices
                  │
          scalar errors / ranking stability, NO learned token lists
                  │
          completed diagnostic → awaiting review
                  │
          explicit limits + provenance decisions + selected batch/precision
                  │
          scoped qualification receipt OR preserved failed decision
```

There are **13 resumable units**: one independent arithmetic unit and 12 sensitivity units (2 batch arrangements × 3 nonzero conditions × 2 retained precisions). There are **zero decoder forward passes**. The profile's existing batch-1/batch-4 differences are preserved; this added runner measures retention within each arrangement. It does not issue a cross-batch invariance gate.

The notebook requests 1,000 paired document resamples with local seed 42. This is a bounded four-document sensitivity calculation, **not** the final 200-document inferential bootstrap. Repeated multisets are computed once: four draws from four documents have at most 35 distinct count vectors. Positive and negative normalized top-20 stability is measured without publishing IDs or strings. On the real vocabulary the score projects across all available token rows, not just 32 rows. Only the independent arithmetic oracle is restricted to 32 rows.

## Notebook sequence

After the existing profile is complete and verified in 20, use a CPU runtime with setup 1–2 and its required environment setup:

| Section | Work | Saved result |
| --- | --- | --- |
| 20A | Verify profile, extract/reuse pinned norm/head and vocabulary | Readout assets config, arrays, hashes |
| 20B | Prepare, run/resume and verify bounded real-artifact diagnostics | Plan, 13 unit results, bootstrap indices, aggregate result and unapproved review draft |
| 20C | After evidence review, apply a separately saved reviewed JSON | Hash-bound scoped receipt or a saved failed decision |

**Stop after 20B and share its evidence.** No acceptance values are supplied automatically. Do not change the immutable `review.draft.json`: copy it to a separately named JSON before entering reviewer, notes, all provenance decisions, selected batch/precision and six limits. Section 20C requires that reviewed path. Production also needs its separate capture numerical qualification, scope, budget and predictions. Do not use Run all.

A fresh CPU cache may download the approximately 3 GB pinned model weight file to extract its final norm and tied head. The extracted FP32 head is approximately 0.9 GB. Existing downloaded weights/assets are reused. This is CPU file extraction, not transformer inference. CPU/Drive timing has not been measured; no end-to-end minute estimate is asserted.

## Provenance decision

The original inspection remains unchanged. The new diagnostic plan records a separate supplement linking the published producer at commit `f835828b235c8f1c027b4205a80220cabf26f145` and the canonical model's revision history. That code targets the final block, consistent with source keys 0–26. Historical model revision compatibility is an inference; an original fitting-run manifest binding this binary to an exact invocation is absent.

The producer defaults to BF16 fitting, the distributed matrices are FP16, and its fitting default excludes the first 16 positions. Applying the lens at positions 0–4 is therefore a transfer of the fitted operator. Numerical stability cannot establish semantic fidelity at those positions. Acceptance requires explicitly acknowledging these limits; fitting a replacement is never automatic.

Sources: [producer](https://github.com/SharathSPhD/prabodha/blob/f835828b235c8f1c027b4205a80220cabf26f145/scripts/experiments/fit_lenses.py), [norm/head convention](https://github.com/SharathSPhD/prabodha/blob/f835828b235c8f1c027b4205a80220cabf26f145/vendor/jacobian-lens/jlens/hf.py), [model revision history](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commits/main).

## Script interface and durable state

Public commands are `python -m pilot_eval traces-lens-qualification-{prepare,diagnose,verify,approve}`. Each has `--help`. Preparation requires `--profile`, `--inspection`, `--assets`, `--name`, `--bootstrap-draws`, `--seed`, `--output-root`. Diagnosis and verification take `--config`; approval additionally takes `--review`. Diagnosis supports `--max-new-units` for bounded runs.

Plans live at `plans/<name>/traces.lens-qualification.json`. Evidence lives at `runs/pilot-5/<name>/lens-qualification/`. Unit seals allow reuse after interruption. Parent hashes bind lens, profile and assets; changing implementation or inputs requires a new name. Completed diagnostics remain `qualified: false`; approval writes a separate immutable receipt under `reviews/<review-hash>/qualification.json`. Failed reviews are preserved and raise an error. The consumer honors the receipt's block/window restriction; other Jacobian requests are marked unavailable while logit-lens requests remain available.

Local tests use independent tiny CPU fixtures and never download a model. They establish implementation behavior, not real-artifact acceptance. The real saved-profile diagnostic must still be run and reviewed.

# SPAR evaluation and fine-tuning pilots

## Pilot 4: rank-1 GRPO arm

The [Pilot 4 notebook](notebooks/pilot-4-colab.ipynb) implements source matching,
untuned sampling, tiny-model validation, GPU preflight and reviewed protocol
freeze. The scientific trainer and comparison stages are still pending.
See the [workflow guide](docs/pilot4-grpo.md) and
[first-principles preflight companion](docs/pilot4-preflight-concepts.md).
This work is local until pushed; real GPU acceptance must run in Colab.

## Pilot 3: fixed-token activation measurements

[Open the Pilot 3 notebook in Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-3-colab.ipynb)

[Cell-by-cell HTML guide with diagrams](output/learning/pilot3-notebook-explained.html) explains inputs, objects, measurement formulas and recovery. Open the downloaded HTML in a browser.

Reuse completed Pilot 2 adapters at steps 0/8/16/32/64; measure block-output
changes and direct module contributions on fixed GSM8K gold sequences and
FineWeb passages. The guided stages verify sources, audit masks, validate the
instrument, profile/review batches, freeze execution, measure/resume and
reconstruct uncertainty/plots on CPU. It performs no training or generation.
See the [instrument guide](docs/pilot3-instrument.md) and
[spec/design](.scratch/pilot3-write/spec.md).

The notebook pins measurement code at `c547501`; the notebook itself can be
opened from the newer main branch. CPU workflow and recovery are locally tested.
Actual Drive access, package installation, source checkpoints and CUDA gates
must be verified in Colab. Section 11 identifies when the GPU can be disconnected.

## Pilot 2: exploratory rank-1 SFT

[Open the Pilot 2 notebook in Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-2-colab.ipynb)

The [Pilot 2 guide](docs/pilot2-sft.md) covers frozen training inputs, a real T4
preflight, a fresh matched FP32 GSM8K baseline, rank-1 SFT training and checkpoint
resume, and paired accuracy/length reports. The new exploratory protocol uses
512 training problems and 64 optimizer steps; historical collapse numbers are
references, not acceptance gates. See [.scratch/pilot2-sft/spec.md](.scratch/pilot2-sft/spec.md).

The CPU suite includes an optional real tiny-model TRL/PEFT integration test
when training dependencies are installed. It constructs the model locally,
requires no model download, and verifies training and resume. Actual Qwen/T4
execution remains unverified until the notebook's GPU preflight runs.

## Pilot 1

[Open the guided notebook in Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-1-colab.ipynb)

The notebook explains the SPAR context, freezes a pinned harness revision, persists artifacts to Google Drive, guides a five-item audit per cell, and runs/resumes the full evaluation after a recorded manual review. It also displays uncertainty ranges and supports export. Its audit-to-results flow was verified with fake boundaries locally; real Drive mounting, dependency installation, and GPU inference require Colab execution.

Untuned `Qwen/Qwen2.5-1.5B-Instruct`, plus one optional PEFT adapter:

- GSM8K: 150 frozen test items, greedy generation, strict and flexible extraction.
- MMLU: 20 frozen test items per subject, 1,140 total, generated-text and constrained-logit scorers at 0 and 5 shots.
- Durable inputs, configs, responses, uncertainty intervals, historical gaps, and resumable progress under `artifacts/`.

The earlier SPAR scores (GSM8K 0.640 / 288 tokens; MMLU 0.570) are contextual references. This is a new sampled protocol, because the original prompts, items and scorers were unavailable. See [.scratch/pilot-1-evaluation/spec.md](.scratch/pilot-1-evaluation/spec.md).

## CPU tests

Use Python 3.10 or later; Python 3.12 is the locally verified environment.

```bash
python -m pip install -e '.[test]'
pytest -q
python -m pilot_eval --help
```

Tests use fake model/dataset boundaries. They need no GPU, model download, torch, transformers, or datasets. The local suite also exercises the CLI with injected boundaries; real GPU inference is not yet verified.

## Colab or RunPod

Upload or clone this repository, change to its directory, and use one visible CUDA GPU. Colab notebook cells can run the same commands with a leading `!`. Keep `artifacts/` on persistent storage or copy it back before a session expires.

```bash
python -m pip install -e '.[gpu,test]'
pytest -q
python -m pilot_eval prepare --plan baseline-v1 --dtype bfloat16 --batch-size 4
```

Preparation downloads dataset records and tokenizer/config files, resolves exact Hub commits once, and freezes the five cells. It does not load model weights. Repeating the same plan reuses its saved revisions and inputs. Plan options are immutable.

Use `--dtype float16` with a different plan name on a GPU without bf16 support, such as a T4. Precision and batch size are recorded; use a new plan for changed settings. An out-of-memory failure keeps durable responses but does not silently reduce the batch size.

### Fixed real-item audit

Run the first five items in a separately named audit for **each** cell:

```bash
for config in artifacts/plans/baseline-v1/*.config.json; do
  python -m pilot_eval run --config "$config" --audit-items 5 || break
done
```

Inspect `inputs/items.json`, `results/responses.jsonl`, and `logs/errors.jsonl` in the audit run directories. Check chat formatting, same-subject dev examples, no test-answer leakage, extracted answers, token counts, stop reasons, and four candidate logits. Audit metrics are small-sample diagnostics. Approve the instrument by this manual inspection before running full cohorts.

### Full baseline and resume

```bash
for config in artifacts/plans/baseline-v1/*.config.json; do
  python -m pilot_eval run --config "$config" || break
done
```

Repeat the exact command after a recoverable interruption. Completed IDs are read from JSONL; only missing items generate again. Completed results are checked against saved responses and do not regenerate. Runtime versions/hardware must match the saved config on resume. A deliberate replicate uses a new plan name.

Constrained scoring checks that ` A/B/C/D` are distinct single tokens in context. A top-logit tie or nonfinite logit stops the run and preserves the invalid item in the error log. Corrupted or duplicate records also stop; no final aggregate is written for an incomplete run. Inspect the error and use a new run after resolving its cause; invalid-item errors are not silently discarded.

### Optional adapter

```bash
python -m pilot_eval prepare --plan adapter-v1 --adapter /path/to/adapter --dtype bfloat16
```

Local directories are pinned by a content SHA256; Hub adapters are pinned by a commit SHA. The loader validates `base_model_name_or_path` against the configured model and loads the adapter for inference. Local paths must exist on the GPU host. Adapter stacks, merged adapters and sampled-decoding aggregation are future work. Pilot 2 adds the separately versioned SFT workflow described above.

## Artifacts

For recovery after a batch-8 five-shot OOM, use the [same-notebook recovery cells](docs/colab-recovery.md). `fork-plan` copies the original frozen inputs with a smaller batch size; `report` combines explicitly selected completed runs while retaining their paths, hashes, batch sizes and protocol warnings. Completed earlier evaluations do not need to run again.

For an exact top-logit tie, use the [logit tie recovery cells](docs/colab-logit-ties.md) (notebook sections 16–19). `revise-logits` freezes scorer v2, reuses verified raw source logits, and counts ties as invalid and incorrect while continuing. Completed GSM8K and text runs remain selected in the combined report.

If notebook progress polling raises `JSONDecodeError`, first [check and monitor the existing evaluation process](docs/colab-progress-recovery.md). It may still be running. The corrected helper tolerates unreadable progress snapshots; no new plan is required.

For CPU-only GSM8K flexible-v2 rescoring, use [these two Colab cells](docs/colab-flexible-v2.md) (sections 20–21). They verify saved source responses and create a separate immutable report containing all old/new scores and changed extractions. No model inference is required. Strict/v1 and MMLU results remain unchanged; see [ADR 0003](docs/adr/0003-gsm8k-flexible-v2.md) for the grammar.

```text
artifacts/
  cohorts/<dataset>/<revision>/<cohort>/{manifest.json,source.json}
  plans/<plan>/{pins.json,manifest.json,*.config.json,*.items.json}
  runs/pilot-1/<model>/<dataset>/<cohort>/<variant>/<run-id>/
    config.json
    inputs/{items.json,cohort.json}
    meta/{run_manifest.json,status.json}
    checkpoints/progress.json
    logs/{run.log,errors.jsonl}
    results/{responses.jsonl,results.json}
```

Model repository slashes become `--` in directory names. Each run config contains exact revisions, adapter identity, prompt template and indices, decoding settings, seed, precision, attention implementation, environment versions and GPU identity. Each successful batch is flushed to disk. The response file's SHA256 is stored in the aggregate.

GSM8K reports both accuracies, Wilson 95% intervals, mean/median generated tokens, and invalid/cap rates. MMLU reports four separate cells, per-subject counts, and 95% intervals from 2,000 within-subject bootstrap draws. Five-shot constrained scoring is the primary MMLU metric. Historical gaps are descriptive and never declare a reproduction failure.

Progress is recorded in `meta/status.json` and `logs/run.log`; the CLI prints the final JSON summary. All baseline artifacts should be retained outside Git. Dependency ranges support installation; exact resolved versions are saved per run. Numerical GPU reproducibility and performance still require the real-item audit.

## Source references

The external boundaries follow the official [Transformers generation and padding guidance](https://huggingface.co/docs/transformers/llm_tutorial), [chat template guidance](https://huggingface.co/docs/transformers/chat_templating), [PEFT loader API](https://huggingface.co/docs/peft/package_reference/peft_model), and dataset records from [GSM8K](https://huggingface.co/datasets/openai/gsm8k) and [MMLU](https://huggingface.co/datasets/cais/mmlu).

## Shared workspace

All three guided notebooks live in `notebooks/`. Their code checkout is
`/content/spar-pilot`; their persistent Drive root is
`SPAR/spar-pilot/artifacts`. The pilots retain separate run identities
inside that shared root. See [workspace migration notes](docs/workspace-layout.md).

Flexible v3 is implemented in the scripts and available in **section 12 of the
Pilot 2 notebook**. For completed runs, run only that CPU section to write the
corrected report to Drive; no GPU evaluation repeats.

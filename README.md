# SPAR pilot 1 evaluation harness

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

Local directories are pinned by a content SHA256; Hub adapters are pinned by a commit SHA. The loader validates `base_model_name_or_path` against the configured model and loads the adapter for inference. Local paths must exist on the GPU host. Adapter stacks, merged adapters, sampled-decoding aggregation and training are future work. Before later training, check training IDs against the frozen test manifest.

## Artifacts

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

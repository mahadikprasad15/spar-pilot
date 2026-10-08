# Pilot 4 Ticket 7: fixed-token writes

This is a forward-only measurement stage after verified GRPO training. It does
not generate answers, retrain SFT, or implement Ticket 8's coefficients/KL.
Use sections 17–27 in `notebooks/pilot-4-colab.ipynb`.

## Products and source contracts

- **GRPO trajectory:** five checkpoints, 0/8/16/32/64, verified against the real
  GRPO training seal/history. `grpo-writes-prepare` binds them to the existing
  Pilot 3 preparation. Exact input IDs, masks, full gold solutions, FineWeb
  documents and all three views are reused without tokenizing or downloading.
- **Random intervention:** one isolated NumPy PCG64 generator, seed 42, draws
  independent Gaussian input/output directions in saved layer/projection order,
  normalizes in FP64, then saves FP32 factors. All 196 effective update norms
  match verified GRPO step 64 (alpha/r = 1); zero targets remain exactly zero.
  Unit directions and construction evidence are retained. It is measured as a
  separate product with verified zero and the control labelled by its step-64
  norm-matching target. That label is not an optimizer step of a random training
  arm. Weight matching does not imply activation matching or a null distribution.

Earlier artifacts remain read-only. Random factors live beneath the new
`runs/pilot-4/.../random-fixed-writes/<name>/control-adapter/` directory. GRPO
measurement evidence lives beneath `runs/pilot-4/.../fixed-writes/<name>/`.
Input paths point to the original sealed Pilot 3 records.

## Commands

All commands take `--output-root`; use the existing Drive artifact root in Colab.
Paths below are illustrative variables, not new default run names.

```bash
python -m pilot_eval grpo-writes-prepare --config "$TRAINING_CONFIG" \
  --measurement-config "$PILOT3_PREPARED" --name "$GRPO_WRITES_NAME" \
  --output-root "$ARTIFACT_ROOT"
python -m pilot_eval grpo-writes-control --config "$GRPO_WRITES_PREPARED" \
  --name "$RANDOM_WRITES_NAME" --output-root "$ARTIFACT_ROOT"
```

For **each** new preparation, run the existing public instrument workflow:

1. `activation-calibration-prepare --config ... --name ...`
2. `activation-calibrate --config <calibration>`
3. Inspect evidence, then `activation-calibration-freeze --config ... --review-notes ...`
4. `activation-calibration-validate --config ...`
5. `activation-profile --config <prepared> --calibration <validated-calibration>`
6. Review passing candidates, then `activation-freeze --config <prepared> --profile ... --batch-size ... --name ... --review-notes ...`
7. `activation-measure --config <execution>` (resume the same configuration)
8. `activation-verify --config <execution>` (CPU-only)
9. `activation-report --config <execution> --name ...` (CPU-only)

Numerical acceptance is fresh and bound to each product/runtime. Matching FP32
alone does not authorize using an old SFT envelope. All 196 rank-1 hooks,
zero initialization, disabled-adapter switching invariance, exact frozen-base
hashes, counted positions and finite summaries retain Pilot 3's gates.
A new batch size/runtime requires new profiled, frozen evidence; no OOM fallback
changes a running plan silently.

Production shares one untuned reference across variants within each batch.
Incomplete/unsealed payloads cannot establish progress. Verified shards resume
without duplicate example/variant combinations. Final boundary checks precede
publication. The CPU verification command checks all required shards and
aggregate sums. Disconnect the GPU only after **both** products verify complete.

## Reporting and SFT reuse

```bash
python -m pilot_eval grpo-writes-report \
  --grpo-report "$GRPO_REPORT_DIR" --control-report "$RANDOM_REPORT_DIR" \
  --sft-report "$EXISTING_PILOT3_REPORT_DIR" --name "$COMPARISON_NAME" \
  --output-root "$ARTIFACT_ROOT"
```

The report verifies source files/seals and reuses existing SFT mean vectors only
when fixed sequences, base identity, precision, numerical runtime and batch
membership agree. Git commit alone may differ; numerical settings may not.
Source reports retain their separate calibration/direction-resolution evidence.
Missing or incompatible evidence stops the comparison instead of manufacturing
an alignment result.

Outputs under `reports/<name>/results/` include:

- `results.json`: all per-arm block/module magnitudes, intervals, denominator
  diagnostics, coverage and 504 endpoint cosine records (3 pairs × 3 views ×
  2 weightings × 28 layers).
- `cosines.csv`: signed random–GRPO, random–SFT and SFT–GRPO cosines at the
  step-64 endpoint, with defined status, reason, norms and resolution scales.
- `mean-vectors.npz`: the reused endpoint mean-change vectors.
- `report.md`: coverage and interpretation limits; `config.json` plus a final
  seal bind source reports and implementation.

Zero vectors, below-resolution vectors and directions with no calibrated
resolution have null cosine values and explicit reasons. One random realization
cannot yield a significance threshold or an empirical percentile noise floor.
Per-arm depth/module plots, token and equal-example weighting, 2,000 paired
example-bootstrap intervals and sufficient summaries use the existing instrument.

## Validation boundary

Software checks run on CPU with no model downloads. Tiny real Transformers/PEFT
models validate that generated safetensors load through the actual instrument,
all 196 hooks execute and state restores after switching. Controlled workflow
checks cover preparation, independent calibration, profiling/freezing, interrupted
measurement/resume, CPU reports, compatible SFT reuse and corrupt evidence.
These are implementation checks; real Qwen/CUDA capacity and numerical acceptance
must pass in Colab before scientific measurements are interpreted.

## Profiling overhead

Pilot 4 profiles record source-verification and total-invocation wall time, plus
per-candidate loading, warmup, summary saving, bytes written and wall time through
model cleanup. These complement synchronized instrument throughput/memory.
Each field states its scope; final JSON/seal writes are excluded from the wall
record written into those files. The provisional fastest candidate still uses
the synchronized workload, so inspect overhead before projecting session cost.

# 03b: Qualify all adapters and choose the production measurement plan

**Status:** needs-triage — draft for adversarial review

**What to build:** Before production, run the approved small real-GPU qualification using the same model, workers and software as the eventual run; benchmark the neuron and optional head collectors, and freeze batch membership and measured head scope.

**Blocked by:** 03a, 07. The partial exporter must exist before the GPU qualification so its evidence can be saved even if qualification fails.

- [ ] Each selected arm's live active PEFT A/B tensors, keys, shapes, dtype, rank, scale and target mapping match sealed checkpoints; test disabled adapter vs an independently loaded base and active branch/logits. Handle genuine zero/unresolved effects explicitly.
- [ ] Before measuring, freeze numeric acceptance rules and representative inputs; compare repeated/singleton/alternate-batch FP32 summaries and selected logits, FP32 vs BF16 as a labeled diagnostic, and approximately ten FP64 reference examples. Save per-layer/per-neuron discrepancies; no unsupported FP64 run is called a pass.
- [ ] Record the defined D-resolution diagnostic, including an exploratory <5× measured-floor flag without treating it as a confidence bound; investigate a failed rule instead of widening it.
- [ ] Use only the frozen, disjoint calibration split; compute per-arm/layer 99th percentile neuron error from paired D in FP32 versus same-weights FP64 on approximately 10 examples, with zero floor labeled unresolved. Keep batch-composition comparisons as separate robustness diagnostics; do not assume matching batches makes their effects cancel.
- [ ] Benchmark 20–50 examples with 1A hooks and with 1A+1B, base and one adapter; use disjoint calibration inputs where feasible and save tokens/s, batch timing, peak VRAM and projected data size. Use a preapproved cost decision for 1B; bind frozen same-across-arm batch and padding manifests.
- [ ] Benchmark the same FP32/TF32-off attention backend and hooks as production. Read the reviewed, approved plan rather than the `proposed-not-approved` candidate; no production launch while numeric or operational fields are unresolved.
- [ ] Run process-local interpreter, pinned packages, model cache/offline load and relevant runtime checks in the real worker. A skipped/incomplete GPU check is not a qualification.

## Comments

### 2026-11-10 — run plan drafted, still unapproved (Claude)

- `.scratch/joint-neuron-adapter-audit/frozen_plan_v1.yaml` holds the user's 03b decisions (two messages, 2026-11-10) plus Claude's fills, tagged `[user]` / `[proposed]`. Status is **unapproved** by request until the remaining details are checked.
- Decisions recorded: a zero measured FP64 floor is **unresolved** (never FP32 spacing). Early-layer (p99 > 1e-4, layers 0–2, RMS-normalized) and padding (10× unpadded) are **review flags**; 04 needs a recorded sign-off. Batched-vs-singleton logits ≤ 1e-3 is a hard gate on logits only, documented as spec amendment `reuse-adr-0014-logit-gate`, because the spec said not to import that Stage 0 threshold. ADR 0015's dot-product bound is the second amendment.
- `pilot_eval/neuron_audit_plan.py`: `load(plan, stage)` refuses unless the plan is approved and the required fields are filled. It also refuses if a user-fixed rule has changed (zero floor, logit gate scope, exact disabled-base equality, legacy TF32 setters forbidden, eager, no compile/autocast). `input_problems` checks the sealed inventory/sequences and the accepted background review. CLI: `python -m pilot_eval.neuron_audit_plan --plan … --stage 03b --artifact_root …`.
- Current blockers (from the CLI): approval by/on, `runtime.code_commit`, `operations.qualification_max_gpu_hours`, `storage.provisioned_volume_bytes`, `preflight.expected_sys_executable/prefix`, background review pending.
- Code updated to the plan: the read/branch checks use the ADR 0015 bound (and record the legacy fraction). Mutations still fail at d = 8960 over a 64-token batch, and a correct FP32 dot product passes. Precision uses torch 2.9 per-backend `fp32_precision="ieee"` (CUDA matmul, cuDNN conv/rnn, mkldnn) plus the plan's `set_float32_matmul_precision("highest")`, read back and recorded; no legacy `allow_tf32` setters. Export archives build in local temp, not on the volume.
- Not yet built: the 03b runner itself (preflight, live qualification, benchmark, batch freeze, FP64 floor, review-flag plots) and the Phase 1B head hooks.

### 2026-11-10 — 03b runner built and CPU-tested; not yet run on a GPU (Claude)

- `pilot_eval/neuron_audit_qualify.py`: sealed, resumable stages `preflight → masks → arms → benchmark → batch_agreement → fp64_floor → batch_freeze`, writing `qualification.json` under `runs/joint-neuron-adapter-audit/qualification/<run_id>/`. Failures write `stages/<stage>/failed-*.json` and `meta/status.json: failed`; progress goes to `logs/progress.jsonl`; `--stop_after <stage>` pauses at a stage boundary.
  - masks: decoded-mask test on all 750 frozen rows (reply_start = header newline after `assistant`, contiguous content, termination = last content token + `<|im_end|>`, trailing newline unmasked).
  - arms: adapter bytes = inventory hash; live = saved bitwise with inventory keys and scale; zero-B unresolved; disabled vs fresh base exactly equal; active adapter changes logits and the direct write; ADR 0015 identities incl. 1B heads. An arm-specific failure marks the arm failed and the study partial; an untuned/reference failure stops.
  - benchmark: plan budgets and production config, 1A then 1A+1B (untuned + first qualified arm), peak VRAM ≤ 80%, 1B scope all → measured subset 10–17 → omit, projected bytes/hours, and the 60% capacity rule confirmed with a 5% probe file.
  - batch_agreement: calibrate/validate sets packed at the chosen budget; batched vs singleton logits ≤ 1e-3 is a hard stop (ADR 0014, logits only); early-layer and padding review flags; discrepancy-vs-layer curves (plus a PNG if matplotlib is installed). `signoff --decision accept|reject` records the human review that 04 needs.
  - fp64_floor: D = |mean (arm − base) mean|z||; floor = p99 |D32 − D64| per arm/layer; fallback GPU → batch 1, ≤ 512 tokens → CPU 3 examples → labelled "proxy floor"; zero floor = unresolved.
  - batch_freeze: production discovery+held-out rows packed (length desc, id; size × max_len ≤ budget), manifest hashed for 04.
- `pilot_eval/neuron_audit_preflight.py` runs in the worker: offline flags set before Hub import; interpreter/prefix/package pins; IEEE precision + deterministic + device read back; writer ops on volume and local temp; provisioned-minus-used capacity; real offline model+tokenizer load. The online HF check runs in a **separate online process**: private, upload, fresh pinned download, hash compare, token never printed.
- Collector additions: FP64 reference mode and Phase 1B per-head `o_proj` stats, with a head-read identity gate. The loader passes an explicit `torch_device` to PEFT (it otherwise infers one; on a Mac that picked MPS, which has no FP64).
- Plan additions (numeric fields; new `[proposed]` items): tolerance numbers, flag threshold/factor/floor, VRAM fraction, 1B ratio/disk/subset, benchmark arm = first qualified trained arm, packing rule, arm-failure semantics.
- Tests: 114 audit tests pass in `.venv` (15 runner + 9 preflight new): full CPU run, resume, stop/resume, mask gate, tampered adapter → partial, logit gate, FP64 fallback to CPU and to proxy floor, signoff, pure rules.
- **Not verified:** anything on a real GPU, the TransformersBackend loader against the real model, the live HF subprocess, and real timings/VRAM. Example worker config: `qualification.config.example.json`.
- Pod launch (after approval): restore the preparation snapshot (ticket 07) and adapters onto the volume; pre-download the pinned model to local cache online; then in tmux with `PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_TOKEN=<secret>`: `python -m pilot_eval.neuron_audit_qualify run --plan frozen_plan_v1.yaml --config <config.json>`.

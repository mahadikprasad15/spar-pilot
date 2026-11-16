# Review request — round 6: why the real runs keep failing, and what else will

For the external reviewer (Codex). Reply in `review-round-6.md` in this folder: numbered points, each tagged
**blocking** / **non-blocking** / **question**, with file and line. Claude answers in `review-response-round-6.md`.

## Why this round

Every real run so far has failed on something the CPU tests did not exercise. Each failure was cheap to fix, but
each one costs the user a Pod setup (about 30 minutes) and their confidence. **The question for this round is not
"is the design right" (rounds 1–5) but "what will break the next time this runs on a real GPU, before it runs".**
Find the next failures, not the last ones.

## What happened, in order (all on 2026-11-16 unless noted)

| # | Where | Failure | Cause | Fix | Evidence |
|---|---|---|---|---|---|
| 1 | Colab 06a r1 | stopped at `bind_sources` | GSM8K test parquet not in the prep snapshot; message said "differs" not "missing" | fetch step `attribution_backend fetch-gsm8k`; message fixed | `docs/agents/experiment-lessons.md` (06a r1) |
| 2 | Colab 06a r1 | inventory verify failed, then 8 cascading errors | rebuild-from-originals check where only the sealed copy exists | `attribution_traces.inventory_integrity` (seal-only); `needs=` skips dependants | same |
| 3 | Colab 06a r2 | CUDA OOM (40 GB) | selection re-ran attribution over all rows in one batch | read sealed units; bounded batches | same |
| 4 | Colab 06a r3 | FD check failed (max rel err 60.8) | FP32 finite-difference reference is noise-limited; `.double()` is not FP64 for Qwen2 | `attribution_instrument.full_fp64`; FP64 FD reference (user decision) | `06a-r3-fd-diagnosis.md` |
| 5 | Pod setup | prep archive downloaded as 0 bytes, no error | `HF_HOME` defaulted onto the sl-0 FUSE volume (`/workspace/.cache`) | `export HF_HOME=/root/hf-home` (runbook) | lessons (pod-1 setup) |
| 6 | Pod launch | — | launcher calls plain `python`; venv not active | `source /root/venv/bin/activate` (runbook) | `scripts/attribution_overnight.sh` |
| 7 | Pod launch 1 (`rehearsal-pod-1`, `d5ab854`) | preflight `TypeError: string indices must be integers` after 24 s | `TransformersBackend.preflight` passed a path to `neuron_audit_pod.disk_problems`, which takes `local_disk(path)` | `d2e52d3`, red/green test running the real preflight body | HF `neuron-audit/rehearsal-pod-1/aff49aa4…`; lessons |
| 8 | Pod relaunch | would be refused | chain state on the volume is at fixed paths; a resealed record (new commit) under the same run ID is refused by `seal_run`; a new run ID by `state.json` | operator moves `$VOL/attribution` aside; run ID `rehearsal-pod-2` | lessons |
| 9 | Pod launch 2 (`rehearsal-pod-2`, `2bce6a6`) | qualification `TypeError: '<=' not supported between instances of 'str' and 'int'` after 761 s (calibration traces had sealed) | launch record `compute.fp32_peak_flops: 19.5e12`: PyYAML (YAML 1.1) reads an exponent without a sign as **text**; `attribution_gpu.mfu` compared it with 0. `seal` never checked types | `attribution_chain.mistyped` + `TEXT_FIELDS`: `load_launch_record` (so `seal` too) refuses text where a number belongs and any bool; YAML now `19.5e+12` | HF `neuron-audit/rehearsal-pod-2/` (pre-export summary); lessons |

Launch 1's teardown worked end to end on the real Pod (volume copy verified, HF export verified, final summary
uploaded, `runpodctl stop pod` returned 0). The failure path is the one part that has been exercised for real.

### Launch 2 failure

Summary (from the HF backup): state `failed`; budget used preflight 23 s, qualification 761 s; failure
`qualification: TypeError: '<=' not supported between instances of 'str' and 'int'`; volume copy verified (launch,
qualification, calibration generation and rows, preflight compute record). Teardown again ran to the Pod stop.

Note what this says about the class of bug: items 7 and 9 are both **type or shape mismatches at a boundary**
(path vs dict; text vs float), found only when real inputs reached that line. Please look for every other boundary
where an input from YAML/JSON, the environment, a subprocess, NVML or `torch.cuda` is used without its type being
checked, and every place the Python side assumes a unit or type the other side does not promise.

## The pattern to attack

The CPU tests drive `attribution_stages.Stages` and `attribution_chain.Chain` with **fakes**
(`TinyBackend` and `Scripted` in `tests/test_attribution_stages.py`; `Calls` in `tests/test_attribution_chain.py`).
06a on Colab used the real `TransformersBackend`, but through its own driver (`attribution_rehearsal.py`), which
calls only some `Stages` methods. So code exists that has **never run on real inputs**:

- `TransformersBackend.preflight` (failed, fixed), `untuned`, `untuned_generator`, `unresolved_arms`, `cohort_items`,
  `parameter_count`, and `stage_d_generator`'s edited-checkpoint branch through `Stages` (06a ran it via its driver).
- `Stages`: qualification as a whole (`_sweep`, `_generation_batches`, `_worst_case_check`, sign-off),
  `base_traces`, attribution and freeze (`build_freeze`, `seal_freeze`), `stage_c`, `stage_d`, `report`.
- `Chain` on a real Pod: start, stage caps, sign-off, resume — only teardown has run for real.
- `scripts/attribution_overnight.sh` and `attribution_watchdog.sh` outside a mock.

Please:

1. **Diff fake against real.** For every method a fake implements, compare its signature, return type and shape,
   units (bytes vs GB, seconds), device (CPU vs CUDA tensors), dtype, and side effects with the real
   `TransformersBackend` / `neuron_audit_qualify.TransformersBackend`. Item 7 was exactly this class.
2. **Walk the real qualification path by reading it** with real sizes: A100-80GB, Qwen2.5-1.5B FP32, ~20 calibration
   rows, token sweep 2k→65k, generation batches 1/8/16 at 1,024 new tokens, FP64 model copy alongside the FP32 one,
   the untuned model loaded in addition. Memory: what is resident at each point, and is anything not freed (OOM
   handling in `_sweep`, `is_oom`, `empty_cache`)? Time: does it fit the 60-minute qualification cap?
3. **Find every `except` that turns a real error into a refusal or a stop**, and every check whose threshold could
   fail on real hardware for reasons other than a bug (e.g. `disk_problems`' 60 GB minimum, batch-invariance
   tolerance 1e-4, the generation batch rule's flips/stop changes, memory fraction 0.80).
4. **Propose the cheapest real-path test** that would have caught items 7–9 on CPU (e.g. the whole `Stages` chain on
   a tiny random Qwen2 + PEFT through the real `TransformersBackend`, with only CUDA and the Hub stubbed).
5. Re-check the deferred round-5 items 1–4 (ticket 11) only insofar as they affect a supervised rehearsal.

## Where the code is

| File | What |
|---|---|
| `pilot_eval/attribution_chain.py` | launch record, `Chain` (start, stages, caps, sign-off, teardown, stop), CLI `seal`/`run`/`deadline`/`watchdog-copy` |
| `pilot_eval/attribution_stages.py` | `Stages`: preflight, qualification (`_fd_check`, `_sweep`, `_generation_batches`, `_worst_case_check`), base traces, attribution/freeze, Stage C, Stage D, report; `build()` |
| `pilot_eval/attribution_backend.py` | `TransformersBackend` (real model/tokenizer/data), `bind_sources`, `fetch_gsm8k`, cohort/rehearsal draws |
| `pilot_eval/neuron_audit_qualify.py` (`TransformersBackend`, line ~236) | the joint audit's loader that the attribution backend wraps |
| `pilot_eval/attribution_instrument.py` | attribution (`Instrument`), `full_fp64`, gates |
| `pilot_eval/attribution_ablation.py` | removal, edited checkpoints, selection, `StageC` |
| `pilot_eval/attribution_traces.py` | generation (`HFGenerator`), sealing, sources |
| `pilot_eval/attribution_units.py` | sealed runs, summaries, `Budget`, export |
| `pilot_eval/attribution_report.py` | statistics and report |
| `pilot_eval/attribution_gpu.py` | NVML sampler, MFU |
| `pilot_eval/attribution_rehearsal.py` | the 06a Colab driver (the only real-GPU path so far) |
| `scripts/attribution_overnight.sh`, `scripts/attribution_watchdog.sh` | launcher and watchdog |
| `tests/test_attribution_*.py` | CPU tests (202 pass); fakes in `test_attribution_stages.py`, `test_attribution_chain.py` |

Context: `spec.md`, `pod-runbook.md`, `launch-record-pod-rehearsal.yaml`, `06a-r3-fd-diagnosis.md`,
`review-round-5.md` + `review-response-round-5.md`, `issues/11-overnight-chain-runner.md`,
`docs/agents/experiment-lessons.md` (sections from "Attribution Colab rehearsal 06a r1" on). Branch
`claude/bold-gauss-5qkzwi`, head at or after `2bce6a6`.

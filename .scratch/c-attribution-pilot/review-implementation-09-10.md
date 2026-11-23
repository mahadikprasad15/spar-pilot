# C-attribution tickets 06–10: critical implementation review

Reviewed `c-attribution-pilot` at `0be77cf` (including fix `0be77cf` for the earlier 01–06 review). Scope: major correctness, Pod completion, recovery and result-integrity risks. No real GPU, Pod, Hugging Face operation or network action was performed. Tests in this review use local CPU fixtures and fakes.

## Verdict

The earlier CUDA-device, zero-signal finite-difference, empty model hash map, generation-unit identity and signed-top fixes are present. The previous FD row-at-a-time change is also present. I would fix the issues below before an unattended production run, then run the bounded GPU rehearsal; CPU tests cannot establish GPU memory fit, actual IEEE settings, a real Hub readback or safe Pod shutdown.

## P1 — FP32 is named but the IEEE arithmetic policy is not enforced or recorded

`pilot_eval/c_attribution_backend.py:65-72,122-145` loads FP32 weights and records `dtype=float32`, but this backend never sets or reads back CUDA matmul's TF32/IEEE precision or the cuDNN settings. The earlier instrument explicitly invokes `set_ieee_precision` and records the results (`pilot_eval/attribution_backend.py:163-177`; `pilot_eval/neuron_audit_collector.py:34-81`). Dtype and disabled autocast alone do not enforce IEEE matmul.

**Consequence:** if the active backend settings differ, qualification and production may not have the intended FP32 arithmetic and the runtime record cannot demonstrate which policy was used. This is code-inspected; no actual TF32 computation or GPU discrepancy was observed here. Local CPU-only torch reported matmul `highest` and CUDA matmul TF32 disabled, which cannot qualify an A100.

**Fix/test:** use the pinned, established precision setter and readbacks before measurements, record the actual settings with the runtime, and fail on mismatch without changing tolerances. Exercise the setter/readback path in a focused test and verify the recorded values in the real GPU rehearsal.

## P1 — FP64 reference remains alive during memory qualification and later loads

`pilot_eval/c_attribution_stages.py:234-241,269-276` keeps caller references to `reference`, `model` and `inst` after invoking `backend.release`. `pilot_eval/c_attribution_backend.py:153-159` deletes only the callee's local name, not the caller's references. A local weak-reference test reproduced the retained object. The subsequent peak-memory test consequently runs with the FP64 reference still live; at an arm transition, old objects can also coexist while new ones load.

**Consequence:** inflated peak measurements can fail a valid memory gate, and additional overlapping models may cause an OOM. The actual impact on an A100 has not been measured. This is especially relevant given the prior Colab memory failure and the mandatory FP64 reference.

**Fix/test:** release all owning references, including instrument/model holders, at clear lifetimes and in error paths before collecting and emptying cache. Test object lifetime with weakrefs; on GPU measure peak allocation at each transition and the intended production batch, without weakening the approved gate.

## P1 — A base-model hash can attest to the wrong source

`pilot_eval/c_attribution_bundle.py:99-126` now correctly requires both model files, but calculates their expected hashes from whichever local files it is handed, while unconditionally declaring the pinned Qwen revision. `check_base_model` at `:140-152` then proves only that the Pod matches those self-recorded bytes. A local fixture with arbitrary fake files was accepted and labeled with revision `989aa798…` (a real model loader would later reject those particular fake bytes).

**Consequence:** a valid but wrong local Qwen snapshot could become the trusted baseline and produce apparently coherent results under the wrong model. This is a demonstrated provenance-check gap, not evidence that the user's actual files are wrong.

**Fix/test:** establish expected file hashes from a trusted, independently bound source for the pinned revision or verified historical model manifest, rather than from the files being certified. Test that two valid-looking, hash-consistent local snapshots cannot both be labeled as the pinned revision. Preserve the existing fail-closed missing-file check.

## P1 — Progress and failure markers do not provide the promised recovery

`pilot_eval/c_attribution_stages.py:325-377,480-509` seals work under `local_root`; `Context.boundary` (`c_attribution_chain.py:260-264`) updates status/heartbeat only. The first volume copy occurs during teardown (`:471-474`) or through the watchdog after a process crash. A local fixture observed a completed rows artifact on local storage and not on the volume before teardown. If the entire Pod/container disappears before teardown or watchdog recovery, completed units still only on the ephemeral disk are lost. A mere SSH disconnect is different: tmux and the Pod may continue normally.

Separately, `teardown` writes the `done` marker for every state, including `failed` or `stopped` (`:506-507`), while `precheck` refuses any run with that file (`:241-247`). A fake-stage run reproduced `failed` followed by `ResumeRefused ... already finished (failed)`. The empty-disk resume test explicitly unlinks the marker to mimic an interrupted teardown (`tests/test_c_attribution_chain.py:199-215`), so it does not test recovery after a handled failure.

**Fix/test:** durably mirror verified work during the long computation at a practical checkpoint frequency, and distinguish an attempt ending from the study completing, so the supported recovery procedure can resume verified prior work. Test a handled failure and an abrupt Pod/local-disk loss independently, checking precisely what survives. Do not infer Pod death from an SSH disconnect.

## P1 — Resume trusts old preflight and qualification without checking the current environment

`pilot_eval/c_attribution_chain.py:378-385` skips any previously complete stage once its old output seal verifies, including `preflight` and `qualification`. Real preflight (`pilot_eval/c_attribution_stages.py:149-175`) is where the current input bundle, model files, catalog and runtime are checked. A local fake-stage resume reproduced a complete run even when the new preflight callback would raise on invalid current inputs: both preflight and qualification were marked `resumed=True` without being called. The fixture simulated a crash before a done marker, as the existing resume test does.

**Consequence:** after a supported crash/takeover, changed local model files or a different runtime/GPU can escape the current checks; a historical sealed receipt only proves its own bytes, not today's environment. No changed production input or actual GPU result was observed.

**Fix/test:** perform current-session preflight on every start; bind reuse of qualification and output units to the actual verified model, tokenizer, code, GPU/runtime and numerical settings. Refuse incompatible reuse or requalify under an appropriate identity. Test an empty-local-disk restore with modified model bytes and a changed runtime as well as the matching case.

## P1 — A timed-out copy can still be running when the Pod is stopped; completeness is overstated

`pilot_eval/attribution_chain.py:250-264` implements timeout by returning while a daemon thread may still be working. `pilot_eval/c_attribution_chain.py:471-503` then proceeds through export and requests stop. In a fake-copier test, timeout and the fake Pod stop occurred while the copy thread was blocked and still alive. A slow or stuck copy could thus be interrupted by Pod stop; whether useful data is lost depends on what was already copied/exported. No real stop was attempted.

Also, `copy_sealed` (`attribution_chain.py:317-358`) says `verified=True` if no copies failed, including if nothing existed. The runner calculates expected missing paths (`c_attribution_chain.py:480-495`) but still includes the volume in `verified backups`. Expected stage outputs are added only once each stage returns (`:403-414`), so partial in-progress work is not a complete expected-work inventory. A local fake run reproduced an empty copy labeled verified despite missing expected paths. An HF readback verifies uploaded bytes, not automatically that all intended work was included.

**Fix/test:** ensure timed-out copy/export processes have actually ended before stop, using a cancellation/join design compatible with the bounded fallback. Separately report verified copied bytes versus complete expected coverage; only call a backup complete after matching the reconstructed expected units/runs. Test slow copy and a missing expected unit with a fake stop and a clean restore.

## P1 — The recorded Pod ID is not checked against the Pod executing the launch

The stop command uses only the sealed record's `pod_id` (`c_attribution_chain.py:150-156,510-518`). The launcher and precheck do not compare it with the current `RUNPOD_POD_ID` (`scripts/c_attribution_launch.sh:31-44`; `c_attribution_chain.py:241-249,578-595`). A local fake showed precheck accepting `current-pod-other` with a record that would stop `podtest1`.

**Consequence:** using a stale record on another Pod could request a stop for the old Pod and leave the current one billing; in a shared account the old Pod might now have other work. A foreign live/ambiguous writer marker is separately refused, so this is not a claim that every automatic replacement resume succeeds.

**Fix/test:** before taking watchdog ownership or launching, bind the stop target to the actual current Pod and refuse a mismatch. Design any intentional cross-Pod recovery explicitly, including the writer ownership and identity rules. Test with different current and recorded IDs and a fake stop; no real Pod needs to be stopped.

## P2 — Required scientific results are missing or not machine-readable in the final output

`c_attribution_stages.py:418-426` calculates neuron bootstrap only for `per_layer` before selecting reads; no later step computes intervals for the selected reads and controls, which the spec prioritizes (`.scratch/c-attribution-pilot/spec.md:194-201`). Generation creates untuned baseline units (`:502-505`), but `report` loads only the arm conditions (`:513-535`), omitting the requested untuned row (`spec.md:395`).

Separately, bootstrap helpers return NumPy arrays (`c_attribution_stability.py:170-214`), and these are passed to `seal_run` (`c_attribution_stages.py:444-445,463-469`). `attribution_units.py:136-141` uses `json.dumps(..., default=str)`. A CPU reproduction showed arrays becoming strings such as `"[-1.9625 -2.9625]"`, rather than numeric JSON arrays; the report then carries those strings, with NumPy's display precision. The sealed raw per-question data may permit recomputation, but a successful run would not directly deliver the specified readable intervals or baseline.

**Fix/test:** after selection, compute selected/control intervals using the frozen selection, include untuned in the report, and serialize numeric arrays losslessly as numeric lists or a documented array format. Round-trip a small real `aggregate_freeze → report` output and compare actual numeric values, coverage and types against the input fixture and spec.

## Verification and limits

- Focused suite: `185 passed, 2 warnings in 70.44s` for `tests/test_c_attribution*.py`, using the project venv and local shell shims for the macOS test environment.
- The session-start full suite did not pass: it stopped after 41 passes at an unrelated notebook test because `IPython` was not installed. The default system Python also has a broken Torch dylib; neither failure establishes a defect in this branch's C-attribution logic.
- The timeout/copy and incomplete-copy scenarios were reproduced with fakes; the recovery, provenance and Pod-ID scenarios were reproduced in local fixtures. The retained-model reference was reproduced with a weakref. Precision and actual GPU OOM are risks from inspected code, not observed A100 failures. Report serialization was reproduced on CPU.
- Sampled-batch replay remains explicitly `not-run`, as already documented by the author. No model provenance, unattended execution, external backup, final report or Pod-stop guarantee is established by CPU tests alone.

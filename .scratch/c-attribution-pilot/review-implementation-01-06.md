# C-attribution tickets 01–06: critical implementation review

Reviewed tickets 01–08 at `3abca74` against `ed2521b`. Limited inspection of 07–08 is included because they determine whether the Pod can safely run and resume. Claude then committed ticket 09 at `e10ca4d` during this review; that runner has not been fully reviewed. Scope: high-impact scientific and Pod-readiness defects; no GPU execution.

## P1: no-op qualification builds greedy prompt on CPU

`pilot_eval/c_attribution_checks.py:265-285` builds the normal inputs on the model device via `Instrument.batch`, but its nested `greedy()` creates `torch.tensor(...)` without a device. On a default CUDA run, this leaves `input_ids` and `attention_mask` on CPU while the Qwen model is on GPU. Transformers 4.57.6 warns on a device mismatch; Qwen2 rotary embedding combines model-device `inv_freq` with `position_ids` derived from the CPU inputs, so a device mismatch is expected. CPU test `test_noop_passes_for_an_empty_context_and_fails_when_a_weight_is_left_modified` cannot detect it. The new runner and stage integration were not GPU tested; no real GPU failure was observed. Construct the prompt on the same device as the model and exercise the real no-op path during the GPU rehearsal.

## P1: valid zero-signal input yields an infinite relative error and fails FD qualification

`pilot_eval/c_attribution.py:51-63` intentionally gives all-correct and all-wrong groups exactly zero leave-one-out coefficients. `pilot_eval/c_attribution_checks.py:153-155` maps FD = 0 to infinite relative error, even when the attribution is also exactly 0. Reproduced with the existing tiny Qwen/PEFT fixture and all-zero weights: both values 0, median and maximum relative error `inf`, `passed=False`. The spec records 61/150 online and 70/150 offline all-same groups, and does not require mixed groups in the finite-difference subset. This is a conditional false stop, not an observed Pod failure. Select informative mixed groups and reject a zero-signal diagnostic as uninformative before gating, while keeping the approved tolerances unchanged.

## P1: model hash verification can succeed with zero expected hashes (ticket 08)

`pilot_eval/c_attribution_bundle.py:109-117` silently skips missing base-model files when building the bundle. `:131-139` iterates only recorded hashes, so an empty hash map passes even when the model path does not exist. Reproduced: `build('/private/tmp', [], repo_root='.', model_dir='/private/tmp/nonexistent-base-model-c-review')` yields `{}` for base file hashes, and `check_base_model` returns True on that nonexistent directory. The actual input fixture explicitly builds with a nonexistent model path. Require a trustworthy expected hash for every required base-model file, bind to the intended revision, and fail closed at build/verification if any hash is unavailable; keep an explicit fixture-only route separate.

## P1: sealed generation units are reused after their actual configuration changes (ticket 07)

`pilot_eval/c_attribution_generation.py:145-155` checks existing unit hashes and question IDs, but ignores the saved condition, adapter hash and decoding settings before reuse; `:167-170` records several of these values. In a tiny-model fixture with generation stubbed, a unit written with cap 6 returned `reused` when called with cap 1, and separately returned `reused` after modifying a LoRA weight. Thus one run directory can mix old responses with new model/configuration results if its caller resumes after a change. The new ticket-09 runner binds its state to a launch-record hash, which helps if every relevant input is bound there; its real C-attribution stage implementation is not yet present, so an end-to-end guard is not established. Bind the unit to the full relevant model, freeze/condition, prompt/gold, tokenizer/scorer, decoder, batch, code/runtime and launch identities, and refuse reuse when any differ.

## P2: sign-stability uses a different top set than selection

`pilot_eval/c_attribution_stability.py:98-107` labels the statistic sign agreement among top components but chooses the top by absolute value; the spec (`spec.md:169-173`) and selection use largest signed dJ/dλ. Large negative components can displace the helpful positive components. Reproduced with A=[4,3,-100,-90], B=[-1,-2,-100,-90], k=2: current function returns 100% agreement, whereas the signed-top sets' union agrees 0%. This is a descriptive-report error, not an execution gate. Compute sign agreement on the documented signed top sets, or explicitly change the intended metric and label if that is the research decision.

## Precision and validation context

The existing `full_fp64` path addresses known Qwen2 FP32 islands in RoPE, RMSNorm and attention; no GPU validation was performed here. `surrogate_finite_difference` forwards all supplied rows together and materializes FP64 full-vocabulary logits: a hypothetical 16 × 1,151 × 151,936 logits array is 20.85 GiB before weights and intermediates. This is a size estimate, not an observed OOM; bound/measure the actual FD batch in rehearsal without weakening the gate.

The focused venv CPU suite for tickets 01–08: 138 passed. Independent probes reproduced empty-model-hash success, reuse after adapter/decoder changes, all-zero FD failure, and sign-statistic discrepancy. The startup full suite was interrupted after 59 passed and one unrelated notebook test failed because IPython was missing; it is not a full-suite pass. No CUDA, HF online, or RunPod end-to-end test was performed. Ticket 09 was committed during review and has not been fully reviewed; real stage integration and real-path/rehearsal tickets remain necessary.

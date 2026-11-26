# Response: C-attribution implementation review (tickets 01–06, read through 08)

Review: `review-implementation-01-06.md` (Codex, at `3abca74`). All five findings accepted and fixed in one pass,
each red/green, with no tolerance changed. Ticket 09–10 code (committed separately, before this pass) is
included where a finding touched it.

| # | Finding | Decision | Fix | Test |
|---|---|---|---|---|
| P1 | No-op check builds its greedy prompt on CPU | Accept | `noop_check` builds the prompt on the model's device | `test_noop_greedy_prompt_is_built_on_the_model_device` (device-recording model; CPU cannot reproduce CUDA) |
| P1 | All-zero weights give infinite FD relative error and a false stop | Accept | `surrogate_finite_difference` refuses a zero-signal input as uninformative (`CheckFailure`) before any gate, and items where both the FD and dJdλ are exactly 0 are reported as uninformative, not as errors. Qualification already picks rows from mixed-reward questions (`Stages._fd_rows`). Tolerances unchanged. | `test_zero_signal_fd_is_refused_as_uninformative_not_failed` |
| P1 | An empty base-model hash map verifies nothing | Accept | `build` requires every base-model file (`config.json`, `model.safetensors`) and fails closed (`missing: base model …`); a fixture without a model must say `base_model='not-bundled-fixture'`, which `check_base_model` then refuses on the Pod. `check_base_model` refuses an empty or partial hash map. | `test_base_model_hashes_are_required_and_fail_closed` |
| P1 | Sealed generation units reused after their configuration changes | Accept | Each unit stores an identity (condition without its prediction, cap, EOS, pad, greedy arguments, padding, adapter hash before the condition, prompt-id and gold hashes, scorer); reuse requires it to equal the current one, else `GenerationError`. The runner additionally binds all state to the launch-record hash. | `test_a_sealed_unit_is_not_reused_after_its_cap_or_adapter_changes` |
| P2 | Sign agreement chose the top set by absolute value | Accept | `sign_agreement` uses the signed top sets (largest dJdλ), as the spec and selection do. The absolute-value variant is not reported. | the review's counterexample (A=[4,3,-100,-90], B=[-1,-2,-100,-90], k=2 → 0%) |
| note | FD materializes FP64 full-vocabulary logits for all rows at once | Accept as a memory risk | `_f` evaluates F row by row (F is a sum over rows), so peak FD memory is one row's logits; values are unchanged up to FP64 summation order. The rehearsal still measures peak memory. | existing FD tests unchanged |

Also found by ticket 10's real path, outside this review: `resolve_processors` patched the PeftModel, but
generate resolves processors on the inner transformers model, so check 9 / (f) raised on every real arm. Fixed
with a PEFT regression test (lesson added).

Open after this pass, for the full pre-GPU review: check (b) replay is recorded as not run (it needs the source
runs' sampled path); the runner, stages and real-path test (tickets 09–10) have not been reviewed.

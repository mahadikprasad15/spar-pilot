# C-attribution implementation review follow-up (tickets 06–10)

Reviewed commit `6dfc11a` against the [original review](review-implementation-09-10.md) and [Claude's response](review-implementation-09-10-response.md). This round asks for two fixes. The shutdown issue is the blocker for an unattended production run.

## P1 — Computation success can stop the Pod without a complete backup

**Evidence.** A local CPU chain ran every computation stage successfully with an empty volume copier, a failing fake Hub, and a fake Pod-stop command. It returned `state=complete`, reported zero complete backups, requested the fake stop, and wrote `done`. In `pilot_eval/c_attribution_chain.py:575–599`, teardown computes backup completeness but still calls `_stop()` and closes the run ID based on computation state alone. No real Pod or Hub write was involved.

**Fix and acceptance test.** Keep the run recoverable when neither destination has a complete verified backup. Separate computation completion from backup completion. Exercise the launcher and watchdog as well as the worker, since a deliberate exit without `done` can trigger watchdog recovery and a stop request. Reproduce the double-backup failure with fakes and assert that no stop is requested, the run stays eligible for backup recovery, and the operator can see why. Then make a destination succeed and show that the completed work can be backed up and the run closed without repeating successful computation.

## P2 — Selected-read and matched-control intervals are absent from the final report

**Evidence.** `pilot_eval/c_attribution_stages.py:475` saves the selected intervals in each arm's `bootstrap.json`. The report builder at `:557–585` reads the generation summaries and adapter comparison but never loads those saved intervals. The untuned row and numeric JSON serialization are fixed.

**Fix and acceptance test.** Surface the saved intervals in the final report with their arm, group, target or control role, layer and neuron index, and the exploratory interval label. Extend the CPU whole-chain test to read the final report and compare its interval values and types with the sealed aggregate artifact. Existing artifacts suffice for rendering this once the report code is fixed.

## Decisions confirmed

- **Use a new run ID when the qualified setup changes.** Fresh qualification cannot carry backward to previously computed units. The current check catches the changes it fingerprints. CUDA driver and cuDNN build are not explicit fields. A replacement Pod also needs a fresh launch record and local work directory; the old Pod binding prevents automatic same-ID recovery on replacement.
- **Pin the model to Hub file identities.** Independently verified both pins against Hugging Face metadata for revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`: the Git blob ID for `config.json` and the LFS payload SHA-256 for `model.safetensors`.

The recheck passed 59 targeted CPU tests. The broader suite hit a missing `IPython` dependency in a notebook test and did not finish. No GPU qualification or live backup/shutdown rehearsal was performed in this review.

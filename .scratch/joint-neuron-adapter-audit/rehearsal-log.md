# Rehearsal log — joint neuron/adapter audit

Bugs and setup problems found by running the pipeline on real hardware, newest last. Every code fix has a
regression test and is on branch `neuron-audit-phase1`. Rehearsal outputs are labelled and never feed production.

## Colab rehearsal r1 (2026-11-11, Colab Pro, A100-SXM4-40GB, Python 3.13, torch 2.9.0+cu128)

| # | Stage | Symptom | Cause | Fix | Commit |
|---|---|---|---|---|---|
| 1 | notebook setup | `command failed (1)` with no error text | Colab does not show a subprocess's own stderr | `sh()` streams the child's stdout+stderr into the cell | `61797dc` |
| 2 | notebook setup | `import peft` fails: `operator torchvision::nms does not exist` | Colab's preinstalled torchvision is built for its own torch; after pinning torch 2.9.0 it no longer matches, and transformers imports torchvision whenever it is installed | uninstall torchvision/torchaudio before pinning torch (the audit uses neither) | `735ff59` |
| 3 | notebook setup | `SecretNotFoundError: HF_TOKEN` | the Colab secret was not created yet (setup, not a bug) | user added the secret; the notebook still refuses to run without it | — |
| 4 | 03b arms | every arm: `explicit padding token required`; 03b stops (reference arm failed) | Qwen2.5's `config.json` has no `pad_token_id`; the pad token lives in `tokenizer_config.json`. The CPU test model set `pad_token_id=0` in its config, so tests never hit this | collector takes the pad id explicitly from the pinned tokenizer (`<|endoftext|>`, 151643, what training padded with); 03b records it, 04 refuses a different one; plan `runtime.pad_token` | `c5d4f1a` |
| 5 | 03b summary | `inputs.code_commit` = `735ff59` although later stages ran on `c5d4f1a` | the preflight stage sealed under the old commit was reused after the fix; stage identity did not include the code commit | production stage identity includes the code commit (a code change re-runs stages); every stage seal and the summary record the commit per stage. Rehearsals may still reuse stages across fixes | `2d53bb3` |
| 6 | 05 selection | `ValueError: need at least one array to stack` | rehearsal 04 collected only the first frozen batch; longest-first packing puts long UltraChat rows first and the short GSM8K math rows only in the last batches, so 05 had zero math rows | rehearsal option `rehearsal_batch_choice: cover` picks frozen batches (unchanged) greedily so every group/split has ≥ N rows (here batches 0 and 18, 28 rows); 05 and 06 now refuse clearly when a group has < 2 rows | `558e77e` |

**Passed on the real model:** 03b with all 7 arms qualified; batched-vs-singleton logits 3.6e-4 to 5.8e-4
(gate 1e-3); no review flags; FP64 floor on GPU for every arm, no unresolved layers; 1B on all layers (time
ratio 1.03); storage rules pass (≈ 55 GB projected upload). 04 (first-batch rehearsal) collected, mirrored,
packaged, uploaded and verified every arm plus the 03b package; final manifest complete and read back.

**Not covered by Colab (Pod rehearsal):** sl-0 mount and identity file, the Pod's A100-80GB runtime fingerprint
and interpreter, Python 3.12, fresh-local-disk resume, `restore-final` from the real HF packages.

**Open after r1:** 05/06/evidence on the covering batches.

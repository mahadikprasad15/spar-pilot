# Source records (ticket 01)

Checked 2026-11-24, read-only, no GPU. Every hash below is the SHA-256 of the file bytes, computed on Prasad's Mac;
`source-records.json` has the full list (sizes, all checkpoint seals, receipts) and is the machine-readable copy.

## Where the files are

| Store | Name / path | Used here |
|---|---|---|
| RunPod volume | Global Volume **`sl-0`**, mounted at `/workspace`; experiments under `VOL=/workspace/neuron-audit` (identity file `.neuron-audit-volume.json`) | No: only production-1's per-neuron arrays need it (below) |
| HF mirror | private dataset **`Prasadmahadik/spar-sampling-sft-artifacts`** | No: the cloud container and the desktop VM both get HTTPS 403 from their proxy |
| Local (Mac) | `~/spar/pilot1/artifacts`: gold run and plan, `sequences-v2`, production-1 report | Yes |
| Local (Mac) | `~/spar/pilot1-c-attr/artifacts/imports`: `c-production-1`, restored from HF on 2026-11-22 | Yes |
| Drive | `My Drive/SPAR/spar-pilot/artifacts`: gold's original root (trained on Colab L4) | No |

Gold SFT was never on `sl-0`: it ran on Colab and its artifact-root paths are relative, so the same paths resolve
under any root.

## Gold SFT (`pilot2-sft-fp32-l4-batch8-v1`)

| File (artifact-relative) | SHA-256 |
|---|---|
| `plans/pilot2-sft-fp32-l4-batch8-v1/sft.config.json` | `e70679ab52600e5cca8c33daa1ee7c6206ed309daa5796346a9bc97c24da87e1` |
| `plans/…/training.items.json` (canonical `1b7a6be5…`, as in the config) | `f27d0d75740438adafcdc3bf56104ac6fee06f80784366e5fc4e1a2c5850d260` |
| `plans/…/evaluation.items.json` (canonical `22c5ed62…`) | `76383395095c48a048b1477dc45ab1426bd0c3ca9d02f173c8a152f9e239874a` |
| `plans/…/analysis-plan.json` | `67da92c4025d5efedb0a2e1831cb99682399ab014942f1387981dcc90143bd87` |
| `runs/pilot-2/…/pilot2-sft-fp32-l4-batch8-v1/inputs/order.json` (canonical `18ba3e0a…`) | `3a92872e79ef88c2b7534e997370fd53c21ca95b98ab36b73ec5a4b9c942c0ab` |
| `runs/…/logs/steps.jsonl` | `e4d711079fee23265d1cae2aa7edb73fb1f621f7ae7cb28ee49c5c30d7f1c284` |
| `runs/…/meta/resolved_trainer.json` | `11c793c62799ee9ad412d8b49c53265c23a5f64cc4c376e2dfed57d8eafac3b5` |
| `runs/…/checkpoints/checkpoint-64/adapter_model.safetensors` | `a20a344efabe576669b3c89701b161dd568efdf776bb512f7f128f9a8b75aeaa` |

(`runs/…` = `runs/pilot-2/Qwen--Qwen2.5-1.5B-Instruct/gsm8k/train-512-seed-42/rank1-float32/pilot2-sft-fp32-l4-batch8-v1`.)
The plan and checkpoint hashes equal Pilot 5's preflight inventory and the catalog's five checkpoint hashes.

Checks:
- `sft.load_sft` accepts the plan (every canonical hash and the label/end-turn checks pass).
- Rebuilding all 512 items with `sft.gold_item` from the pinned GSM8K parquet (`ea82612e…`, the hash in the
  `sequences-v2` manifest) and the pinned tokenizer (`tokenizer.json` `c0382117…`) gives rows identical to
  `training.items.json`. The chat-template hash matches.
- `order.json` is a permutation of the 512 IDs; `order[0:8]` is `train:3104, 2902, 4885, 1746, 6609, 1148, 6833, 2308`,
  the recomputed first batch. The code reconstruction (`gold_order_reconstruction`) reports `matches`.
- `steps.jsonl`: 64 steps, and step s's `example_ids` equal `order[(s-1)*8 : s*8]` for every s. Exactly one epoch.
- `resolved_trainer.json` agrees with every "code" row in `design.md` §2.2: LR 1e-4 constant, 0 warmup, clip 1.0,
  `adamw_torch` with `foreach=False`, betas (0.9, 0.999), eps 1e-8, wd 0, microbatch 1 × accumulation 8, 64 steps,
  seed and data seed 42, full determinism, `tf32=False`, no bf16/fp16, non-reentrant checkpointing,
  `completion_only_loss`, no packing, `max_length` 554.
- `runtime.json`: NVIDIA L4, CUDA 12.8, torch 2.9.0, transformers 4.57.6, trl 0.26.2, peft 0.18.0, datasets 4.4.1,
  accelerate 1.12.0, huggingface-hub 0.36.0 (the pinned set); code `3c47286`. `results.json`: 64 steps, 512
  exposures, 196 targets, base hash unchanged (`08d084a9…`).
- `lengths.json` does not exist in the plan or run directory. The only length record is `max_length` 554.

## Adapter-hash conflict (`gold-sft.greedy.test150.pilot2`)

Resolved as a hash-kind difference, not a different adapter. Pilot 2's evaluation config stores
`workflow.adapter_digest(checkpoint dir)`, which hashes the names and bytes of every file in the checkpoint directory
(optimizer, scheduler, RNG and tokenizer files included). The catalog and inventory store the hash of
`adapter_model.safetensors` alone. So `26fdcb24…` (step 64) and `71ee7a33…` (step 0) cannot equal `a20a344e…` and
`b26c524a…`. Both configs point at this run's `checkpoint-64` and `checkpoint-0` on Drive. The local copy keeps only
`adapter_config.json`, the safetensors file and `complete.json`, so the directory digest cannot be recomputed here.
That needs the full Drive checkpoint directory. The 0.647 vs 0.66 accuracy gap is a separate run and is not
explained by this.

## `sequences-v2` (joint audit)

`runs/joint-neuron-adapter-audit/sequences/sequences-v2/`: seal `complete.json` `c0d2b146…`, `manifest.json`
`9a2ee322…`, `sequences.jsonl` `3d68bccd…` (equals the manifest's `sequences_sha256`). `neuron_audit_sequences.verify`
passes. Math rows are GSM8K train questions: calibration 25, discovery 200, held-out 100, i.e. 325 IDs
(`ids_sha256` `6f416b3c…`), none in the gold 512.

Other train IDs used by a run (D11): a scan of every JSON/JSONL/YAML under both local artifact roots, the HF
download and the production-1 copy found GSM8K train IDs outside the gold 512 only in the joint audit
(`sequences-v1`, `extract-v1/index.json`, `analysis-v1/…/split_half.json`), all subsets of `sequences-v2`, plus
synthetic CPU fixtures under `artifacts/validation/`. The scan matches the `gsm8k:train:N` form only.

## Validation set (ticket 04, frozen)

Built with PR 13's `sparse_validation.build` at `3ef8e43`, with the local tokenizer and parquet injected in place of
`HFDependencies`. Saved in `artifacts/validation/sparse-rewiring-ticket04-v1/`.

- Exclusions: gold plan 512 + catalog `gsm8k-train-512` (read from its source, `memberships.json`) + `sequences-v2`
  325 = 837 IDs (`edfd0730…`). No text duplicates or cohort150 matches. 6,636 eligible.
- 128 IDs, seed 20261122: `ids_sha256` `5ea47bc5954a15f4570af511bca8e623c0ca7ab27fab8f1b35ccf80fea082e61`,
  `items_sha256` `cc1707353db950793e9e54fd0138c59cdb3c3d2ec61bb98a37c36537bf9909b4`.

## Plans from the real files (PR 6)

With PR 6's `sparse_rewiring_plan` at `fc702c5`, a dry seal of the draft (placeholder Pod and paths) and `plan` for
each arm on the real `training.items.json` and `order.json` both succeed: `training_items_sha256` `1b7a6be5…`,
`gold_order_sha256` `18ba3e0a0179fed7817faa65100efbcd27de26d4f8e7e7c96b625d8d172e2b55`, reconstruction `matches`,
step 1 = gold's first batch, 1,024 steps. The plan hashes themselves change with the real code commit and Pod ID.

## `c-production-1`

Under `imports/c-attribution-production/c-attribution/` (local) = `$VOL/c-attribution/c-production-1/c-attribution/` (sl-0):

| Arm | `aggregate/<arm>/neurons.npz` | `freeze/<arm>/<arm>.json` |
|---|---|---|
| online-grpo | `81e8542374202d7430f9f610686d80ec3cffd0d8934cae06829ad8b7c5a20d58` | `a50dd41d1f9037e42b17881769ea33dd7dbcf2d492a90cad671863fff6bf64b0` |
| offline-grpo | `165ade28223bfa1a550fc34b52b177f5c34b85caf37f3db29ed061ac012f45a3` | `145463611470d20c6ab6411fb9b8bdc62a5d233076c928836bc5b34d72e597a4` |

Both directories are sealed and every listed hash in their `complete.json` matches. Arrays (float64):
`dJdlambda`, `removal_J`, `read_size` [28, 8960]; `read_halves` [2, 28, 8960]; `S` [150, 28, 8960]. Freeze schema
`c-attribution-freeze-v1`; targets use the flat zero-based index `layer * 8960 + neuron`.

## Production-1 adapter attribution

Report `imports/attribution-production-1/attribution/report/report.json` `ccccc004…` (identical copy in
`~/production-1`). Top `down_proj` layer per GRPO arm (`descriptive.concentration`, from the freeze):

| Arm | trace | stop |
|---|---|---|
| online-grpo | 13 | 14 |
| offline-grpo | 14 | 14 |

**Not located: the per-neuron ranking files.** They are not in either local copy or the HF bundle, and the catalog
says the full production outputs are only on `sl-0`. From the code, they are `$VOL/attribution/freeze/freeze.json`
and the `ab` units in `$VOL/attribution/store`. That location comes from the code and runbook and is not yet checked
on the volume. Reading them needs a CPU Pod with `sl-0` attached, or a copy pushed to HF. This blocks only
ticket 10's production-1 entries, not training.

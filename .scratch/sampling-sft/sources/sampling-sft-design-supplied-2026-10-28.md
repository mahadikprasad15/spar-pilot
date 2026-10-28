# Sampling SFT Arm: Implementation Design Doc

acOct 28, 2026 · @Prasad Mahadik

## 1. Summary and scope

We add a Sampling SFT arm to X1 by porting the Karan et al. sampler into our pilot codebase, generating boosted GSM8K traces for our 512 training problems at k = 0, 2 and 5, and, optionally, training on them with gold SFT's exact trainer, configuration, batches and hardware. Gold SFT is not re-run: its existing adapters and outputs are the comparison. Sampling runs in FP32 on any single GPU type (an L40S by default), on RunPod by default, with a Colab notebook as a fallback.

**What we take from their repo** ([aakaran/finetuning-with-sampling](https://github.com/aakaran/finetuning-with-sampling)): the algorithm in `boost_math.py` (block extension, cut-and-regrow, greedy acceptance on mean per-token log-prob, whole-trace cuts), the structure of the proposer prompt (`SEQ_PROMPT_MATH` in `constants.py`), and the proposer settings (temperature 0.6, repetition penalty 1.1). Their repo is a reference, not a dependency: nothing imports it.

**What we replace:** the model, data, prompts, length cap, grader, batching, seeding, output format and the whole training side (their verl full fine-tuning launcher is not used). Our existing pilot code supplies the prompt builder, the flexible-v3 extractor, the rank-1 trainer and the evaluation harness; the new code imports them rather than copying them.

**Hypotheses served** (X1 policy-distance axis): H3a, monotone change in length, forgetting and KL along gold → Sampling (k = 0 → 2 → 5) → GRPO; H3c, Sampling SFT keeps the model's answer length; H3d, closer arms learn the gold format more slowly. H3b needs the self-correct arm, which is deferred (§15).

**Out of scope:** the internal readouts (effective write, π, residue test) that consume this arm's adapters, and the self-correct SFT arm.

## 2. Fixed decisions and configuration

Every value below is set explicitly in the run config; no library default is trusted. Values marked \[Open\] are read from the existing Pilot 2/4 configs (§3), never retyped.

| Area | Item | Value | Why |
| --- | --- | --- | --- |
| Model | Proposer, scorer, trained model | Qwen2.5-1.5B-Instruct, one checkpoint for all three roles | Data must be on-policy for the model we train |
| Model | Precision | FP32 with TF32 disabled, for generation, scoring, training, evaluation | Pilots: BF16 vs FP32 moved untuned accuracy by 25 points |
| Hardware | GPU | Sampling: one GPU type for all k runs (L40S by default). Training and evaluation: the GPU type recorded in gold SFT's runtime metadata | Sampled data depends on precision, not GPU type; trained adapters must match gold SFT's runtime |
| Data | Training problems | The same 512 GSM8K problems as gold SFT and GRPO, same ids | Matching |
| Data | Held-out | The same 150 problems as the pilots | Matching |
| Prompt | Scorer prompt | The trainer's prompt, built by importing the trainer's function | Defines what "on-policy" means |
| Prompt | Proposer prompt | §6 template; no format instruction of its own; cleaned gold | Limits gold leakage |
| Sampler | Score | Mean per-token log-prob of the response (end token included) under the scorer prompt | As in their code |
| Sampler | Accept rule | Greedy: accept if proposal score > current score | Replicates the published method |
| Sampler | Cut range | Uniform over the whole trace so far | As in their code |
| Sampler | Jump size J | 64 tokens, set directly | About 4 blocks for a \~230-token trace |
| Sampler | Max blocks | ceil(cap / J); last block truncated at the cap | No divisibility assert |
| Sampler | Steps per block k | Separate runs at k = 0, 2, 5 | The dial; k = 10 conditional (§8, Stage 4) |
| Sampler | Proposer sampling | Temperature 0.6, repetition penalty 1.1, top-p 1.0 | As in their code |
| Sampler | Stop tokens | \`< | im\_end |
| Sampler | Length cap | GRPO's `max_completion_length` \[Open\] | One ceiling for every arm |
| Sampler | Data seed | `run_seed = 0`; per-problem streams (§7) | Reproducible, common random numbers across k |
| Sampler | Retries | Up to R = 3 per wrong or truncated problem, each with a new seed | Cover all 512, to match gold SFT's problem set |
| Grading | Correctness | flexible-v3 extractor vs gold `####` number | Same definition as GRPO's reward |
| Grading | Filter | Drop wrong and truncated traces | Every arm trains on correct text only |
| Training | Trainer | Existing rank-1 SFT trainer, config identical to gold SFT | Only the data changes |
| Training | Adapter | Rank 1, alpha 1, every linear module, every layer | B0 |
| Training | Batch | 8 problems per step | Matching |
| Training | Problem set | All 512, in gold SFT's exact per-step batches; a problem with no correct trace leaves its slot empty | Same problems at the same step as gold SFT |
| Training | Steps | 64, as gold SFT; same schedule | Identical schedule and step count |
| Training | Training seed(s) | Same as gold SFT \[Open\] | Same initial A0 and data order |
| Training | Inputs | Stored token ids, loss on response tokens only | No re-tokenization |
| Baseline | Typical-NLL pass | Base model, scorer prompt, T = 1.0, n = 8 per problem, same cap | NLL offset in the gate; reuse Pilot 4's samples if they cover the 512 |

## 3. Open inputs (resolve before building)

Codex resolves items 1–6 from the existing repo and records each answer in `configs/sampling_sft.yaml` with the file and line it came from. Items 7–8 are Prasad's decisions, fixed before Stage 1.

| # | Input | Where to find it | Used by |
| --- | --- | --- | --- |
| 1 | GRPO `max_completion_length` | Pilot 4 GRPO config | Length cap |
| 2 | Prompt-building function (system message, user text, chat template call) | Pilot 2 SFT trainer / Pilot 4 GRPO data module | Scorer prompt, proposer task text, trainer |
| 3 | flexible-v3 extractor and gold-answer parser | Pilot 1 evaluation harness | Grading |
| 4 | The 512 training problem ids and the 150 held-out ids | Pilot 2/4 data manifests | Data |
| 5 | Gold SFT training config: lr, schedule, loss averaging, optimizer, seed(s), checkpoint steps, the problem ids in each step's batch, and the runtime metadata (GPU, precision, TF32) | Pilot 2 config | Training |
| 6 | Saved Pilot 4 base samples at T = 1 (the cap-setting pass), if any | Pilot 4 outputs | Typical-NLL baseline |
| 7 | Step-5 acceptance threshold for running k = 10 | Prasad, before Stage 1 | Stage 4 decision |
| 8 | Whether "k = 5 + 5 polish" is in scope | Prasad | Optional run |

If an input cannot be found, Codex stops and asks; it does not substitute a default.

## 4. Architecture and module layout

The new code is one package, `sampling_sft/`, inside the existing repo. It imports the pilot's prompt builder, extractor and trainer; it never copies them. Paths under `pilot_*` stand for the existing modules found in §3.

```
repo/
  configs/
    sampling_sft.yaml            # one config, all stages (§5)
  sampling_sft/
    __init__.py
    config.py                    # load + validate YAML; record resolved inputs
    prompts.py                   # scorer prompt (imports pilot builder), proposer prompt, gold cleaning
    model.py                     # vLLM engine (FP32), tokenizer, stop ids, HF scorer for checks
    sampler.py                   # batched block sampler (§7)
    seeds.py                     # per-problem RNG and per-request vLLM seeds
    grading.py                   # wraps flexible-v3; review CSV
    retries.py                   # retry failed problems, up to 3 times
    coverage.py                  # per-arm coverage, retries, uncovered report
    baseline.py                  # typical-NLL sampling pass (n = 8, T = 1)
    gate.py                      # data gate sheet (§8, Stage 4)
    export.py                    # stored ids + gold SFT batches -> trainer dataset
    throughput.py                # Stage 0.5 sweep
    capture.py                   # per-step gradient and optimizer logging hooks
    io.py                        # JSONL/CSV writers, atomic append, resume
  scripts/
    run_stage.py                 # CLI: python -m scripts.run_stage --stage 2 --k 5
  tests/
    test_sampling_sft.py         # T1-T14 (§10)
  notebooks/
    sampling_sft_colab.ipynb     # optional Colab runner (§13)
  outputs/sampling_sft/<run_id>/ # all artifacts (§9)
```

| Module | Imports from the pilot | Must not do |
| --- | --- | --- |
| `prompts.py` | Prompt-building function (§3, item 2) | Retype the prompt text |
| `grading.py` | flexible-v3 extractor and gold parser (§3, item 3) | Add a second definition of correct |
| `export.py` | Trainer dataset format (§3, item 5) | Decode and re-encode responses |
| `model.py` | — | Run in any dtype but the configured one |
| `sampler.py` | — | Use Python's global `random` or unseeded vLLM sampling |

Training itself is a call into the existing trainer with a new dataset path; no trainer code changes except accepting pre-tokenized ids if it does not already (§8, Stage 5).

## 5. Config schema

One YAML file drives every stage. `config.py` refuses to start if any `OPEN` value remains, and writes the fully resolved config plus its SHA-256 into each run's folder.

```yaml
run:
  run_id: sampling_sft_v1
  output_root: outputs/sampling_sft
  data_seed: 0                  # sampler run_seed; retry r uses data_seed + 1000 + r
  training_seeds: OPEN          # gold SFT's (§3 item 5)

hardware:
  sampling_gpu: L40S            # any one type, the same for every k run; recorded
  training_gpu: OPEN            # gold SFT's recorded GPU type (§3 item 5)
  tf32: false                   # enforced and checked (T13)

model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  revision: OPEN                # pin the exact HF commit used by the pilots
  dtype: float32
  max_model_len: 4096           # >= proposer prompt + 2 x cap; checked at startup
  gpu_memory_utilization: 0.85
  enable_prefix_caching: true
  max_num_seqs: OPEN            # from Stage 0.5
  max_num_batched_tokens: OPEN  # from Stage 0.5

data:
  train_ids_file: OPEN          # the 512 (§3 item 4)
  heldout_ids_file: OPEN        # the 150
  pilot_subset_size: 32         # Stage 1
  pilot_subset_seed: 0
  throughput_subset_size: 64    # Stage 0.5

prompts:
  builder: OPEN                 # import path of the pilot prompt function (§3 item 2)
  proposer_template: sampling_sft/prompts.py:PROPOSER_TEMPLATE_V1
  clean_gold: true              # strip <<...>>, '#### N' -> 'The final answer is N.'

sampler:
  k_values: [0, 2, 5]
  jump_tokens: 64
  cap_tokens: OPEN              # GRPO max_completion_length (§3 item 1)
  temperature: 0.6
  top_p: 1.0
  repetition_penalty: 1.1
  accept_rule: greedy_mean_logprob
  cut_range: whole_trace
  stop_tokens: ["<|im_end|>", "<|endoftext|>"]
  max_retries: 3

grading:
  extractor: OPEN               # import path of flexible-v3 (§3 item 3)
  drop_truncated: true
  drop_incorrect: true

baseline:
  reuse_pilot4_samples: OPEN    # path, or null to run the pass
  n_samples: 8
  temperature: 1.0

gate:
  k10_acceptance_threshold: OPEN  # Prasad, before Stage 1 (§3 item 7)
  truncation_alarm: 0.02
  correctness_alarm: 0.90

training:
  trainer_config: OPEN          # gold SFT config, copied unchanged (§3 item 5)
  batches_file: OPEN            # gold SFT's problem ids per step; reused exactly
  empty_slot_rule: skip         # uncovered problem: its slot stays empty
  capture: [adapters, optimizer_state, grads_per_step, grad_stats]  # §8 Stage 5
```

## 6. Prompts

The scorer prompt is the trainer's prompt, byte for byte; the proposer prompt is a fixed template that wraps the same user message plus the cleaned gold. Both are frozen before Stage 1 and never tuned on results.

### 6.1 Scorer prompt

- `scorer_prompt_ids(problem) = tokenizer(pilot_builder(problem), add_special_tokens=False)`, where `pilot_builder` returns the chat-templated string with `add_generation_prompt=True`, exactly as the trainer builds it (including any system message the template inserts).
- The response is appended as token ids: `scorer_input = scorer_prompt_ids + response_ids`.
- Test T1 asserts these ids equal the trainer's on 10 problems.

### 6.2 Proposer prompt (template v1)

The user turn below is wrapped with the same chat template; the assistant turn is pre-filled with the partial response's token ids, so `proposer_input = chat(user_text, add_generation_prompt=True) + prefix_ids`.

```
You are given a grade-school math word problem, a reference solution, and the beginning
of a response. Study the reference solution, then continue the response in your own words,
showing your reasoning. Keep it consistent with the reference solution, and make sure the
final answer equals the reference solution's final answer.

Task:
{TRAINING_USER_MESSAGE}

Reference solution:
{CLEANED_GOLD}

Beginning of the response:
{PARTIAL_RESPONSE_TEXT}

Continue the response from exactly where it stops.
```

- `{TRAINING_USER_MESSAGE}` is the user content the pilot builder puts in the user turn (problem plus any format instruction), so the proposer sees the same format request as the scorer and no other.
- `{PARTIAL_RESPONSE_TEXT}` is `tokenizer.decode(prefix_ids, skip_special_tokens=True)`; empty string when the prefix is empty.
- The template carries a version string, `PROPOSER_TEMPLATE_V1`, recorded in every output.

### 6.3 Gold cleaning

1. Remove every calculator annotation `<<...>>` (regex `<<[^>]*>>`).
2. Replace the final line `#### N` with `The final answer is N.`
3. Keep all other text unchanged.

Test T11 checks 20 examples by hand-written expectation. The cleaned gold is used only inside the proposer prompt; gold SFT still trains on the original gold.

## 7. Sampler specification

The sampler is their `proj_samp_vLLM` loop run in lockstep across problems: same operations per problem, batched per stage, with per-problem randomness. A run is one `(k, data_seed)` pair over a list of problems.

### 7.1 Per-problem state

| Field | Type | Meaning |
| --- | --- | --- |
| `pid` | int | Problem id |
| `ids` | list\[int\] | Current response token ids (end token kept if present) |
| `score` | float | Mean log-prob of `ids` under the scorer prompt |
| `block` | int | Blocks completed |
| `done`, `truncated` | bool | Finished at an end token / stopped at the cap |
| `rng` | `random.Random` | Cut positions, seeded per problem |
| `accepts`, `attempts` | int | MCMC counters |

### 7.2 Algorithm

```python
states = [State(pid, rng=Random(cut_seed(data_seed, pid))) for pid in problems]
for b in range(max_blocks):                          # max_blocks = ceil(cap / J)
    active = [s for s in states if not s.done]
    if not active: break
    # 1. extension: one batched generate
    ext = generate(active, prefix=[s.ids for s in active],
                   max_tokens=[min(J, cap - len(s.ids)) for s in active],
                   seed=[req_seed(data_seed, s.pid, b, -1) for s in active])
    for s, e in zip(active, ext): s.ids = s.ids + e
    # 2. score: one batched prompt_logprobs call
    for s, sc in zip(active, score([s.ids for s in active])): s.score = sc
    # 3. k MCMC steps, each = one batched generate + one batched score
    for j in range(k):
        cuts  = [s.rng.randint(0, len(s.ids) - 1) for s in active]
        props = generate(active, prefix=[s.ids[:c] for s, c in zip(active, cuts)],
                         max_tokens=[len(s.ids) - c for s, c in zip(active, cuts)],
                         seed=[req_seed(data_seed, s.pid, b, j) for s in active])
        props = [s.ids[:c] + p for s, c, p in zip(active, cuts, props)]
        new   = score(props)
        for s, c, p, sc in zip(active, cuts, props, new):
            acc = sc > s.score
            log_step(s, b, j, c, len(p), s.score, sc, acc)
            s.attempts += 1
            if acc: s.ids, s.score, s.accepts = p, sc, s.accepts + 1
    # 4. end-of-block checks
    for s in active:
        s.block = b + 1
        if ends_with_stop(s.ids): s.done = True
        elif len(s.ids) >= cap: s.ids = s.ids[:cap]; s.done = True; s.truncated = True
    checkpoint(states)                               # resume point (§7.6)
```

With k = 0 the loop is extension + score only, which is the paper's Rewrite SFT data.

### 7.3 vLLM calls

- Precision: before the engine starts, set `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")`; record all three in `env.json` (T13).
- Engine: `LLM(model, revision, dtype="float32", max_model_len, gpu_memory_utilization, enable_prefix_caching=True, max_num_seqs, max_num_batched_tokens)`, the last two from Stage 0.5. If the installed vLLM refuses FP32, stop and report; the fallback (BF16 proposer, HF FP32 scorer) needs Prasad's sign-off because it changes the k = 0 data.
- Generation: one `llm.generate` call with a list of `TokensPrompt(prompt_token_ids=proposer_input)` and a matching list of `SamplingParams(max_tokens, temperature=0.6, top_p=1.0, repetition_penalty=1.1, stop_token_ids=[im_end, endoftext], seed=req_seed)`.
- Scoring: one `llm.generate` call with `TokensPrompt(prompt_token_ids=scorer_prompt_ids + response_ids)` and `SamplingParams(max_tokens=1, prompt_logprobs=0)`; the score is the mean of the actual tokens' log-probs over positions `len(scorer_prompt_ids)` onward.
- Stop token: if a generation stopped on a stop token and vLLM's `token_ids` omit it, append it, so every finished response ends with its stop id (T7).
- A proposal can be shorter than the current trace (it stopped early); this is allowed, as in their code.

### 7.4 Seeds

- `cut_seed(d, pid) = (d * 1_000_003 + pid) % 2**31`; one `random.Random` per problem, consumed only by that problem's cuts.
- `req_seed(d, pid, b, j) = (((d * 1_000_003 + pid) * 1_009 + b) * 1_013 + (j + 1)) % 2**31`, where j = −1 is the block extension.
- No use of Python's `hash()` or the global `random` module. Retries use `d = data_seed + 1000 + retry_index`.
- Consequence: runs at different k share the block-1 extension and their common block-1 steps; outputs are reproducible up to GPU numerics (T8 counts any differing tokens).

### 7.5 Lengths and limits

- `cap` = GRPO's `max_completion_length`; `max_blocks = ceil(cap / J)`; the last extension asks for `min(J, cap - len(ids))` tokens.
- At startup, assert `max_model_len >= proposer_prompt_tokens_max + 2 * cap` (the partial response appears twice in the proposer input).

### 7.6 Checkpoint and resume

- After every block, write all states (ids, score, block, flags, counters, RNG state via `rng.getstate()`) to `state_k{k}.json` atomically (write temp, then rename).
- On restart with `--resume`, load the states and continue at the next block. Seeds are derived, not drawn from a stream, so a resumed run makes the same requests as an uninterrupted one.

## 8. Pipeline stages

Eight stages (0, 0.5, 1–6), each started by `python -m scripts.run_stage --stage N`; a stage refuses to start until the previous stage's outputs exist and its checks passed.

&#91;embedded content: pipeline · 7 stages, 2 decision loops\]

Stages run left to right and top to bottom; the only loops are a prompt revision at the pilot and a conditional k = 10 run before training.

### Stage 0: Tests

- **Run:** `pytest tests/test_sampling_sft.py` (§10), on the GPU box, before any generation.
- **Pass:** T1–T4 and T7–T13 before Stage 0.5; T5–T6 on the Stage 1 outputs, before Stage 2; T14 before Stage 5. Any failure blocks the next stage.

### Stage 0.5: Throughput test (\~10 min, sampling GPU)

- **Run:** 64 fixed problems, one block at k = 2, for `max_num_seqs` in {32, 64, 128, 256, 512} and `max_num_batched_tokens` in two values (e.g. 8192, 32768).
- **Record:** decode tokens/s, prefill tokens/s, wall time, peak GPU memory, per setting.
- **Choose:** the fastest setting; write it into the config; replace §11's estimates with the measured throughput.
- **Check:** rerun T8 at the chosen setting. The setting changes speed only, never data, because every request carries its own seed.

### Stage 1: Pilot (32 problems, k = 5)

- **Input:** 32 problems drawn from the 512 with `pilot_subset_seed`.
- **Output:** traces, step log, run summary (§9) under `pilot/`.
- **Report:** length distribution (median, p90, max); acceptance rate by step 1–5 and by cut-position decile; score gain per generated token; correctness, truncation; measured decode and prefill tokens/s and wall time.
- **Decision rules:**
  - Median length far above 230 tokens → recompute J to keep about 4 blocks per median trace, and re-estimate §11 (cost is roughly quadratic in length).
  - Truncation above 2% → inspect the truncated traces; a repetition loop is a proposer problem. The cap is never raised.
  - Correctness below 90% → review the proposer template; this is the last point at which it may change (bump to v2, rerun Stage 1).

### Stage 2: Full generation (512 problems)

- **Order:** k = 0, then k = 2, then k = 5; then the typical-NLL baseline pass, unless Pilot 4 samples are reused (§3 item 6).
- **Each run:** sampler (§7), then grading (Stage 3), then one retry for failed problems.
- **Baseline pass:** scorer prompt, T = 1.0, n = 8, same cap and dtype; saves ids, text, mean log-prob and correctness per sample.

### Stage 3: Grading, retries, common set

- **Grade:** flexible-v3 on each trace; compare numerically with the gold `####` value.
- **Review CSV:** rows with no extraction, strict vs flexible disagreement, truncation, or several candidate final numbers. A person (or the coding agent, then a person) fills `human_verdict`; corrections overwrite `correct` and are logged.
- **Retry:** each wrong or truncated problem is re-run with a new seed (§7.4), up to 3 times, stopping at the first correct trace; `retry_index` records which attempt succeeded.
- **Filter:** drop wrong and truncated traces.
- **Coverage:** write `coverage.json` per arm: covered problem ids, problems still uncovered after 3 retries (with reasons), and retried problems. Uncovered problems leave their slot empty in training (Stage 5).

### Stage 4: Data gate (no training)

Compute `gate_sheet.csv`, one row per arm (gold, k = 0, k = 2, k = 5, base samples), over each arm's covered problems (paired comparisons use problems covered by both arms):

| Measure | Definition |
| --- | --- |
| Length | Median, p90 response tokens |
| NLL | Mean per-token NLL under the scorer prompt, recomputed with HF `transformers` in FP32 |
| NLL offset | Arm NLL minus the mean NLL of the base samples for the same problem |
| Format | Share ending in `####`, `\boxed{}`, "The answer is", other |
| Gold leakage | Share containing `<<`; mean 4-gram overlap with the gold solution |
| Pass rates | Correct, truncated, retried, coverage of 512 |
| Acceptance | Rate by step and by cut-position decile (k = 2, 5) |

**Decisions:**

- **Merge rule:** if the 95% paired-bootstrap interval (over problems covered by both arms, 10,000 resamples) of the k = 5 minus k = 0 difference includes zero for both length and NLL offset, train only one of them.
- **k = 10 rule:** if the step-5 acceptance rate at k = 5 is at or above the threshold fixed before Stage 1, run a fresh k = 10 (Stages 2 and 3 again for that k).
- **Training list:** Prasad confirms which arms go to Stage 5.

### Stage 5: Training

- **Arms:** each sampler arm approved at Stage 4. Gold SFT is not re-run; its existing adapters and outputs are the comparison.
- **Batches:** gold SFT's exact per-step problem ids (§3 item 5), step for step. Each problem's slot holds that arm's correct trace; an uncovered problem's slot stays empty, so that step trains on 7 problems. The count of empty slots per arm is reported.
- **Config:** gold SFT's trainer config unchanged (lr, schedule, loss averaging, optimizer, max grad norm, 64 steps, checkpoint steps), same training seed(s).
- **Hardware:** the GPU type and precision settings in gold SFT's runtime metadata, TF32 matching; recorded in `env.json`.
- **Data:** `export.py` writes stored ids; loss only on response tokens. If the trainer only accepts text, add an ids path to it rather than re-encoding.
- **Capture during training** (only the path between checkpoints is lost if not logged live):

| Capture | When | Required? | Used for |
| --- | --- | --- | --- |
| Adapters, same PEFT format as gold SFT | Gold SFT's checkpoint steps: 0, 8, 16, 32, 64 | **Required** | Every Stage 0 experiment, post hoc |
| Checkpoint-0 equal to gold SFT's (same a₀) | Step 0 | **Required** (T15) | E0.5 and E0.2 baselines assume one shared a₀ |
| Whatever gold SFT's trainer logged (loss, lr, grad norm) | Every step | **Required** | Like-for-like training curves |
| Run manifest (config, seed, batches, GPU, precision, versions) | Once | **Required** | Provenance |
| Optimizer state (Adam m, v) | Checkpoint steps | Optional, cheap | Resume; replay |
| Full adapter gradient, pre-clip | Every step | Optional (\~300 MB per arm) | Path questions among sampler arms only; gold has no counterpart |

- **Comparability:** no Stage 0 experiment needs anything captured during training; all of them run later on saved adapters, data and the base model. Per-step gradients, if captured, compare sampler arms with each other only. At step 0 the gradient on A is exactly zero (B starts at zero); this is expected.
- **T15, shared start:** each sampler arm's checkpoint-0 adapter equals gold SFT's checkpoint-0 bit for bit (B = 0, same a₀ from seed 42); run before Stage 5's first optimizer step.

### Stage 6: Hand-off

- **This doc ends at saved adapters.** Behavioural evaluation, the internal measurements and the Stage 0 experiments come later, from what Stage 5 saved; none of them needs anything from the training run beyond the required captures in Stage 5.
- **Hand-off package:** adapters at steps 0, 8, 16, 32, 64; the traces, coverage files and gate sheet; the training logs and run manifest; all in gold SFT's layout, so later jobs can add these arms as extra inputs.
- **For later, so nothing is lost:** internal measurements should run on the same GPU and precision as the existing write reports (A100 40 GB, FP32); evaluation on the same harness and GPU as gold SFT's; Stage 0's E0.6a, when it runs, supersedes the gate's NLL-offset code.

## 9. Output schemas

Everything lands under `outputs/sampling_sft/<run_id>/`; JSONL files are appended one line at a time with flush and fsync, so a crash loses at most one record.

```
<run_id>/
  config.resolved.yaml  config.sha256  env.json
  pilot/                 traces_k5.jsonl  steps_k5.jsonl  run_k5.json  pilot_report.md
  gen/                   traces_k{0,2,5}.jsonl  steps_k{0,2,5}.jsonl  run_k{0,2,5}.json  state_k{k}.json
  baseline/              samples.jsonl  run_baseline.json
  grading/               review_k{k}.csv  corrections.jsonl
  throughput/            sweep.csv  chosen.json
  coverage_k{k}.json
  gate/                  gate_sheet.csv  acceptance_k{k}.csv  gate_report.md
  train/<arm>/           dataset.jsonl  batches.json  grads/step_{t}.pt  step_log.jsonl  trainer outputs (gold SFT layout)
```

### traces\_k{k}.jsonl (one line per problem per attempt)

| Field | Type | Notes |
| --- | --- | --- |
| `pid` | int | Problem id |
| `arm` | str | `k0`, `k2`, `k5` |
| `k`, `J`, `cap` | int | Sampler settings |
| `data_seed`, `retry_index` | int | Seed used; 0 or 1 |
| `template_version` | str | `PROPOSER_TEMPLATE_V1` |
| `prompt_ids` | list\[int\] | Scorer prompt ids |
| `response_ids` | list\[int\] | As scored, stop token included if finished |
| `response_text` | str | Decoded, special tokens skipped (for reading only) |
| `final_score` | float | Mean log-prob under the scorer prompt (vLLM) |
| `length` | int | `len(response_ids)` |
| `blocks_run` | int | Blocks executed |
| `finished`, `truncated` | bool | Ended on a stop token / hit the cap |
| `extracted`, `gold`, `correct` | str, str, bool | From flexible-v3 |
| `accepts`, `attempts` | int | MCMC counters |
| `wall_time_s` | float | Share of the run's wall time, for scale only |

### steps\_k{k}.jsonl (one line per MCMC step)

`pid`, `retry_index`, `block`, `step`, `cut`, `prefix_len`, `current_len`, `proposal_len`, `score_before`, `score_proposal`, `accepted`.

### run\_k{k}.json (one per run)

`k`, `n_problems`, `decode_tokens`, `prefill_tokens`, `n_generate_calls`, `wall_time_s`, `gpu_name`, `dtype`, versions of `vllm`, `torch`, `transformers`, the config hash, and the git commit of our repo.

### review\_k{k}.csv

`pid`, `reason`, `response_text`, `strict`, `flexible`, `gold`, `human_verdict`, `reviewer`.

### coverage\_k{k}.json

`{"covered": [pid, ...], "n_covered": int, "uncovered": [{"pid": int, "reason": "wrong|truncated", "attempts": int}], "retried": [{"pid": int, "succeeded_at": int}]}`

### train/\<arm>/dataset.jsonl

`pid`, `step`, `prompt_ids`, `response_ids`, `source` (`k0`, `k2`, `k5`); `batches.json` lists, for each step, gold SFT's problem ids and which slots are empty.

## 10. Test suite

Fourteen tests in `tests/test_sampling_sft.py`; T1–T4, T7–T13 run in minutes on the sampling GPU, T5–T6 reuse the Stage 1 outputs, and T14 runs before Stage 5. Each test names what it protects.

| ID | Test | Pass condition | Protects |
| --- | --- | --- | --- |
| T1 | Scorer prompt ids vs trainer prompt ids, 10 problems | Identical lists | Definition of on-policy |
| T2 | Stop tokens | \`tokenizer.eos\_token == "< | im\_end |
| T3 | Score agreement | vLLM `final_score` vs HF `transformers` FP32 recompute on 20 traces, abs difference ≤ 1e-3 nats/token | Precision, prompt, slicing |
| T4 | Score slicing | Hand-built case: score equals the mean over exactly the response positions, stop token included | Off-by-one at the prompt boundary |
| T5 | Round-trip | Report the share of traces with `encode(decode(ids)) != ids`; the trainer dataset holds stored ids | Training on unscored sequences |
| T6 | Within-block monotonicity | From the step log: a problem's score never falls across steps within a block | Accept-rule bugs |
| T7 | Nothing after the stop token | No response has tokens after its first stop id; finished responses end on one | Stop handling |
| T8 | Batch independence | 8 problems run alone vs inside a 64-problem batch, same seeds: count differing tokens; expect zero or a handful | Seeding; safe throughput tuning |
| T9 | k = 0 path | No step-log lines at k = 0; trace = extensions only | Dial endpoint |
| T10 | Length limits | `max_model_len >= max proposer prompt + 2 x cap`; trainer max length ≥ prompt + cap | Silent truncation |
| T11 | Gold cleaning | 20 hand-checked examples: no `<<`, last line "The final answer is N." | Gold leakage |
| T12 | Grader identity | `grading.py` returns the same verdicts as the pilot's flexible-v3 on 50 saved pilot responses | One definition of correct |
| T13 | Numerical mode | The vLLM engine starts with `dtype=float32`; TF32 flags are off in the engine process; training's flags match gold SFT's metadata | Precision, the real hardware variable |
| T14 | Batch replay | `batches.json` reproduces gold SFT's per-step problem ids exactly, empty slots marked | Same problems at the same step |

A failing test blocks the next stage. T5 and T8 report numbers rather than fail on any difference; their numbers go into `gate_report.md`.

## 11. Compute budget

The sampling dial itself (k = 0, 2, 5) should take about 20–30 minutes on an L40S, and the whole sampling session about an hour with setup, tests and the gate (about $1 at RunPod's listed $1.09/hr); an A100 in FP32 takes about 4–5 times longer. Training costs what one gold SFT run cost, per arm. True FP32 with TF32 off runs on ordinary FP32 cores, where an A100 does 19.5 TFLOP/s and an L40S about 92, so compute-bound work (prefill, large-batch decode) favours the L40S. Token counts come from a simulation of the loop on 512 GSM8K problems with a median trace of 230 tokens; times assume about half of FP32 peak and are replaced by Stage 0.5's measurements ([pricing](https://www.runpod.io/gpu-instance/pricing)).

| Run | GPU | Decode tokens | Prefill tokens | Est. time |
| --- | --- | --- | --- | --- |
| Setup and model download | Sampling | — | — | \~30 min |
| Stage 0 tests | Sampling | small | small | \~10 min |
| Stage 0.5 throughput test | Sampling | \~0.1M | \~1M | \~10 min |
| Stage 1 pilot (32 problems, k = 5) | Sampling | \~0.08M | \~0.8M | \~3–5 min |
| k = 0 (512) | Sampling | 0.13M | 2.3M | \~2–3 min |
| k = 2 (512) | Sampling | 0.58M | 6.4M | \~5–8 min |
| k = 5 (512) | Sampling | 1.27M | 12.6M | \~10–15 min |
| Retries (up to 3, \~5% of problems) | Sampling | \~+5–10% | \~+5–10% | \~2 min |
| Baseline pass (n = 8, T = 1) | Sampling | \~1.0M | \~0.4M | \~5–10 min |
| Gate (HF FP32 NLL recompute) | Sampling | — | \~3M | \~5–10 min |
| Training, per approved arm | Gold SFT's GPU | — | — | As gold SFT's run |
| k = 10, only if Stage 4 says so | Sampling | 2.43M | 22.9M | \~20–30 min |

Sampling times are for an L40S; multiply by about 4–5 for an A100 in FP32. Prefill dominates because the proposer prompt is re-read on every call; vLLM's prefix caching can cut it by up to about half.

**What moves these numbers:** decode cost grows linearly in k and roughly quadratically in trace length; if Stage 1 measures a median of 400 tokens instead of 230, multiply the sampler rows by about 3. Batching across problems keeps calls near 180 instead of about 25,000, and keeps the GPU's arithmetic units busy: in FP32, an A100 is saturated at about 20 sequences, an L40S at about 200.

## 12. Runbook: RunPod

RunPod rents a GPU machine (a "Pod") by the second; you connect to it through a browser terminal, JupyterLab or SSH, and everything under `/workspace` survives a stop but not a terminate. The steps below follow RunPod's docs ([Pods overview](https://docs.runpod.io/pods/overview), [connecting](https://docs.runpod.io/pods/connect-to-a-pod)); console labels may differ slightly.

### 12.1 One-time setup

1. Create an account at runpod.io and add credit (about $15 covers this arm with margin).
2. Optional, for SSH: generate a key on your laptop (`ssh-keygen -t ed25519`) and paste the public key into the console's Settings → SSH keys.

### 12.2 Start a Pod

1. Storage first: Storage → create a **network volume** of about 60 GB in a data center that offers both the L40S and gold SFT's GPU type. A network volume outlives any Pod and can be attached to the training Pod later, so outputs never need copying between machines.
2. Pods → Deploy, in that data center. For sampling choose **L40S (48 GB)**, on-demand, Secure Cloud for reliability; attach the network volume (mounted at `/workspace`).
3. Template: the official **Runpod PyTorch** template (it ships PyTorch and JupyterLab). Container disk 30 GB.
4. Deploy, wait for "Running", then Connect → **Jupyter Lab** (port 8888) or the web terminal.
5. For training, stop the sampling Pod and deploy a second Pod on gold SFT's GPU type with the same network volume attached.

### 12.3 Environment

```bash
cd /workspace
git clone <our repo URL> spar && cd spar      # private repo: use a GitHub token or deploy key
export HF_HOME=/workspace/hf_cache            # model downloads persist on the volume
pip install -r requirements.txt               # pilot deps
pip install "vllm==<pinned version>"          # pin; record in env.json
nvidia-smi                                    # expect the configured GPU type
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.get_device_name())"
```

The home directory sits on the container disk and is wiped when the Pod stops, so put the exports (`HF_HOME`, any tokens) in `/workspace/env.sh` and run `source /workspace/env.sh` after each restart.

### 12.4 Running stages

- Long runs go inside **tmux**, so closing the browser tab does not kill them: `tmux new -s gen`, run the command, detach with Ctrl-b then d, re-attach with `tmux attach -t gen`.
- Order: `python -m scripts.run_stage --stage 0`, then 1, then `--stage 2 --k 0`, `--k 2`, `--k 5`, baseline, 3, 4, 5.
- Each stage writes a log to `outputs/sampling_sft/<run_id>/logs/`; watch with `tail -f`, and GPU use with `watch -n 5 nvidia-smi`.
- If the Pod restarts mid-run, re-run the same command with `--resume` (§7.6).

### 12.5 Getting results out, and shutting down

1. `tar czf results.tgz outputs/sampling_sft/<run_id>`; download through JupyterLab's file browser, or `scp` over SSH, or upload to the project Drive.
2. **Stop** the Pod between sessions: the GPU bill stops; the network volume and everything in `/workspace` are kept; the container disk is wiped.
3. **Terminate** Pods freely once stopped (the network volume survives); delete the network volume only after the results are downloaded.

Pricing at the time of writing: L40S at $1.09/hr, A100 80GB at $1.79/hr; network volume about $0.07/GB/month ([pricing](https://www.runpod.io/gpu-instance/pricing)).

## 13. Runbook: Colab notebook (optional)

`notebooks/sampling_sft_colab.ipynb` is a thin runner: every cell calls the same functions as `scripts/run_stage.py`, so a Colab run and a RunPod run with the same config produce the same files. The notebook holds no pipeline logic.

### 13.1 Requirements

- A GPU runtime matching the stage: for sampling, any one type used for all k runs (Colab's L4 works, at roughly 3–4 times the L40S time; an A100 also works); for training, gold SFT's GPU type. The first cell refuses to continue if the GPU does not match the config.
- Google Drive mounted for outputs and state, because Colab sessions disconnect and the local disk is wiped.

### 13.2 Cells

| # | Cell | Does |
| --- | --- | --- |
| 1 | Runtime check | `nvidia-smi`; assert the GPU name matches hardware.sampling\_gpu or hardware.training\_gpu for the stage; check TF32 is off; print CUDA and driver versions |
| 2 | Mount Drive | `drive.mount('/content/drive')`; set `OUTPUT_ROOT=/content/drive/MyDrive/spar/outputs/sampling_sft` |
| 3 | Get code | Clone the repo with a GitHub token read from Colab Secrets (never typed into a cell); checkout the pinned commit |
| 4 | Install | `pip install -r requirements.txt "vllm==<pinned>"`; then restart the runtime once (Runtime → Restart) and re-run cells 1–3 |
| 5 | Config | Load `configs/sampling_sft.yaml`, override only `output_root`; set `HF_HOME=/content/hf_cache` (local, fast); print the resolved config and its hash |
| 6 | Stage runner | `run_stage(cfg, stage=..., k=..., resume=True)`; one cell per stage, in the §8 order |
| 7 | Status | Read `run_k{k}.json` and the state files; print blocks done, problems finished, tokens, wall time |
| 8 | Gate view | Display `gate_sheet.csv` and the acceptance curves |

### 13.3 Disconnects

- State is written to Drive after every block (§7.6); after a disconnect, re-run cells 1–5 and the stage cell with `resume=True`.
- `resume=True` is the default in the notebook, and harmless on a fresh run.
- Writing to Drive is slower than local disk; the per-block checkpoint is small, so this costs seconds, not minutes.

## 14. Confound register

Each known way the arms could differ for reasons other than policy distance, with its control and the check that shows the control held.

| Confound | Control | Check |
| --- | --- | --- |
| Scorer prompt differs from training prompt | Import the trainer's prompt function | T1 |
| Precision differs (FP32 vs BF16, TF32 on) | FP32 with TF32 off for sampling; training's mode copied from gold SFT's metadata | T3, T13; `env.json` |
| Sampling GPU type varies between k runs | One GPU type for all sampler runs | `run_k{k}.json` |
| Training hardware differs from gold SFT | Train on gold SFT's recorded GPU type | `env.json` vs gold SFT's metadata |
| Trained ids differ from scored ids | Train on stored ids | T5 |
| Arms train on different problems or orders | Gold SFT's exact per-step batches | T14 |
| Some problems lack a sampled trace | Up to 3 retries; uncovered slots left empty and counted | `coverage_k{k}.json` |
| Arms differ in supervised tokens per step | Logged; compare at matched step and matched training loss | Stage 5 step log |
| Length ceiling differs | One cap = GRPO's, for generation, training and evaluation | T10; truncation rate |
| Greedy acceptance sharpens the data | Measured, not removed: NLL offset vs base samples | Gate sheet |
| Gold style leaks through the proposer | Cleaned gold; no format line in the proposer | T11; leakage columns |
| Retries select lucky traces | Retried problems flagged | `retried` in `coverage_k{k}.json`; per-item analysis |
| Random luck differs between k | Per-problem seeds, common random numbers | T8 |
| Different correctness definitions | flexible-v3 only, as in GRPO's reward | T12 |
| Per-step gradients exist for sampler arms only | Gold comparisons use checkpoint readouts only | Stated in the write-up |
| Proposer template tuned on results | Frozen after Stage 1; versioned | `template_version` in every trace |

## 15. Definition of done, and deferred items

### Done means

- [ ] All §3 inputs resolved, each with its source file and line in the resolved config
- [ ] T1–T14 pass (T5, T8 numbers reported)
- [ ] Stage 0.5 setting chosen; measured throughput replaces §11's estimates
- [ ] Stage 1 report written; J, cap and template confirmed or changed by the decision rules
- [ ] Traces, step logs and run summaries for k = 0, 2, 5 on all 512 problems, with retries
- [ ] Review CSVs filled; corrections logged
- [ ] `coverage_k{k}.json` and the gate sheet produced; merge and k = 10 decisions recorded
- [ ] If training is approved: each approved arm trained in gold SFT's batches, config and hardware, with the Stage 5 captures
- [ ] Adapters, logs and manifests saved in gold SFT's layout, ready for later evaluation, internals and Stage 0

### Deferred

- **Self-correct SFT arm.** The baseline pass (n = 8, T = 1) already produces its raw material; adding it later needs only selection (one correct sample per problem, chosen uniformly at random) and training. Without it, H3b is untested and sharpening cannot be separated from on-policyness.
- **Gold SFT re-run with per-step gradient logging.** Not planned; gold SFT's existing outputs are the comparison.
- **Exact Metropolis–Hastings variant.** Sequence log-prob target plus the proposal correction (one extra prefill per step).
- **k = 5 + 5 polish.** Five whole-trace steps on the final k = 5 traces, about +0.32M decode tokens, run only if §3 item 8 says so.
- **A second data seed**, to measure how much the result depends on the particular traces.

## Sources

- [aakaran/finetuning-with-sampling](https://github.com/aakaran/finetuning-with-sampling): `boost_math.py`, `constants.py`, `scripts/` (read 2026-10-28)
- [Finetuning with Sampling: SFT Learns Better Than You Think](https://arxiv.org/abs/2610.02140), Karan, Chen and Du
- [RunPod pricing](https://www.runpod.io/gpu-instance/pricing), [Pods overview](https://docs.runpod.io/pods/overview), [Connect to a Pod](https://docs.runpod.io/pods/connect-to-a-pod)

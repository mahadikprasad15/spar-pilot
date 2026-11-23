# Design: sparse neuron rewiring on GSM8K

**Status:** Research design and handoff for Claude; not an implementation specification or launch authorization.  
**Date:** 22 November 2026  
**Owner:** Researcher / SPAR selective fine-tuning project

## 1. Purpose and the task for Claude

Design a small pilot that applies Max Loeffler's sparse-rewiring method to **Qwen2.5-1.5B-Instruct on GSM8K** and compares the neuron identities it selects with the project's existing SFT and online/offline GRPO analyses. The longer-term question is whether different training objectives and data sources make different neurons useful to modify or attribute.

**Claude's assignment:**

1. Read the current checkout's `AGENTS.md`, `CONTEXT.md`, relevant ADRs, experiment lessons, specs, and source code. Determine the active branch, current state, and exact artifact locations; the branch and paths in this document are search leads, not a declaration that they are current. Follow repository rules. Do not disturb the parallel attribution run or alter its records.
2. Reconstruct the **executed** gold-SFT, sampling-SFT, online GRPO, offline GRPO, evaluation, and attribution contracts from pinned source code, manifests, sealed cohorts, and saved artifacts. For each populated value, cite the exact repository file and revision and, where applicable, the artifact path, hash, and run ID. Reconcile documentation with actual execution records; mark missing or conflicting evidence explicitly.
3. Inspect Max's pinned code and determine what can be reused directly and what must be adapted, including its license and provenance. Preserve the scientific method; document intentional differences.
4. Fill the decision register and recommend a minimal, affordable pilot. Resolve routine choices only within the user's stated preferences and repository rules; label unapproved scientific choices and thresholds as proposals. Do not invent values, thresholds, paths, data coverage, or results.
5. After filling the design, turn it into a separate draft implementation spec and dependency-ordered tickets, following the repo's existing tracker conventions. Mark any unresolved scientific decisions in those drafts and identify work blocked by them; do not make up answers. Keep the distinction between design, implementation, qualification, and scientific execution clear. This document itself is **not** the spec or ticket breakdown.

The user wants the settings from their SFT runs carried over where that improves comparability, alongside the defining parts of Max's method. When these conflict—for example, read-update learning rate, precision, or mask optimizer—show the trade-off and recommend a choice rather than silently taking a default from either project. Do not start training or spend GPU budget during design reconstruction.

## 2. Sources and what has actually been checked

### Max's work and code

- **Primary release:** [Sparse Rewiring: owl-rewired-neurons](https://github.com/maxsloef/owl-rewired-neurons). It links the 89-neuron model, owl dataset, evaluations, and atlas.
- **Relevant trainer:** [`train/train.py`](https://github.com/maxsloef/owl-rewired-neurons/blob/a6dc314740acbf726dcac435b4e84eebbbde49d8/train/train.py), especially `HardConcrete`, `MaskedReRead`, `setup_reread`, response-only `loss_of`, mask freezing, and checkpoint/summary logic.
- **Author's run definitions:** [`train/modal_runs.py`](https://github.com/maxsloef/owl-rewired-neurons/blob/a6dc314740acbf726dcac435b4e84eebbbde49d8/train/modal_runs.py). The pinned commit is the public `main` head observed while preparing this design; Claude should verify and record any newer revision it actually uses.
- **Rank caveat:** [`RANK0.md`](https://github.com/maxsloef/owl-rewired-neurons/blob/a6dc314740acbf726dcac435b4e84eebbbde49d8/RANK0.md). The reported 89-neuron case was **rank 4**; rank 0 in that code means independent full rows, not zero capacity. Configurations for rank 1 exist in the run definitions, but a verified rank-1 result was not located during this review. Do not invent one or infer that configured runs completed.
- **Interpretation and evaluation artifacts:** [neuron atlas](https://maxsloef.github.io/owl-rewired-neurons/), [side-by-side replies](https://maxsloef.github.io/owl-rewired-neurons/replies.html), and [`THREAD.md`](https://github.com/maxsloef/owl-rewired-neurons/blob/a6dc314740acbf726dcac435b4e84eebbbde49d8/THREAD.md).
- **Related methods paper:** Louizos, Welling and Kingma, [Learning Sparse Neural Networks through L₀ Regularization](https://arxiv.org/abs/1712.01312), the hard-concrete/L₀ background. This is **not** a paper by Max and should not be presented as the owl experiment. A separate formal paper for Max's sparse-rewiring release was not verified; check whether one now exists rather than inventing a citation.
- **Code provenance:** [`train/SOURCE.md`](https://github.com/maxsloef/owl-rewired-neurons/blob/a6dc314740acbf726dcac435b4e84eebbbde49d8/train/SOURCE.md) describes an upstream Sauers trainer and some licensing. Verify the applicable license for every file and subsequent changes before copying or vendoring; preserve attribution and pin the exact upstream commit.

### Existing project: entry points, not substitutes for execution records

- [SPAR pilot repository](https://github.com/mahadikprasad15/spar-pilot); [gold SFT experiment](https://github.com/mahadikprasad15/spar-pilot/blob/main/docs/experiments/pilot-2.md); [online GRPO experiment](https://github.com/mahadikprasad15/spar-pilot/blob/main/docs/experiments/pilot-4.md); [shared methods](https://github.com/mahadikprasad15/spar-pilot/blob/main/docs/methods.md).
- A branch observed during this design, `claude/bold-gauss-5qkzwi`, contains `.scratch/adapter-neuron-attribution/`, `.scratch/offline-grpo/`, `.scratch/sampling-sft/`, and `.scratch/pilot2-sft/`. Read the **current** files and newer launch records/results as applicable. The code repository says scientific artifacts are maintained separately and not included in the Git clone.
- The existing attribution handoff identifies useful entry points such as `neuron_audit_inventory`, `neuron_audit_sequences`, `neuron_audit_collector`, `stage0_evaluation_backend`, `scoring.score_gsm8k_flexible_v3`, token-alignment and selection modules, and sealed artifact/export machinery. Check their current APIs and applicability; some were documented as patterns rather than reusable drop-in components.

**Evidence already found:** the public Pilot 2 documentation records 512 GSM8K training questions, completion-only supervision including end-turn, no packing or truncation, unquantized FP32, AdamW, LR `1e-4`, weight decay 0, clipping 1.0, 64 updates, constant LR/no warmup, microbatch 1 with accumulation 8, and checkpoints 0/8/16/32/64. It records a rank-1, alpha-1 LoRA adapter on seven projection types. These are useful **documented leads**, not proof of which saved config/artifacts the current attribution project actually uses. The pinned model revision documented in shared methods is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; verify against the actual artifact and tokenizer. The project calls the instruction-tuned checkpoint without project adapters the **untuned model**.

**Evidence about attribution:** an observed `.scratch/adapter-neuron-attribution/spec.md` distinguishes 25 calibration, 200 discovery and 100 confirmation base-generated correct traces, and a separate 150-question ADR-0001 generation cohort. Its neuron attribution refers to terms of a `down_proj` adapter's read, not entire base-neuron ablation; its confirmed scope and current status must be checked. Other online/offline attribution work may use 150 questions × 8 generated responses. These may be distinct studies or cohorts. Do not assume that the “150 questions” in training evaluation, generated samples, attribution, and the current parallel work are interchangeable. Existing seals, prior use, discovery/confirmation restrictions, and partially known online-GRPO lineage govern any reuse.

## 3. Research questions and scope

### Pilot question

Starting from the **same untuned Qwen2.5-1.5B-Instruct checkpoint** and matched GSM8K training questions, which MLP neuron reads are selected by Max-style sparse gold-SFT rewiring? How do those neuron identities relate to the available neuron-attribution results from online and offline GRPO, and where available gold and sampling SFT?

This first comparison is exploratory. The selected object (a learned change to a neuron's reads) and the attribution object (the declared contribution of a neuron or an adapter term to a metric) are distinct. Overlap alone is neither evidence that the weights were changed the same way nor proof that the same neurons are causally necessary for both behaviors.

### Longer-term questions, not mandatory for the first run

1. Within the **same attribution method**, how similar are the neuron rankings across existing gold SFT, sampling-SFT k=0/2/5, online GRPO and offline GRPO?
2. Within the **same sparse-rewiring parameterization**, do CE and GRPO select different neurons? Use a common fixed offline response pool where feasible: e.g. correct-response CE versus offline GRPO over grouped correct and incorrect responses. They still differ in how responses are used; account for no-correct and no-reward-variation groups and their coverage.
3. With the same CE objective and model, how do gold versus sampled-solution data, or the sampling-SFT k values, affect selection? Sampling-SFT is a gold-conditioned proposal/search/filtering recipe; do not casually label it fully on-policy.
4. How sensitive are selected neurons to seed, update rank, training duration, sparsity, and response format?

Do not interpret a gold-CE-vs-online-GRPO difference as an objective-only effect: data, parameterization, trajectory, optimizer and selection mechanism differ. Do not launch the expanded factorial comparison as part of the minimal pilot.

## 4. Method to preserve from Max

For MLP neuron `(layer, neuron)`, the frozen model has gate and up reads and a down-projection write. Add **learned changes** to its gate and up reads; one learned hard-concrete gate controls those changes for that neuron. The base neuron itself is not switched off. The original base weights—including `down_proj`—remain frozen. All model parameters outside these added read changes and mask parameters are frozen. The read changes act on every applicable model-input position, including prompt positions, unless a separately declared experiment intentionally restricts them.

At rank `r > 0`, each layer and each of gate/up has `r` shared learnable input directions and neuron-specific mixing coefficients. One gate per neuron applies to both read changes. Rank 1 thus shares one input direction for gate and one for up per layer; rank 4 shares four each. Max initializes the shared directions randomly and the neuron coefficients to zero, making the additional read path a no-op at initialization; his mask `log_alpha` starts at 1 by default. Learn the shared directions, coefficients, and masks; do not freeze a factor simply because another project used a different LoRA initialization.

The training objective is:

`mean supervised response-token cross-entropy + lambda * sum(expected active neuron gates)`.

Use Max's hard-concrete parameterization and expected-activity penalty unless an explicitly approved deviation is documented. The supervised CE loss is applied to response target tokens and end-turn, using the existing SFT tokenization and loss-mask contract; prompt and padding targets do not receive direct CE. The model still processes the full prompt and answer. For causal language modeling, logits at position `t-1` predict target token `t`; verify the first response and end-turn positions with an independent small example.

The author trains with stochastic gates, evaluates with deterministic gates, and in the published case freezes a deterministic survivor set for the last 100 of 1,400 steps. In the pinned trainer, the pre-freeze deterministic gate is clipped continuous and the final freeze uses `z_det() > 0` to choose survivors, then sets them to binary on/off. Therefore **expected active count**, **nonzero deterministic gate count**, and **binary frozen survivor count** are different quantities. Record all three, the exact export rule, and the loss/behavior immediately before and after hardening; do not treat a learned probability, continuous gate and final binary mask as identical.

Rank 1 makes one aspect closer to the existing LoRA arms but is **not** equivalent to their rank-1 training: the previous arms adapt seven projection types, lack this per-neuron selection mechanism, and may differ in initialization and which factors train. Verify factor behavior in actual old checkpoints. If investigating identical shared-read initialization or freezing a factor, label it an additional controlled variant, not Max's default.

## 5. Fill from the existing experiments

Claude should replace every `VERIFY` with a sourced value or leave it explicitly unresolved. Cite source file/revision and the saved manifest path/hash; note any disagreement and which source governs. Do not infer artifact availability from a spec or from this table.

| Area | Existing value to reconstruct | Proposed new-run relationship |
|---|---|---|
| Model/tokenizer | Exact model and tokenizer IDs, immutable revisions, hashes; 28-layer/8,960-neuron shape verified from loaded config | Same untuned checkpoint/tokenizer; enumerate actual candidates and zero-based indexing |
| Training IDs and ordering | Frozen GSM8K dataset revision, split, 512 question IDs, example ordering/batches, seed, duplicate and coverage status for each arm | Prefer exact gold-SFT cohort and order; record repeated exposure over additional epochs |
| Gold answers | Exact strings/token IDs, role/message template, any system prompt, answer formatting, end token | Reuse exactly; no silent prompt or formatting change |
| Loss and masks | CE implementation, token shift, prompt/assistant boundaries, padding, end-turn, packing, token-vs-example normalization | Match gold SFT; response-only including the same ending; verify against Max's reduction |
| Sequence limits | Training caps, original actual lengths, truncation/packing policy, padding, max context | Preserve no-silent-truncation policy; preflight longest real example and report any incompatibility |
| Optimizer/runtime | Actual dtype, TF32/matmul and attention settings, package versions, gradient accumulation, clipping, optimizer/betas/epsilon, LR and schedule | Prefer matched input batching and FP32 runtime; make read LR and extra mask optimizer choices explicit |
| Steps/seeds | Original seed(s), optimizer-step counts, sample order, checkpoint steps and initialization artifacts | Reuse documented 0/8/16/32/64 points and seed where supported; add later points if useful |
| Existing LoRA | PEFT version, initial and final factor hashes, target modules, alpha, actual A/B trainability | Document differences from sparse rewiring; do not assume same integer seed means identical random draws |
| Behavior evaluation | Exact 150 IDs/order/split, prompt tokens, decoding, max new tokens, EOS/pad, scorer version, response artifacts and sealed use | Use same held-out cohort for predeclared comparable generation **if permitted by its current seal** |
| Attribution | Exact current project/branch/commit, objective(s), model checkpoints, sample IDs and response token IDs, masks, ranking definition, neuron coverage/layers, weights/signs, discovery/confirmation roles | Reuse compatible saved outputs; explicitly record missing or non-comparable rows, not zeros |
| Prior exposure | Source and cohort prior-use catalog, attribution/test seals, all known training-lineage overlaps | Preserve current restrictions; `no overlap with available evidence` is not proof of no overlap |
| Validation | Find independent validation question IDs disjoint from all training and protected cohorts | Hyperparameter/length selection uses validation only; if none available, leave tuning unresolved and report |
| Compute/artifacts | Available GPU, memory, persistence, upload/restore, runtime, current run status | Benchmark and record actual budget; do not interfere with running jobs |

The public attribution handoff says not to compute on its confirmation rows or ADR-0001 cohort before the relevant freeze. Check the current state before assigning a new use. The researcher's preference for the same 150 questions does not override an existing study's frozen procedures. Register any newly authorized use prospectively without modifying or invalidating the parallel run.

## 6. Proposed pilot and decision register

### Pilot A: sparse gold-SFT

- **Candidate:** all MLP neurons in all layers, unless verified architecture or resource constraints require a separately documented change; do not preselect only activity- or attribution-ranked neurons in the primary run.
- **Data:** exactly the earlier gold-SFT training `sample_id`s and targets, subject to verification. Use an independently identified validation set for decisions. Retain question and target IDs/hashes, input IDs and masks, token counts, and length distributions.
- **Model:** same untuned model and tokenizer revision; verify input/output shapes and neuron index mapping. No existing SFT or GRPO adapter is attached during training of this new arm.
- **Arm:** Max `reread`, gate+up, down write frozen, rank 1 first as a useful connection to the existing rank-1 arms. Rank 4 is a planned, separately budgeted Max-faithful comparison; do not claim rank 1 is an exact match to the prior adapters. If compute is constrained, discuss whether a rank-1 repeat seed is more informative than rank 4.
- **Loss:** same response-token CE reduction as gold SFT plus Max's expected-active L₀ penalty. Reuse `HardConcrete`/`MaskedReRead` semantics and preserve the actual model's Qwen2.5 MLP forward, bias, dtype, device, masks, and gradient behavior; do not just substitute a Qwen3-specific prompt helper or a bare shell command.
- **Outputs:** trained read-change factors, original gate logits, deterministic and binary masks, per-layer counts and neuron identities, losses and training logs, complete configuration, and recovery state. A selected neuron is a selected **read change**, not an ablated or newly created neuron.

### Settings needing explicit choices before launch

| Decision | Initial recommendation / alternatives | What to resolve and record |
|---|---|---|
| Read-update LR | Existing SFT used `1e-4` according to its documentation; Max's published rank-4 example uses `3e-3` | A shared LR improves recipe matching but does not guarantee similarly scaled optimization across different parameters. Propose a starting value with rationale; retain the other as a possible sensitivity run, not an automatic sweep. |
| Mask optimizer/LR | Max trainer defaults to AdamW mask LR `0.1`, distinct from read LR; verify the actual case-study flags | Existing SFT has no direct counterpart. Verify initialization, betas, weight decay, parameter groups, schedule, clipping, and whether mask gradients are clipped; do not silently inherit SFT settings for this extra optimizer group. |
| Runtime/precision | Match the actual gold-SFT FP32 / TF32 policy where practical | Max trainer defaults to BF16 frozen weights and explicitly casts some read computations. Check Qwen2.5 forward equivalence, runtime, memory, dropout/eval mode, attention backend and dtypes before copying. Any BF16 variant needs separate identification and qualification. |
| Lambda | Start with Max's `6e-4` as a reference point, **not** an expected answer for GSM8K | If useful, propose a bounded lower/higher validation-only calibration such as `3e-4`, `6e-4`, `1.2e-3`; no automatic three-run commitment. Define in advance acceptable validation performance/sparsity and cost, and report the result even if no suitable point is found. Same numerical lambda across objectives/ranks does not imply equal effective pressure. |
| Steps and schedule | Preserve checkpoints 0, 8, 16, 32, 64; propose later checkpoints, e.g. 128/256, only after budgeting | Existing 64-step constant-LR schedule and Max's 1,400-step schedule are not equivalent. Predeclare total cap, scheduler shape, freeze start and final fixed-mask phase. Do not change a 64-step cosine schedule into a longer one after the fact and claim identical first-64 trajectory. Treat step 64 as an informative reference, not equal token exposure or convergence. |
| Mask hardening | Max-style fixed-mask finishing phase, length TBD | Measure immediately before and after hardening on validation; retain both checkpoints and mask definitions. Ensure a step-64 comparison states whether masks were still continuous or frozen. |
| Seed(s) | Start with existing gold-SFT seed where verifiable | Separate data-order, read-initialization and mask-noise seeds. A common seed integer is not proof of matched randomness. A repeat sparse seed is the first stability follow-up. |
| Rank 1 vs 4 | Rank 1 first; rank 4 after pilot if budget permits | At same lambda test combined rank/sparsity effects; at similar survivor count test performance/selection at comparable sparsity. Avoid selecting via final test outcomes. |
| Validation criterion | **OPEN** | Define metric, acceptable loss/behavior trade-off, check cadence and independent data before tuning; never use the 150 test cohort to select lambda, steps or mask. |
| Compute | **OPEN** | First benchmark realistic longest examples, one forward/backward and save/restore. Estimate time, memory, storage and GPU budget for each proposed arm. |

If the earlier gold-SFT training set uses all 512 intended questions, do not silently remove some for validation. Prefer a separate clean GSM8K training-split validation subset with documented non-overlap against known training/attribution/test sources. If one cannot be established, mark selection as blocked or perform a clearly labeled fixed-reference exploratory run without tuning; do not rebrand test as validation.

### Minimal execution sequence, once separately authorized

1. **Preflight and smoke:** establish exact cohorts, code and runtime identities; CPU fixture tests; real longest-example GPU memory/numerics check; zero-update and frozen-parameter checks. Do not confuse disposable test updates with the scientific run.
2. **One initial rank-1 run:** use the frozen proposed configuration and record all checkpoints, including 64. Track response CE, L₀ term, LR by parameter group, gradients, expected/deterministic/frozen counts, and mask identity stability. If clearly degenerate, stop and diagnose before proposing a bounded validation-only adjustment.
3. **Evaluate:** compare untuned and sparse models with the existing gold-SFT / online / offline models under compatible fixed protocols. Use validation for run selection; use the protected 150 only when the applicable freeze/registration allows its planned role.
4. **Neuron comparison:** join eligible lists by exact `(layer, neuron)` and metric, as described below; preserve the original attribution source and status.
5. **Optional follow-up:** one additional seed or rank 4; later consider the matched offline-pool objective comparison. Require a new documented budget/config for each.

## 7. Evaluation and comparisons

### Keep the comparisons distinct

1. **Behavior:** same verified 150 question IDs and order, same prompt, generation settings, cap, scorer, precision/runtime and eligible prior-use status where feasible. Retain all response IDs/text/termination and score records. Report greedy and sampled results only when each protocol is actually run; mean sampled correctness and pass-at-k are different. Include capped/invalid/early-stop rates and length. Compare per question, not just aggregate.
2. **Fixed-text comparison:** on identical, role-permitted token IDs and positions across models, measure the sparse model's changes and the compatible attribution quantities. Make the per-example or per-token aggregation explicit. The same question with two different generated answers is **not** the same fixed input.
3. **Own-rollout comparison:** if using each arm's own saved samples, label it separately; model and trajectory text differ. Compare only when decoding, scorer and provenance are compatible.
4. **List overlap:** compare selected sparse read changes with the exact scope of available attribution rankings. Some existing attribution may cover only a selected `down_proj` adapter or a subset of neuron terms, not all 250,880 neurons. A missing value means unmeasured, not zero. Restrict the comparison universe to measured, compatible `(layer, neuron)` entries and report coverage. Do not imply global overlap if only a subset was measured.

For list overlap, report per-layer counts, intersection and overlap curves for predeclared rank cutoffs, with a layer-matched random-set reference; add an activity-matched reference only if the relevant comparable activity measurements exist. Report the neuron universe, set sizes, null construction, seed and uncertainty. Stratify by attribution sign or magnitude when meaningful and avoid treating a signed attribution as a survivor probability. Match data, token positions and aggregation where the two measurement methods permit it; state where they cannot match.

For performance, use paired question-level comparisons with uncertainty appropriate to the sample size, preserving the established scorer and cohort policy. A 150-question interval concerns sampled questions, not variation across training seeds. Compare at declared checkpoints and optionally at comparable validation performance or sparsity, but do not assert equal compute, capability gain, or update magnitude without measurement.

### Focused causal follow-up

For a small prespecified or clearly exploratory subset, remove the **new read changes** of a neuron or set in the sparse model, leaving the neuron's original base reads and writes intact. Compare with the intact sparse model and matched controls on permitted held-out inputs. This tests the contribution of those rewired changes under a specified intervention; it is not an ablation of the entire neuron and is not the same intervention as removing one term of a GRPO LoRA adapter's read. Track interactions and avoid assuming single-neuron effects add linearly.

## 8. Engineering and scientific qualification

In the later spec, define tests and measurable qualification criteria using the existing repository's TDD and runtime rules; do not invent numerical tolerances here. At minimum cover:

- **Source identity and data:** exact model/tokenizer revision, data IDs/hashes, prompt bytes, response labels, end-turn and padding; no overlap with protected sets according to available lineage; no silent truncation or missing labels. Verify batch loss equals an independent calculation, including variable answer lengths and accumulation denominator.
- **Zero and frozen invariants:** zero-coefficient initialization and explicit rewiring-off agree with untuned model within a justified precision envelope; all original weights remain frozen and unchanged; only the intended read factors and mask parameters train. Confirm Qwen2.5 module mapping and zero-based layer/neuron indexing.
- **Mask/gradient semantics:** gradients reach shared directions, coefficients and gate logits as expected over the training trajectory, accounting for zero-initialized coefficients; `z_det`, expected count, final binary mask and exported selected indices agree with independent checks. Confirm that gate randomness and gradient-checkpoint recomputation are consistent with the declared RNG policy. Validate pre-/post-freeze behavior, and exactly what an intervention disables.
- **Precision and memory:** FP32 is not sufficient evidence that TF32 is off. Check actual forward operations, matmul flags, attention/RMSNorm behavior and dtype casts. Use realistic longest-input, forward/backward, gradient accumulation and serialization probes. For small-effect attribution or removal validation, measure the FP32 floor and, if needed, use the project's genuinely FP64 reference; `model.double()` alone is not necessarily a full-FP64 Qwen2 path. Do not import unrelated numerical thresholds.
- **Artifact integrity and reproducibility:** save unmerged read factors and all gate state, full optimizer/scheduler/RNG/data-order state for resumption, plus immutable run config, package/runtime, data/cohort hashes, per-checkpoint losses and sample IDs. A weights-only file is not a resumable trainer checkpoint. Verify a clean restore and identical expected mask/checkpoint on a small uninterrupted-versus-resumed fixture or qualification run. Preserve existing seals and outputs.
- **Reuse and execution:** inspect the existing code before reimplementing scorer, batches, prompt construction, evaluation and cohort registration. Exercise real export/restore paths if they will be used. Follow applicable experiment lessons and measure GPU time and memory; CPU success does not prove GPU qualification or experiment completion.

The final spec should distinguish: CPU test success, GPU qualification, training completion, evaluation completion and scientific conclusions. A project-specific runbook or actual latest launch record may add constraints; those take precedence over assumptions here.

## 9. Questions to bring back with the filled design

1. Which exact training IDs and evaluation/attribution cohort IDs are shared? What are the current prior-use and freeze restrictions, and which lineage is unknown?
2. What did gold SFT actually use for loss mask, ending token, reduction, sequence length, training seed, optimizer, LR, schedule, dtype and batching? Which values differ in Max's trainer and why?
3. Which outputs does the ongoing attribution code actually produce: full per-layer neuron rankings, only selected-layer adapter-read terms, or something else? Under which metric, inputs and token positions?
4. What is the smallest GPU, memory and run-time budget for the first proposed rank-1 run? What does an additional seed or rank-4 run cost after measurement?
5. Which choices require a researcher decision before converting this design into the launchable spec: LR, mask LR, lambda and calibration budget, total steps, freeze schedule, validation criterion, evaluation permissions, and rank/seed follow-up priority?
6. Is Max's rank-1 experiment accompanied by saved, verifiable results in a source not yet checked? If so, link exact run IDs, settings and measured outcomes; otherwise state unavailable.
7. Which useful code can be reused under its license, and which pieces require adaptation? Identify actual functions and integration seams, including scorer and artifact recovery.

**Handoff deliverables:** (a) this design with values and citations filled in and unresolved points visible; (b) a concise decision log comparing viable options and the recommended minimal pilot; then (c) a separate draft detailed spec and ticket breakdown, with unresolved decisions and their consequences visible. Do not begin implementation or launch solely because this design exists.

# Supplied Pilot 5 source specification

Verbatim paragraph text extracted from the supplied DOCX; table rows appear as sequential cells.

Module 8 Pilot Spec: Readable Traces of Rank-1 SFT vs GRPO

Oct 14, 2026 · @Prasad Mahadik

Purpose

This pilot asks one question: does a rank-1 GRPO adapter leave a trace on unrelated text that a reader can decode into its training task, as well as a matched rank-1 SFT adapter does? It is the first slice of project module 8 ("Do RL writes leave readable traces?"), on the Pilot 2 SFT and Pilot 4 GRPO adapters we already have.

The method follows Minder et al. (ICLR 2026): run the same unrelated text through the untuned model and the adapted model, take the per-token activation difference, average it, and read the average with lenses. If the average decodes to GSM8K-like content (digits, ####, arithmetic words) clearly above a norm-matched random adapter, the trace is readable.

In scope: activation differences at every layer, per token position, for 3 adapters × 5 checkpoints; two readouts (the logit lens and a Jacobian lens); norm-matched comparison; blinded grading.

Out of scope for now: steering, Patchscope, SAE readouts, the interpretability agent, new training runs, more seeds.

Outcomes that count:

Readable for both, at matched norm: RL writes are as auditable as SFT writes at this budget.

Readable for SFT only at matched norm: RL is quieter. This is the result auditors need to know.

Readable for neither above the random floor: 64 steps of rank 1 is below the detection threshold of these lenses. This is a reportable null, with the threshold stated.

What already exists

We have the adapters and a working paired-activation pipeline, but not the activations this method needs. The saved FineWeb results are one mean vector per layer, averaged over all positions of 150 chat-wrapped passages. Minder uses per-position means over 10,000 raw passages; this pilot uses 200.

Item

Status

Where

Usable for this pilot?

11 adapters: GRPO 0/8/16/32/64, SFT 0/8/16/32/64, random-64

Exists, SHA-256 verified

Drive (run folders); copies in deep-dive

Yes, as is

Paired activation pipeline (pilot_eval): base and adapter on identical token ids, FP32, eager attention, all 28 block outputs, sufficient sums

Exists, validated on A100

spar/pilot1/pilot_eval

Yes, with two changes: per-position sums and a raw-text input mode

aggregate-sums.npz per arm: sum of Δh and of ‖Δh‖ per layer, per view

Exists

Drive, Downloads

Partly: gives a token-averaged mean δ per layer on 150 FineWeb passages; enough for a first logit-lens look (stage 0)

Per-position mean δ (positions 0–4 and later)

Missing

—

Needed: Minder finds the trace strongest at the first positions

Raw-text sample (200 sequences for this pilot)

Missing

—

Needed: the saved FineWeb set is 150 sequences inside a chat user turn

Per-token activations

Not stored, by design

—

Not needed: means and paired norms are sufficient

Jacobian lens for Qwen2.5-1.5B-Instruct

Community fit exists (16 wikitext prompts)

qbz506/prabodha-lenses

As a cross-check only; fit our own with jacobian-lens

One useful fact makes the run cheap: the mean difference is linear. The mean of (h_adapted − h_base) equals mean(h_adapted) − mean(h_base). So the base model runs once, and each adapter runs once. Pairing within a batch is still needed for the raw shift ‖δ‖ and for per-example bootstrap, and the pipeline already does it with disable_adapter().

Design

One base model, nine adapted conditions and one control, one fixed corpus of 200 raw web sequences, all 28 layers, all 128 positions. Layer 14 and positions 0–4 are the pre-registered primary cell; everything else is reported as secondary.

Factor

Choice

Reason

Model

Qwen/Qwen2.5-1.5B-Instruct @ 989aa79

Same model and revision as Pilots 2–4

Conditions

GRPO steps 8/16/32/64; SFT steps 8/16/32/64; random-64; step-0 adapter as a zero check

Detectability over training, both objectives, one floor

Corpus

FineWeb sample-10BT @ 9bb295d, 200 sequences: the 150 Pilot 4 passages plus 50 new ones

Minder uses 10,000; 200 is enough to see whether a trace exists. Reusing the 150 links the result to Pilot 4

Sequence form

Raw text, no chat template, truncated to exactly 128 tokens; shorter documents dropped

Fixed length keeps every position filled by all 200 sequences

Second form (secondary)

The same 200 texts inside a chat user turn, as in Pilot 4; and a third form with the text as the assistant turn after one fixed neutral user prompt ("Continue the text.")

User turn links to the existing FineWeb results; assistant turn is the context training changed, with content still unrelated

Layers

All 28 block outputs plus the embedding output

Readouts differ by depth; layer 14 = ⌊28/2⌋ is primary

Positions

Per-position means for 0–127; primary = 0–4 separately and their mean; also 5–127 pooled

Minder finds early positions most informative

Precision

FP32, eager attention, TF32 off, deterministic

GRPO's early-layer shift is 0.2–0.5% of ‖h‖; bf16 rounding (≈0.4%) is as large as the signal

Two details differ from Minder and must be stated in the write-up. First, Qwen2.5 adds no BOS token, so our position 0 is a real text token, while Minder's position 0 is BOS. Second, our models are chat models trained on chat-formatted GSM8K, so raw web text is out of format as well as out of domain. Minder's emergent-misalignment and taboo organisms were also chat-trained and still readable. But GRPO's changes may depend more on context than SFT's (its layer-27 effect is already domain-gated), so a format gap could hide one arm more than the other. The three input forms in the table separate a format gap from a missing trace: compare arms within each form, and test the arm × form interaction.

Input builder change. The Pilot 3/4 preparation puts each FineWeb passage inside the chat template as a user turn, and its mask counts only the 128 content tokens, not the template tokens. The raw mode needs a new builder in pilot_eval:

tokenize the text with no chat template and no added special tokens;

truncate to exactly 128 tokens and drop shorter documents;

count every token in the mask, and index positions 0–127 from the first text token;

keep the existing chat-wrapped builder for the secondary condition, and store a content-relative position index there too, so position j means the same token in both modes.

The Minder paper says it reads pretraining data but does not state whether a template is applied. Check the diffing-toolkit code before the run, and if it differs, follow it.

Measurement

Each batch runs twice on identical token ids: once with the adapter disabled, once enabled. We keep float64 running sums at all 29 layers, and per-token tensors at 6 layers only (table). The core quantity is the mean difference per layer ℓ and position j:

pipeline · capture, two lenses, graded against controls

The mean difference δ̄ is the one object every readout reads; the controls pass through the same readouts and grading.

\bar{\delta}_{\ell,j} = \frac{1}{N}\sum_{n=1}^{N}\left(h^{\mathrm{ft}}_{\ell,j}(x_n) - h^{\mathrm{base}}_{\ell,j}(x_n)\right)

Saved per condition, as one .npz with a JSON manifest:

Array

Shape

Use

delta_sum

29 × 128 × 1536

δ̄ for every layer and position (≈45 MB in float64)

delta_norm_sum

29 × 128

Raw shift E‖δ‖ and coherence ‖δ̄‖ / E‖δ‖

base_sum, base_norm_sum (base run only)

29 × 128 × 1536; 29 × 128

Relative units; the mean-activation baseline readout

delta_example per token, at layers 14 and 17–21, all 128 positions

200 × 6 × 128 × 1536 in float16 (≈0.5 GB per condition and input form)

Bootstrap, sample-size curve, and later questions without a rerun

In short: every sample and every token is stored, but only at 6 layers (14, 17, 18, 19, 20, 21). The other 23 layers keep exact sums only. Storing all 29 layers would cost about 25 GB per input form and add little, because the lenses read means. If a later question needs another layer, rerunning it takes minutes.

Where it is written. Follow the Pilot 3/4 layout and its resumable, sealed-shard pattern:

runs/pilot-5-readable-traces/Qwen--Qwen2.5-1.5B-Instruct/<form>/<condition>/    results/sums.npz            # all 29 layers, float64; small    results/tokens/shard-XXXX.npz  # per-token delta (and h_base for the base run)    complete.json               # SHA-256 of every shard + manifest

Sums and manifests go to Drive at once. They are small, and everything after stage 2 runs on CPU from them. A Colab or RunPod disk vanishes when the session ends.

Per-token shards stay on the GPU machine's local disk first. Run the bootstrap draws in the same session, and save their outputs to Drive.

Copy per-token shards to Drive only if space allows. The raw form is about 5 GB; all three forms are about 15 GB. If they are lost, a rerun of stage 2 takes under 30 minutes and must reproduce the sums exactly.

The manifest records the model revision, adapter SHA-256, corpus revision and indices, token-id hash, positions, layers, hook names and seed, following the project artifact rule.

Two derived numbers go into every table, so readers can judge each readout against the size of the signal: the relative mean shift ‖δ̄‖ / E‖h_base‖ and the coherence. They extend the Pilot 4 FineWeb numbers, where GRPO's mean shift at layer 14 was 0.57% and SFT's 1.22%, against 0.03% for random.

Readouts

Each readout turns one vector δ̄ into a ranked token list. The logit lens and the Jacobian lens are linear in δ̄, so their top-k lists do not change when δ̄ is scaled.

Logit lens (primary, all layers)

Minder applies the final RMSNorm and the unembedding to δ̄. RMSNorm divides by one scalar, so the token ranking equals the ranking of the logits

\ell(\bar{\delta}) = W_U\,(\gamma \odot \bar{\delta})

where γ is the final-norm gain. Report the top 20 and bottom 20 tokens. Also report a variant with the components along the top 8 outlier channels (609, 1421, 503, 909, …) removed, because those channels dominate ‖δ̄‖ at late layers and may carry little token content.

Jacobian lens (primary at layers 8–20)

The lens is lens_ℓ(h) = unembed(J_ℓ h), with J_ℓ = E[∂h_final / ∂h_ℓ] over generic text. It is linear, so we apply it to δ̄ directly. It should read middle layers better than the logit lens, which is weak at half depth.

Use the community lens for Qwen2.5-1.5B-Instruct (prabodha-lenses, file qwen2.5-1.5b-instruct.pt, fitted on 16 wikitext prompts). We do not fit our own in this pilot.

Check it before use: it was made with a different tool, so confirm that jlens loads it, that it covers all 28 layers, and that its config names the same model revision as ours (989aa79).

Acceptance test (from the project doc): on a "think of an item, then name it" prompt, the lens shows the answer before the model emits it. Also, on the GSM8K view, it should rank digit tokens at least as high as the logit lens does at layers 8–20.

Fallback only if a check fails: fit our own with jlens.fit on about 100 FineWeb texts (about 1 A100-hour).

Optional: Patchscope

Minder reports Patchscope stronger than the logit lens (20–80% relevant tokens at the best position). It is cheap once δ̄ exists. It needs a scale λ, so it is the one readout where the norm question returns (see Controls). It is not steering, so it fits inside the current scope if time allows.

Controls and grading

The key control is not norm matching. Lens rankings ignore scale, so what makes a weak δ̄ unreadable is sampling noise. We therefore compare arms at matched signal-to-noise, by sample-size curves.

Controls

Control

What it rules out

How

Random adapter, norm-matched to GRPO-64

"Any rank-1 write of this size reads as math"

Same pipeline; its relevance score is the floor

Random directions

Lens bias toward certain tokens for any vector

200 Gaussian vectors scaled to ‖δ̄‖ per layer; gives a null distribution of the relevance score

Mean activations themselves (Minder's baseline)

Domain tokens appearing because mean activations read as generic text

Same readouts on mean h_base and mean h_ft

Step-0 adapter

Pipeline error

δ̄ must be exactly 0

Sample-size curve

SFT looking more readable only because its δ̄ is larger relative to noise

Relevance at N = 25, 50, 100, 200 (bootstrap draws from delta_example); compare arms at equal SNR = ‖δ̄‖ / standard error

Checkpoints 8/16/32/64

—

Gives when each arm becomes readable

Relevance score (automatic)

Build the reference token set once, before any readout: the 200 tokens with the highest log-odds of appearing in GSM8K training solutions versus FineWeb. Score = fraction of a readout's top 20 tokens inside that set. Report it against the random-direction null as a percentile.

Blinded grading

Pool the top-20 lists from every arm, checkpoint, layer and readout, plus the controls. Shuffle them and strip labels. A grader (an LLM, plus one human on a 10% sample for agreement) answers two questions per list:

Relevance: what fraction of these tokens relate to grade-school math word problems with a final #### answer?

Naming: from a list of 8 candidate training tasks (GSM8K math, code, French, medical QA, …), which one produced this list?

Unblind only after all lists are graded.

Predictions to register

These are draft predictions, each grounded in a Pilot 4 number. Replace the confidences with your own, and freeze them in the project prediction registry before stage 2 runs.

#

Prediction (layer 14, positions 0–4, step 64 unless stated)

Basis in Pilot 4

Draft confidence

P1

SFT relevance is above the random-direction 95th percentile in at least one lens

SFT FineWeb mean shift 45× random at layer 14

0.75

P2

GRPO relevance is above the same threshold

GRPO 21× random; its question and FineWeb shifts point the same way (cos 0.55–0.83)

0.6

P3

At N = 200, SFT relevance > GRPO relevance

SFT shift about 2× GRPO on FineWeb at every layer

0.65

P4

At matched SNR the SFT–GRPO gap shrinks by half or more

Lens rankings ignore scale; GRPO's early shift is as input-independent as SFT's

0.5

P5

Format tokens (####, digits, "answer", "total") appear in some top 20 for SFT; same for GRPO

Strict accuracy 5% to 65% (SFT), 5% to 50% (GRPO)

0.7 SFT, 0.5 GRPO

P6

The Jacobian lens gives higher relevance than the logit lens at layers 8–20

Logit lens is weak at half depth in most models

0.65

P7

Relevance rises with checkpoint, and GRPO becomes readable later than SFT

GRPO FineWeb shift at layer 14: 0.10% at step 8, 1.29% at step 64; SFT 0.18% and 2.63%

0.6

What counts against what: P2 false with P1 true supports "RL is quieter". P1 and P2 both false points to a detection threshold above 64 rank-1 steps. P4 false means SFT carries more task content per unit of shift, not just a larger shift.

Stages, acceptance tests and compute

Five stages. Each stage ends in a test that must pass before the next one starts. The total is under 1 A100-hour (a few dollars), since we use the community Jacobian lens.

Stage 0: first look on existing data (CPU, about 1 hour). Apply the logit lens to the token-averaged δ̄ in the three aggregate-sums.npz files (all layers, 5 checkpoints, FineWeb and GSM8K views). This needs only γ and lm_head from the model.

Test: the step-0 vectors are exactly zero; GSM8K-view lists for SFT contain digits or #### (a sanity check that the lens reads anything at all).

Stage 1: corpus and reference set (CPU). Take the 150 Pilot 4 FineWeb passages plus 50 new ones (200 in total). Tokenize to exactly 128 tokens, save token ids and their hash. Build the 200-token GSM8K relevance set.

Test: the 150 reused indices match Pilot 4; every sequence is 128 tokens.

Stage 2: activation capture (GPU, under 30 minutes). Extend pilot_eval with per-position sums and a raw-text mode. Run the base once, then each of the 10 adapter conditions, paired per batch. Store per-token δ at the 6 layers. Total storage about 5 GB per input form.

Tests: step-0 δ̄ = 0 exactly; the chat-wrapped subset reproduces the Pilot 4 FineWeb layer means for the overlapping setting to 1e-6 relative; two runs of one batch give identical sums; the PEFT identity ΔW x = (Aᵀx)B holds to FP32 precision.

Stage 3: lens checks and readouts (GPU, under 30 minutes). Load and check the community Jacobian lens. Then compute all readouts for every vector, including the controls.

Tests: the Jacobian-lens "name the item" check.

Stage 4: grading and statistics (CPU plus grader calls). Automatic relevance with nulls, the sample-size curves, then blinded grading and unblinding.

Output: one figure of relevance by layer per arm and lens, one table of the primary cell, the sample-size curves, and the registry entries closed with the outcome of each prediction.

Risks and open decisions

The largest risk is a null caused by method choices rather than by the adapters. The decisions below are the ones that can cause it.

Item

Risk or choice

Proposed default

Raw text or chat-wrapped

Our adapters were trained on chat-formatted data; raw text may show a weaker trace

Raw text primary (Minder's setting), chat-wrapped 200 as secondary

Position 0

No BOS in Qwen2.5, so position 0 is not Minder's position 0

Report positions 0–4 separately; do not pool across them for the primary claim

Outlier channels

They dominate ‖δ̄‖ at late layers and can fill the logit-lens top 20 with format-neutral tokens

Report each lens with and without the top-8 outlier components

Community Jacobian lens

Fit on 16 prompts only

Use it after the load, revision and "name the item" checks; fit our own only if they fail

One seed per arm

Any SFT–GRPO difference may be seed noise

State it as a limit; add seeds from B0 when they exist

Grader leakage

An LLM grader may infer the arm from list style

Shuffle, strip labels, include controls in the pool, measure human–LLM agreement

Patchscope in scope?

Stronger readout per Minder, but needs a scale λ

Decide at the team meeting; default is to add it after stage 4 if time allows

Sources: Minder et al., arXiv:2510.13900 · anthropics/jacobian-lens · prabodha-lenses

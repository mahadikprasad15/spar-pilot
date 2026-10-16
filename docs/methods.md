# Shared methods

## Model and numerical variants

The main training arms use Qwen/Qwen2.5-1.5B-Instruct, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. “Untuned” means this instruction-tuned
checkpoint with adapters disabled; it does not mean a pretrained-only model.

Pilot 1 includes unquantized BF16/FP16 evaluation variants. Later training and
internal measurements use unquantized FP32. BF16/FP16 are floating-point dtypes,
not 4-bit/8-bit quantization. TF32 is a separate GPU matrix-multiplication setting;
FP32 tensor storage alone does not establish that it is disabled. FP64 summary
accumulation does not imply FP64 model inference. Retained shard dtype is recorded
separately from both forward and accumulation precision.

Actual frozen runtime manifests govern comparisons: model/tokenizer files,
package versions, attention backend, precision, GPU, batch membership, padding,
position IDs and numerical settings. Different GPUs do not automatically invalidate
all results, but exact reproducibility and sensitive cross-arm measurements require
qualification under the comparison policy. Existing valid runs are preserved as
named variants.

## Rank-1 adapters

For a linear weight `W` of shape `[out, in]`, LoRA adds
`delta_W = (alpha/r) B A`, with `A: [r, in]`, `B: [out, r]`.
Here `r=1`, `alpha=1`, dropout is zero; all `q/k/v/o/gate/up/down` projections
in all 28 decoder blocks are targeted (196 modules). Trainable parameter count is
`sum(r * (in + out))` over actual targets, not the base-model parameter count.

At initialization B is zero, so the branch is zero even if A is nonzero.
Consequently the scalar coefficient `A x` need not be zero for a zero adapter.
The direct branch is `(alpha/r) B (A x)` on the actual module input `x`.

## Behavior and scoring

Keep prompt template, held-out IDs, decoding, cap and scorer with every result.
Greedy accuracy, average sampled accuracy and pass-at-k are different statistics.
Strict/flexible-v2/flexible-v3 are versioned extraction policies. Flexible-v3
uses a valid strict extraction first, otherwise its flexible fallback; capped
responses receive zero. It does not inspect gold to select a candidate. Post-hoc
rescoring uses saved responses and must be identified as such.

Paired comparisons resample item IDs shared across checkpoints/arms. A confidence
interval on held-out items does not measure variation across training seeds.
Non-inferiority requires its frozen margin and interval rule, not just a positive
accuracy change. See the scorer ADRs and experiment pages for exact protocols.

## Fixed-input internal measurements

For identical tokenized text, let `h_base` and `h_adapted` be outputs at the same
named site. The propagated change is `delta = h_adapted - h_base`. The local
adapter write uses the actual adapted-pass module input in the LoRA formula above;
it differs from the full propagated block change.

Pilot 3/4 primary relative quantities are:

| Object | Numerator | Denominator |
|---|---|---|
| Block change | `norm(mean(delta))` | `mean(norm(h_base))` |
| Direct module write | `mean(norm(direct_write))` | `mean(norm(ordinary_module_output))` |

An average of per-token ratios is a separate statistic. Mean-vector norm and
mean of vector norms differ because vectors can cancel. Zero-denominator cases
follow the saved schema; do not replace undefined values with arbitrary constants.

Token weighting pools all counted tokens. Equal-example weighting gives each
example equal weight after its within-example summary. GSM8K question/solution
and FineWeb views use explicit masks; padding and excluded special tokens do not
become observations. Whole examples/documents are bootstrap units.

## Coefficients and KL

The signed rank-1 coefficient is `A x`; its sign/scale depends on the saved factor
basis, so preserve factor identities. Pilot 4's accepted fixed-batch policy
reports singleton/batch sensitivity descriptively while retaining other gates.

The supplemental KL is full-vocabulary `D_KL(p_adapted || p_untuned)` in nats on
identical fixed contexts. Logits at `t-1` predict target token `t`; target masks
exclude position zero, padding and out-of-sequence positions. This is neither a
KL between generated answer strings nor the GRPO optimization penalty. Pilot 4
training uses beta zero.

## Readable traces

Pilot 5 averages adapted-minus-untuned vectors on raw FineWeb, then projects them
through qualified lenses. Primary block 14 is zero-based. Positions 0–4 form
one pooled vector per arm, not five searched primary tests. The positive top-20
normalized unique strings are compared with task-enriched reference lists.
Overlap is lexical evidence, not a semantic or causal guarantee.

Reference construction, normalization, candidate limits, nulls, test families,
resampling and unavailable analyses are recorded in the frozen procedure and
[automatic analysis guide](pilot5-automatic-analysis.md). A random adapter control
and random-direction null answer different questions. Preserve that distinction.

## Controls and interpretation

Zero adapters test instrument identity. Frozen-base hashes test weight integrity.
Negative controls test whether deliberately wrong sign/scale/mapping is detectable.
Numerical envelopes are engineering acceptance rules, not statistical intervals.
Behavior, internal geometry and interventions support different claims; descriptive
activation differences alone do not establish a mechanism for task performance.

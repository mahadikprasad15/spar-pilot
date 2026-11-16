# Attribution, its finite-difference cross-check, and the 06a r3 failure — explainer

Self-contained context for anyone (person or agent) picking this up. Status as of 2026-11-16: diagnosis in progress;
nothing in the pass rule has been changed. Code: `pilot_eval/attribution_instrument.py` (attribution),
`pilot_eval/attribution_stages.py` (`finite_difference_check`). Spec: `spec.md` §Stage A+B and §Tolerances.

## 1. What an adapter does

Each arm is the same base model plus 196 small add-ons ("rank-1 LoRA adapters"), one on each of 7 linear layers
in each of 28 transformer blocks. For an input vector x at one token, an adapter adds

    Δy = s · b · (a · x)

- `a · x` is a single number per token: the adapter's **read**, written c_t.
- `b` is a direction the adapter writes in, and `s` a fixed scale.

Removing an adapter means setting that add-on to zero. To make "removal" a smooth knob, put a gate g on it:
y = base(x) + g · Δy. g = 1 is the trained arm; g = 0 is the arm with that adapter removed.

## 2. The metrics

M is a number computed from the model's output on a fixed text (the untuned model's own correct answer):

- **Trace log-likelihood:** the average log-probability the arm assigns to each token of that answer.
- **Stop pull:** how strongly the arm wants to emit the end-of-answer token before the last line.

## 3. Attribution: the fast, first-order estimate

The question is: if adapter i were removed, how much would M change?

The exact answer needs a full model run with g = 0 for every adapter: 196 runs per arm per text, and the effect
is non-linear. Attribution instead uses the **slope** at g = 1. A straight-line (first-order) prediction of the
change from g = 1 to g = 0 is

    predicted change = −dM/dg (at g = 1).

Because Δy = s·b·c_t, that derivative equals Σ_t c_t · ∂M/∂c_t. So

    attribution_i = − Σ_t c_t · ∂M/∂c_t

One backward pass (automatic differentiation) gives ∂M/∂c_t for every adapter at once.

**Neuron attribution.** For the `down_proj` adapters the read is a sum over 8,960 neurons:
c_t = Σ_j a_j · z_{j,t}, where z_j is neuron j's activation. Removing only neuron j's term from the read has
first-order effect

    attribution_j = − a_j · Σ_t z_{j,t} · ∂M/∂c_t

Summed over all j this equals the adapter's attribution exactly; the code enforces that identity. Neuron effects
are therefore small pieces of an adapter's effect, often of order 1e-5.

## 4. The cross-check: finite differences

This is an independent way to get the same slope, with no gradients: actually turn the knob and measure.

    M+ = M with g = 1 + ε        M− = M with g = 1 − ε        slope ≈ (M+ − M−) / (2ε)

With ε = 0.01 the knob moves ±1%. The prediction to check is attribution ≈ −slope. For a neuron, the gate is put
on that neuron's term only.

If both agree, the gradient code is wired correctly: right layer, right sign, right tokens. The spec's proposed
qualification rule (`spec.md` §Tolerances) is median relative error ≤ 0.05 and maximum ≤ 0.20 over the checked
adapters and neurons. A failure stops the overnight chain before production.

## 5. Why a finite difference can be wrong even when attribution is right

Computers store numbers with limited precision. FP32 keeps about 7 significant digits, and every operation rounds.
M+ and M− each come from a full forward pass of a 28-block model whose output probabilities run over 151,936
tokens, with logits around 10–30 in size, so rounding happens throughout. Their difference therefore contains
rounding noise, and dividing by 2ε = 0.02 multiplies that noise by 50.

If the true change M+ − M− is not much larger than the noise, the finite difference is mostly noise. The analogy
is weighing a feather by weighing a truck with and without the feather on it. Relative error divides by the
(noisy, near-zero) finite difference, so it can become enormous even when attribution is exactly right.

## 6. What 06a r3 measured (gold-sft, ε = 0.01, FP32)

- **Overall:** 48 comparisons, median relative error 0.0078. Every adapter-level comparison and every stop-pull
  comparison agreed; none is among the worst 10.
- **Worst 10:** all are single neurons on the trace metric, with effects of about 1e-5 to 7e-5.
  - Worst (rel err 60.8): `|fd|/noise ≈ 0.7`. The finite difference was below one unit of the rounding estimate,
    i.e. pure noise.
  - Next (6.7, opposite sign): `|fd|/noise ≈ 8.4`, also noise-dominated.
  - Others: errors of 0.2–0.9 at `|fd|/noise` of 7–80.

`|fd|/noise` uses ulp(|M|) / (2ε), the rounding step of the *final* metric value (|M| ≈ 0.3). That is very
probably an underestimate, because rounding happens at every layer and at logits of size ~20, where the step is
about 100 times larger. So the rows at 40–80× may still be noise-dominated. That is unproven.

## 7. Two hypotheses

- **H1 (precision artifact):** attribution is right; an FP32 finite difference cannot resolve single-neuron effects
  this small.
- **H2 (real error):** the neuron-level attribution is wrong on the full-size model. That matters because primary 3
  (dense vs sparse) is built on neuron attribution.

## 8. The decisive test (06a diagnostic 2)

Recompute the same comparisons with the cross-check changed and the attribution unchanged (FP32, as in the run):

| Cross-check | Signal | Noise | Expected under H1 | Expected under H2 |
| --- | --- | --- | --- | --- |
| FP32, ε = 0.01 (as run) | 1× | 1× | large errors on tiny neurons | large errors |
| FP32, ε = 0.1 | 10× | 1× | neuron errors shrink about 10×, plus a small curvature term | errors stay |
| FP64, ε = 0.01 | 1× | ~1e-9× | neuron errors become tiny | errors stay |

Adapter-level rows act as a control: they already agree and should stay that way in every column.

## 9. What follows

- **H1 confirmed:** the instrument is fine and the cross-check method needs a decision by the user before the Pod
  run, without loosening the tolerance. Options:
  - run qualification's finite differences in FP64 (an FP64 copy of one arm needs ~2× its FP32 memory);
  - check neurons with a larger ε;
  - exclude comparisons whose finite difference is below a measured noise floor (and report how many).
- **H2 confirmed:** fix the neuron attribution and re-run 06a. Primary 3 cannot be trusted until then.

## 10. Results and decision (2026-11-16)

Diagnostic 2 on gold-sft (full table: `06a-r3-fd-diagnosis.md`):

| Cross-check | Max relative error | Adapters |
| --- | --- | --- |
| FP32, ε = 0.01 | 60.8 | ≤ 0.025 |
| FP32, ε = 0.1 | 0.151 | ≤ 0.001 |
| FP64, ε = 0.01 | 0.078 | ≤ 0.0005 |

This confirms H1: the attribution is correct, and the FP32 reference was noise-limited for single neurons. The
few-percent residual in FP64 sits on the smallest effects (~1e-5) and matches the FP32 attribution's own rounding.

**Decision (user):** the finite-difference reference runs in FP64 at ε = 0.01, with the tolerances unchanged.

| Part | Precision |
| --- | --- |
| Attribution (forward, two backwards, neuron pass) | FP32, FP64 accumulation of products |
| Finite-difference reference (weights, activations, gates, log-softmax, metric means) | FP64 — *corrected 2026-11-16:* Qwen2's attention softmax, RMSNorm and RoPE stayed FP32 until `full_fp64` (see `06a-r3-fd-diagnosis.md`, Correction) |
| All Stage C / Stage D measurements | FP32 |

The FP64 copy's peak memory is measured in qualification and must fit the memory fraction.

## 11. Resolution (diagnostic 4, 2026-11-16)

The earlier "FP64" reference still computed Qwen2's attention softmax, RMSNorm and RoPE in FP32. A truly FP64
reference (`full_fp64`) brings the ordinary FP32 attribution into agreement to within 5e-5 on both arms, including
the single-neuron effects that had failed. The instrument was right throughout; the cross-check needed a genuinely
FP64 model. Production keeps FP32 attribution with that FP64 reference, and the tolerances are unchanged. Numbers:
`06a-r3-fd-diagnosis.md`.

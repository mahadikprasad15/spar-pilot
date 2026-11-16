# 06a r3 — finite-difference failure and diagnosis (gold-sft)

Colab A100-SXM4-40GB, code `47ef1cc`; diagnostics run 2026-11-16 by the user. The attribution values are the FP32
instrument's and identical in every column; only the finite-difference reference changes. Relative error =
|attribution − FD| / |FD|. Calibration rows 0–3; trace metric; `down_proj` layer 16.

## Original check (FP32, ε = 0.01), against the proposed tolerances (median ≤ 0.05, max ≤ 0.20)

| Arm | Comparisons | Median | Max | Pass |
| --- | --- | --- | --- | --- |
| gold-sft | 48 | 0.0078 | 60.8 | no |
| online-grpo | 48 | 0.029 | 1.92 | no |

## Per-check results (gold-sft, trace)

| kind | target | row | attribution | FD fp32 ε=0.01 (rel) | FD fp32 ε=0.1 (rel) | FD fp64 ε=0.01 (rel) |
| --- | --- | --- | --- | --- | --- | --- |
| adapter | 118 | 0 | +6.346e-03 | +6.346e-03 (0.000) | +6.346e-03 (0.000) | +6.346e-03 (0.000) |
| adapter | 118 | 1 | +6.378e-03 | +6.337e-03 (0.006) | +6.375e-03 (0.000) | +6.378e-03 (0.000) |
| adapter | 118 | 2 | +2.168e-03 | +2.157e-03 (0.005) | +2.166e-03 (0.001) | +2.169e-03 (0.000) |
| adapter | 118 | 3 | +6.716e-03 | +6.699e-03 (0.003) | +6.716e-03 (0.000) | +6.717e-03 (0.000) |
| adapter | 111 | 0 | +4.285e-03 | +4.286e-03 (0.000) | +4.286e-03 (0.000) | +4.284e-03 (0.000) |
| adapter | 111 | 1 | +4.574e-03 | +4.540e-03 (0.007) | +4.574e-03 (0.000) | +4.574e-03 (0.000) |
| adapter | 111 | 2 | +2.028e-03 | +2.042e-03 (0.007) | +2.029e-03 (0.000) | +2.027e-03 (0.000) |
| adapter | 111 | 3 | +5.010e-03 | +5.020e-03 (0.002) | +5.011e-03 (0.000) | +5.009e-03 (0.000) |
| adapter | 97 | 0 | +4.094e-03 | +4.079e-03 (0.004) | +4.091e-03 (0.001) | +4.095e-03 (0.000) |
| adapter | 97 | 1 | +3.827e-03 | +3.792e-03 (0.009) | +3.830e-03 (0.001) | +3.826e-03 (0.000) |
| adapter | 97 | 2 | +9.621e-04 | +9.868e-04 (0.025) | +9.620e-04 (0.000) | +9.623e-04 (0.000) |
| adapter | 97 | 3 | +4.043e-03 | +4.010e-03 (0.008) | +4.048e-03 (0.001) | +4.043e-03 (0.000) |
| neuron | 807 | 0 | +6.660e-05 | +3.498e-05 (0.904) | +6.764e-05 (0.015) | +6.599e-05 (0.009) |
| neuron | 807 | 1 | +4.036e-05 | +2.857e-05 (0.413) | +3.969e-05 (0.017) | +3.973e-05 (0.016) |
| neuron | 807 | 2 | +9.765e-06 | +5.398e-06 (0.809) | +1.079e-05 (0.095) | +9.062e-06 (0.078) |
| neuron | 807 | 3 | +7.074e-05 | +5.860e-05 (0.207) | +7.162e-05 (0.012) | +6.961e-05 (0.016) |
| neuron | 2214 | 0 | +4.025e-05 | +3.669e-05 (0.097) | +4.094e-05 (0.017) | +4.128e-05 (0.025) |
| neuron | 2214 | 1 | +3.675e-05 | +6.630e-05 (0.446) | +3.957e-05 (0.071) | +3.672e-05 (0.001) |
| neuron | 2214 | 2 | +2.410e-05 | +1.854e-05 (0.300) | +2.712e-05 (0.111) | +2.478e-05 (0.028) |
| neuron | 2214 | 3 | +4.961e-05 | +3.016e-05 (0.645) | +4.677e-05 (0.061) | +5.009e-05 (0.010) |
| neuron | 7453 | 0 | +3.323e-05 | +5.375e-07 (60.825) | +3.334e-05 (0.003) | +3.305e-05 (0.005) |
| neuron | 7453 | 1 | +4.223e-05 | +6.729e-05 (0.372) | +3.669e-05 (0.151) | +4.131e-05 (0.022) |
| neuron | 7453 | 2 | +1.108e-05 | +1.125e-05 (0.015) | +1.173e-05 (0.055) | +1.202e-05 (0.078) |
| neuron | 7453 | 3 | +3.570e-05 | −6.275e-06 (6.690) | +3.164e-05 (0.128) | +3.573e-05 (0.001) |

Summary (gold-sft trace, 24 comparisons): FP64 reference max 0.078; FP32 at ε = 0.1 max 0.151; FP32 at ε = 0.01 max 60.8.

## Conclusion

- **The attribution is correct.** Adapter-level agreement is ≤ 0.0005 in FP64.
- **The FP32 failure was noise in the reference.** A 32-bit central difference cannot resolve single-neuron effects of ~1e-5. With an FP64 reference the neuron errors fall to 0.001–0.078.
- **The remaining few percent** sit on the smallest effects. They match the attribution's own FP32 rounding (absolute ~1e-6), which a CPU check also shows: with an FP64 reference only effects of ~1e-7 disagree.
- **Decision (user, 2026-11-16):** FP64 finite-difference reference, ε = 0.01, tolerances unchanged.

## online-grpo (diagnostic 3, code `038c4da`): the Pod's own check, FP32 vs FP64 reference

| Reference | Median | Max | Passes | Peak GPU memory (A100-40GB) |
| --- | --- | --- | --- | --- |
| FP32, ε = 0.01 | 0.0291 | 1.925 | no | 15.0 GB |
| FP64, ε = 0.01 | 0.0021 | **0.926** | **no** | 27.4 GB |

Parts: attribution FP32 (instrument, FP64 accumulation of products); finite difference FP64 throughout.

Comparisons that exceed 0.20 with the FP64 reference (all trace metric, single neurons, `down_proj` layer 14):

| neuron | row | attribution (FP32) | FD (FP64) | abs error | rel error |
| --- | --- | --- | --- | --- | --- |
| 8178 | 3118 | +6.423e-08 | +8.700e-07 | 8.1e-07 | 0.926 |
| 5801 | 3118 | −5.944e-07 | −1.213e-06 | 6.2e-07 | 0.510 |
| 8178 | 3975 | +5.311e-06 | +3.875e-06 | 1.4e-06 | 0.371 |
| 5801 | 3975 | +4.220e-06 | +3.220e-06 | 1.0e-06 | 0.311 |
| 6622 | 3118 | −3.352e-06 | −2.738e-06 | 6.1e-07 | 0.224 |
| 6622 | 3975 | +1.050e-05 | +1.316e-05 | 2.7e-06 | 0.202 |

Every other comparison is ≤ 0.09: every adapter (trace and stop), and every stop-pull neuron (effects ~1e-3,
≤ 0.066). The adapter-level trace comparisons show the same absolute floor: attribution +9.773e-04 vs FP64
+9.754e-04 (1.9e-6); −2.037e-04 vs −2.029e-04 (8e-7).

**Reading.** The FP64 reference is effectively the exact slope; both models have identical weights (FP32 cast
exactly to FP64). What remains is the FP32 attribution's own rounding: an absolute error of about 1e-6 to 3e-6 per
row on the trace metric. That is invisible on adapters (~1e-3) and on stop-pull neurons (~1e-3), but it is the same
size as online-grpo's trace neuron effects (~1e-7 to 1e-5), so their relative error is large. gold-sft passed only
because its checked trace neurons are larger (~1e-5 to 7e-5).

**Status.** The FP64 reference alone does not make the gate pass for online-grpo. On the Pod, qualification would
stop before production. A decision is needed (see the 06a ticket).

## Still to record
- ~~online-grpo with the FP64 reference~~ (done above).
- **Residual risk (now observed on online-grpo).** The gate is relative. A checked single-neuron effect of ~1e-6 on the Pod's calibration rows could still exceed 0.20 because of the attribution's own FP32 rounding. This is a measured limit, not a reason to loosen the tolerance. If it occurs, qualification stops and the decision returns to the user.

## Correction (2026-11-16): the "FP64" reference was not fully FP64

transformers 4.57's Qwen2 casts to FP32 inside **the eager attention softmax** (`modeling_qwen2.py:114`), **every
RMSNorm** (`:198`) and **the rotary embedding** (`:296–301`), even when the model is converted to FP64. The
"FP64 reference" in diagnostics 2 and 3 therefore still carried FP32 rounding from every block. The statement that
it was FP64 "throughout (weights, activations, gates, log-softmax, metric means)" was wrong.

The conclusion that the residual errors are the FP32 attribution's own rounding floor is therefore **not
established**. On the CPU fixture, a truly FP64 reference (`attribution_instrument.full_fp64`, which computes those
three steps in FP64) brings even the ordinary FP32 attribution to agreement within 1.1e-4. The same comparisons had
disagreed by up to 13× against the partly-FP32 reference. Diagnostic 4 repeats this on the real model.

## Diagnostic 4 (code `419f9a3`, Colab A100-40GB): resolved

Same 4 calibration rows, ε = 0.01, 48 comparisons per arm (3 adapters + 3 neurons × 2 metrics × 4 rows).

| Arm | Check | Median | Max | Pass | Peak GPU | Time |
| --- | --- | --- | --- | --- | --- | --- |
| online-grpo | A: FP32 attribution vs old partial-FP64 reference | 0.00207 | 0.926 | no | 27.4 GB | 13 s |
| online-grpo | **B: FP32 attribution vs full-FP64 reference** | 0.00000 | **0.00005** | **yes** | 27.4 GB | 13 s |
| online-grpo | C: FP64 attribution vs full-FP64 reference | 0.00000 | 0.00000 | yes | 36.1 GB | 13 s |
| gold-sft | A | 0.00051 | 0.078 | yes | 27.4 GB | 13 s |
| gold-sft | **B** | 0.00000 | **0.00001** | **yes** | 27.4 GB | 13 s |
| gold-sft | C | 0.00000 | 0.00000 | yes | 36.1 GB | 13 s |

Attribution pass on 4 rows: FP32 27.4 GB / 1.6 s; FP64 36.1 GB / 1.8 s. Peaks include the resident FP32 arm, the
untuned model and, where used, the FP64 copy.

Under B the previously failing comparisons agree exactly to the printed precision. For example, online-grpo
neuron 8178: attribution +6.423e-08, FD +6.423e-08; neuron 5801: −5.944e-07 vs −5.944e-07.

**Conclusion.**
- **The FP32 attribution is correct**, to ≤ 5e-5 relative, including effects down to ~1e-7.
- The whole of the 06a r3 failure came from the finite-difference reference:
  - first from FP32 central-difference noise;
  - then from Qwen2's FP32 softmax, RMSNorm and RoPE inside the nominally FP64 copy.
- With a truly FP64 reference (`full_fp64`) the gate passes with a large margin under the unchanged tolerances.
- **FP64 attribution (option A) is not needed.** Production attribution stays FP32 as specified. The FP64 instrument mode stays available for diagnostics only.

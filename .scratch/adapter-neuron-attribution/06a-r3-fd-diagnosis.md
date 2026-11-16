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

## Still to record

- online-grpo with the FP64 reference (diagnostic 3).
- **Residual risk.** The gate is relative. A checked single-neuron effect of ~1e-6 on the Pod's calibration rows could still exceed 0.20 because of the attribution's own FP32 rounding. This is a measured limit, not a reason to loosen the tolerance. If it occurs, qualification stops and the decision returns to the user.

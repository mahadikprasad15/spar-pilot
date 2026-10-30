"""ADR 0015: FP32 coefficient reconstruction allowance, CPU oracles only."""
import inspect

import numpy as np

from pilot_eval.stage0_drift_math import coefficient_fractions


# Observed real-CUDA case: layer 27 down_proj, gsm8k:test:65, V1.
DOT = -0.0020182250689185777
ERROR = 1.0250757850327474e-06
SUM_ABS = 21.03020810990609


def test_sub_ulp_cancellation_passes_the_fp32_dot_product_bound_but_not_the_legacy_floor():
    fraction, legacy = coefficient_fractions(np.array([DOT]), np.array([DOT + ERROR]),
                                             np.array([SUM_ABS]), 8960)
    assert legacy > 1
    assert fraction < 1e-3


def test_wrong_tensor_still_fails():
    # A wrong capture errs on the scale of the summed terms, not of FP32 roundoff.
    wrong = DOT + 0.1 * SUM_ABS
    fraction, legacy = coefficient_fractions(np.array([DOT]), np.array([wrong]), np.array([SUM_ABS]), 8960)
    assert fraction > 1 and legacy > 1


def test_legacy_tolerance_still_admits_what_it_admitted():
    dot = np.array([1.0, -3.0]); actual = dot + np.array([9e-6, -2e-5])
    fraction, legacy = coefficient_fractions(dot, actual, np.array([1e-9, 1e-9]), 1536)
    assert legacy <= 1 and fraction == legacy


def test_bound_is_the_a_priori_gamma_d_bound():
    u = 2.0 ** -24; d = 1536
    gamma = d * u / (1 - d * u)
    error = 0.99 * gamma * 5.0
    fraction, _ = coefficient_fractions(np.array([0.0]), np.array([error]), np.array([5.0]), d)
    assert 0.98 < fraction < 1
    fraction, _ = coefficient_fractions(np.array([0.0]), np.array([1.02 * gamma * 5.0]), np.array([5.0]), d)
    assert fraction > 1


def test_backend_uses_the_shared_bound_and_keeps_the_branch_gate():
    from pilot_eval import stage0_drift_backend as backend
    source = inspect.getsource(backend.DriftEngine._module_write)
    assert 'coefficient_fractions(' in source
    assert '1e-6+1e-5*expected.abs()' in source

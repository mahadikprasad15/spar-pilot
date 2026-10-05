"""Behavioral tests at the agreed summary/engine/workflow interfaces."""

import math

import numpy as np
import pytest


def test_summary_contract_preserves_cancellation_weighting_and_undefined_ratios():
    from pilot_eval.activation_math import block_summary, module_summary, derive_measurements
    # Example one: two opposite writes; example two: one write orthogonal to them.
    base = np.array([[[3., 0.], [0., 4.]], [[0., 2.], [999., 999.]]], dtype=np.float32)
    delta = np.array([[[1., 0.], [-1., 0.]], [[0., 2.], [999., 999.]]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]])
    summary = block_summary(base, delta, mask)
    token = derive_measurements(summary, 'token')
    equal = derive_measurements(summary, 'example')
    assert token['mean_delta'] == pytest.approx([0, 2 / 3])
    assert token['mean_base_norm'] == pytest.approx(3)
    assert token['relative_write'] == pytest.approx(2 / 9)
    assert token['mean_delta_norm'] == pytest.approx(4 / 3)
    assert equal['mean_delta'] == pytest.approx([0, 1])
    assert equal['relative_write'] == pytest.approx(1 / 2.75)
    # Individual zero denominator stays in the ratio of sums, not per-token ratios.
    ordinary = np.array([[[0.], [4.]], [[2.], [999.]]], dtype=np.float32)
    direct = np.array([[[2.], [2.]], [[2.], [999.]]], dtype=np.float32)
    local = module_summary(ordinary, direct, mask)
    assert derive_measurements(local, 'token')['relative_write'] == pytest.approx(1)
    assert derive_measurements(local, 'token')['mean_token_ratio'] == pytest.approx(.75)
    assert derive_measurements(local, 'token')['defined_token_count'] == 2
    assert derive_measurements(local, 'example')['mean_token_ratio'] == pytest.approx(.75)
    zero = module_summary(np.zeros_like(ordinary), direct, mask)
    assert derive_measurements(zero, 'token')['relative_write'] is None
    assert derive_measurements(zero, 'token')['mean_token_ratio'] is None
    assert derive_measurements(zero, 'token')['undefined_token_count'] == 3

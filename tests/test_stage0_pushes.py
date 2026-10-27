"""Initial pushes: independent dense oracles and frozen public CPU/GPU seam."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_initial_rank_one_gradient_and_finite_difference_oracles():
    from pilot_eval.stage0_push_math import instrument_check
    result=instrument_check()
    assert result['passed'] and result['a_gradient_exact_zero']
    assert result['gradient_relative_error']<1e-5
    assert result['directional_relative_error']<1e-3 and result['epsilon']==1e-4
    assert result['microbatch_global_denominator_passed'] and result['dead_group_zero_passed']


def test_push_statistics_partial_batches_and_gauge_orientation():
    from pilot_eval.stage0_push_math import summarize_vectors,orient_endpoint
    gradients=np.array([[-1.,0.],[-1.,0.],[9.,0.]])
    result=summarize_vectors(gradients,[2,4,1],[8,8,2])
    np.testing.assert_allclose(result['mean_push'],[-7/3,0])
    np.testing.assert_allclose(result['mean_push_full_batches'],[1,0])
    np.testing.assert_allclose(result['token_weighted_push'],[-3/7,0])
    assert result['consistency']==1. and result['partial_batches']==1
    endpoint,proof=orient_endpoint(np.array([-2.,0.]),np.array([-3.,1.]),np.array([1.,0.]))
    np.testing.assert_array_equal(endpoint,[3.,-1.]);assert proof['flipped']
    assert orient_endpoint(np.array([0.,1.]),np.array([3.,1.]),np.array([1.,0.]))[0] is None
    zero=summarize_vectors(np.zeros((2,2)),[1,1],[8,8])
    assert zero['consistency'] is None

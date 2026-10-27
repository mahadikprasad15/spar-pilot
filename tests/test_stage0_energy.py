"""Base-input energy instrument: independent arithmetic and public saved workflow."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_energy_components_preserve_cancellation_and_distinct_denominators():
    from pilot_eval.stage0_energy_math import energy_sums,ratios
    x=np.array([[1.,-1.],[2.,-2.]])
    a=np.array([1.,1.]);parallel=np.array([1.,0.])
    result=energy_sums(x,a,parallel)
    np.testing.assert_array_equal(result,[5.,5.,-10.,0.])
    r=ratios(result)
    assert r['principal_fraction']==.5
    assert r['parallel_over_total'] is None and r['cross_over_total'] is None
    zero=ratios(np.zeros(4));assert zero['principal_fraction'] is None
    reinforce=energy_sums(np.array([[1.,1.],[2.,2.]]),a,parallel)
    np.testing.assert_array_equal(reinforce,[5.,5.,10.,20.])
    r=ratios(reinforce)
    assert r['principal_fraction']==.5 and r['cross_over_total']==.5
    assert r['parallel_over_total']+r['rest_over_total']+r['cross_over_total']==1.


def test_energy_projection_is_subspace_invariant_and_uses_unshifted_masks():
    from pilot_eval.stage0_energy_math import parallel_factors,view_sums
    q=np.array([[1.,0.],[0.,1.],[0.,0.]])
    rotation=np.array([[0.,-1.],[1.,0.]])
    a=np.array([[1.,2.,3.],[-1.,0.,1.]])
    np.testing.assert_allclose(parallel_factors(a,q),parallel_factors(a,q@rotation))
    x=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
    row=dict(id='q',input_ids=[1,2,3],attention_mask=[1]*3,masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3))
    result=view_sums(x,a,parallel_factors(a,q)[:,None,:],row)
    # factor × cutoff × view × component; the final solution token is counted.
    assert result.shape==(2,1,3,4)
    assert result[0,0,1,3]==(x[2]@a[0])**2
    assert result[0,0,0,3]==(x[1]@a[0])**2
    assert not np.any(result[:,:,2])

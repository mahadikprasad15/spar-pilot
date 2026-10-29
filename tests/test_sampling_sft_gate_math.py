"""Independent arithmetic for equal-problem NLL and whole-problem pairing."""
import numpy as np
import pytest

from pilot_eval.sampling_sft_gate_math import paired_bootstrap,summarize_rows


def test_equal_problem_mean_keeps_stop_token_and_does_not_weight_long_answers():
    rows=[dict(problem_id=0,response_ids=[5,3],logprobs=[-1.0,-3.0],text='a',gold='a',
               calculator_marker=False,word4_overlap=None),
          dict(problem_id=1,response_ids=[6,7,8,3],logprobs=[-4.0]*4,text='b',gold='b',
               calculator_marker=True,word4_overlap=0.5)]
    value=summarize_rows(rows)
    assert value['nll']==3.0 # (2+4)/2, not (4+16)/6
    assert value['length_median']==3 and value['n_answers']==2 and value['n_tokens']==6
    assert value['calculator_marker']==dict(numerator=1,denominator=2,rate=0.5)
    assert value['word4_overlap']['defined_answers']==1 and value['word4_overlap']['undefined_answers']==1
    assert value['word4_overlap']['mean']==0.5


def test_paired_bootstrap_is_difference_of_medians_not_median_of_differences():
    left=np.array([0.,100.,100.]);right=np.array([0.,0.,100.])
    value=paired_bootstrap(left,right,'length')
    assert value['estimate']==100
    assert np.median(left-right)==0
    rng=np.random.default_rng(42)
    indices=rng.integers(0,3,size=(10000,3))
    expected=np.median(left[indices],axis=1)-np.median(right[indices],axis=1)
    assert value['ci_low']==np.quantile(expected,0.025)
    assert value['ci_high']==np.quantile(expected,0.975)
    assert value['resamples']==10000 and value['seed']==42


def test_mean_bootstrap_uses_paired_problem_indices_and_reports_empty_cohort():
    value=paired_bootstrap([1.,2.,4.],[1.,2.,4.],'nll')
    assert value['estimate']==value['ci_low']==value['ci_high']==0
    assert paired_bootstrap([],[],'nll')['status']=='undefined-empty-common-cohort'
    with pytest.raises(ValueError):paired_bootstrap([1.],[1.,2.],'nll')
    with pytest.raises(ValueError):paired_bootstrap([float('nan')],[1.],'nll')

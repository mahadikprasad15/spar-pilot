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


def test_explicit_word_policy_and_occurrence_weighted_grams_do_not_fill_short_text_with_zero():
    from pilot_eval.sampling_sft_gate_report import words,leakage
    assert words('ＣＡＮ’T café １２','unicode-words-v1')==["can't",'café','12']
    with pytest.raises(ValueError):words('A, B.','unapproved-policy')
    result=leakage('a b c d a b c d','a b c d\n#### 7','unicode-words-v1')
    assert result['word4_hits']==2 and result['word4_total']==5
    assert result['word4_overlap']==0.4
    assert leakage('short answer','answer\n#### 7','unicode-words-v1')['word4_overlap'] is None
    assert leakage('<<','answer\n#### 7','unicode-words-v1')['calculator_marker']


def test_training_request_costs_use_buffer_exact_caps_and_explicit_flags():
    from pilot_eval.sampling_sft_gate import training_forecast
    ledger=dict(committed_usd='4.000000',review_required=False,stage_totals={'training':{'committed_usd':'0.000000'}})
    value=training_forecast(dict(arms=[0,5],per_arm={'0':dict(rate_usd_per_hour='1',estimated_hours='1',setup_hours='0.1'),
        '5':dict(rate_usd_per_hour='1',estimated_hours='1',setup_hours='0.1')},notes='Measured estimate'),ledger)
    assert value['reservation_usd']=='3.200000' and not value['blocked']
    with pytest.raises(ValueError):training_forecast(dict(arms=[2],per_arm={},notes='Not separately approved'),ledger)
    with pytest.raises(ValueError):training_forecast(dict(arms=[0],per_arm={},notes='Missing arm cost'),ledger)


def test_report_figures_include_paired_estimates_and_interval_inputs(tmp_path):
    import json
    from pilot_eval.sampling_sft_gate_report import render
    (tmp_path/'results').mkdir()
    cohort={m:dict(n_answers=2,n_tokens=4,nll=1.,length_median=2.,length_p90=2.) for m in ['gold','k0','k2','k5']}
    interval=dict(estimate=1.,ci_low=-0.2,ci_high=1.5)
    report=dict(claim='Fixture',execution_kind='cpu-fixture',common_problem_count=2,own_cohort=cohort,common_cohort=cohort,
        paired_contrasts={'gold-minus-k0':{'nll':interval,'length':interval}},coverage={},collection={},word_policy={})
    render(tmp_path,report,[])
    manifest=json.loads((tmp_path/'plots/manifest.json').read_text())
    pair=next(p for p in manifest['plots'] if p['path']=='paired-nll.svg')
    assert pair['intervals']=={'gold-minus-k0':interval}
    assert (tmp_path/'plots/paired-length.svg').exists()

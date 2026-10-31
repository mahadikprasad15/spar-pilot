import pytest
from pilot_eval.sampling_sft_cache_diagnostic import compare_scores


def test_comparison_retains_token_alignment_and_reports_cache_effect():
    fixtures=[dict(trace_id='a',prompt_ids=[1],response_ids=[2,3])]
    def score(values):return [dict(token_ids=[2,3],positions=[1,2],logprobs=values)]
    result=compare_scores(fixtures,score([-1.,-2.]),score([-1.1,-2.]),score([-1.1,-2.1]))
    assert result['on_vs_off']['max_abs_error']==pytest.approx(0.1)
    assert result['off_vs_hf']['max_abs_error']==pytest.approx(0.1)
    assert result['on_vs_hf']['max_abs_error']==pytest.approx(0.1)
    assert result['on_vs_off']['worst']['response_index']==0
    with pytest.raises(ValueError):
        compare_scores(fixtures,score([-1.,-2.]),score([-1.1,-2.]),[dict(token_ids=[3,2],positions=[1,2],logprobs=[-1.,-2.])])


def test_comparison_rejects_missing_or_nonfinite_scores():
    fixture=[dict(trace_id='a',prompt_ids=[1],response_ids=[2])]
    good=[dict(token_ids=[2],positions=[1],logprobs=[-1.0])]
    bad=[dict(token_ids=[2],positions=[1],logprobs=[float('nan')])]
    with pytest.raises(ValueError):compare_scores(fixture,good,good,[])
    with pytest.raises(ValueError):compare_scores(fixture,good,bad,good)

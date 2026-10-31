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


def test_singleton_settings_change_only_scheduler_limit():
    from pilot_eval.sampling_sft_cache_diagnostic import diagnostic_variants
    original={'runtime':{'prefix_caching':True,'max_num_seqs':32,'dtype':'float32'},'mode':'scores'}
    variants=diagnostic_variants(original,'singleton')
    assert variants==[('singleton',{'runtime':{'prefix_caching':True,'max_num_seqs':1,'dtype':'float32'},'mode':'scores'})]
    assert original['runtime']['max_num_seqs']==32
    with pytest.raises(ValueError):diagnostic_variants(original,'unknown')


def test_ieee_variant_keeps_original_runtime():
    from pilot_eval.sampling_sft_cache_diagnostic import diagnostic_variants
    original={'runtime':{'prefix_caching':True,'max_num_seqs':32},'mode':'scores'}
    assert diagnostic_variants(original,'ieee')==[('ieee',original)]

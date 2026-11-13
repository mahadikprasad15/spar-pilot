import pytest
from pilot_eval.sampling_sft_generation_bridge import check_contract


def test_mixed_writer_plan_requires_equal_scientific_contract():
    baseline = {'source_plan':'s', 'pilot_plan':'p', 'pilot_report':'r',
                'settings':{'runtime':{'seed':0}}, 'problems':[{'id':1}],
                'arms':[0,2,5], 'data_seed':0, 'retry_indices':[0,1,2,3],
                'retry_data_seeds':[0,1001,1002,1003], 'algorithm_hash':'a',
                'runtime_hash':'b', 'baseline':'deferred', 'eligibility':'x'}
    other = {**baseline, 'writer':{'mode':'buffered-before-seal-v1'}}
    check_contract(baseline, other)
    with pytest.raises(ValueError): check_contract(baseline, {**other, 'runtime_hash':'other'})
    with pytest.raises(ValueError): check_contract(baseline, {**other, 'problems':[{'id':2}]})
    with pytest.raises(ValueError): check_contract(baseline, {**other, 'retry_data_seeds':[1]})


def test_mixed_verifier_rejects_cpu_fixture_arm(tmp_path, monkeypatch):
    from pilot_eval import sampling_sft_generation_bridge as bridge
    plan = dict.fromkeys(bridge.CONTRACT, 'same')
    plan['writer'] = {'mode': 'buffered-before-seal-v1'}
    monkeypatch.setattr(bridge.old, 'load', lambda *a: (tmp_path, plan, tmp_path))
    monkeypatch.setattr(bridge.new, 'load', lambda *a: (tmp_path, plan, tmp_path))
    monkeypatch.setattr(bridge.new, 'check_writer_replay', lambda *a: None)
    monkeypatch.setattr(bridge.old, 'verify_arm', lambda *a: {'state':'completed', 'execution_kind':'cpu-fixture'})
    with pytest.raises(ValueError, match='real-CUDA'):
        bridge.verify(tmp_path/'old', tmp_path/'new', tmp_path)

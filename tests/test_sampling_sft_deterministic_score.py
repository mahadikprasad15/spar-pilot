import pytest
from pilot_eval.sampling_sft_deterministic_score import compare_scores


def test_exact_score_repeat_and_alignment():
    fixtures = [dict(problem_id=14, prompt_ids=[1], response_ids=[2, 3])]
    a = [dict(token_ids=[2, 3], positions=[1, 2], logprobs=[-.6, -.4])]
    b = [dict(token_ids=[2, 3], positions=[1, 2], logprobs=[-.60002, -.4])]
    result = compare_scores(fixtures, a, b)
    assert result['different_values'] == 1 and result['first_difference']['problem_id'] == 14
    assert result['max_abs_difference'] > .000019
    b[0]['positions'] = [2, 3]
    with pytest.raises(ValueError): compare_scores(fixtures, a, b)


def test_two_lanes_reuse_seals_and_cannot_import_real_engine_without_job(tmp_path, monkeypatch):
    import json
    import os
    from pilot_eval import sampling_sft_deterministic_score as d
    fixture=[dict(problem_id=14,prompt_ids=[1],response_ids=[2])]
    (tmp_path/'fixture.json').write_text(json.dumps(fixture))
    plan=dict(runtime_hash='runtime',fixture='fixture.json',fixture_sha256='fixture-hash')
    parent=dict(problems=[])
    folder=tmp_path/'diagnostic'
    monkeypatch.setattr(d,'load',lambda *args:(tmp_path,plan,folder,parent))
    monkeypatch.setenv(d.ENV_NAME,'prior')
    class Engine:
        def __init__(self):self.calls=0
        def score(self,requests):
            self.calls+=1
            return [dict(token_ids=[2],positions=[1],logprobs=[-.6-self.calls*.001])]
    class Dependencies:
        def __init__(self):self.engine=Engine()
        def open(self,*args):
            assert os.environ[d.ENV_NAME]=='0'
            return self.engine,None,{}
    first=Dependencies()
    a=d.run('ignored',tmp_path,'a',None,dependencies=first)
    assert first.engine.calls==2 and a['within_process']['different_values']==1
    assert os.environ[d.ENV_NAME]=='prior'
    assert d.run('ignored',tmp_path,'a',None,dependencies=Dependencies())==a
    assert d.run('ignored',tmp_path,'b',None,dependencies=Dependencies())['lane']=='b'
    result=d.report('ignored',tmp_path)
    assert result['between_first']['different_values']==0
    assert not result['production_ready']
    (folder/'b/results.json').write_text('{}')
    with pytest.raises(ValueError,match='seal'):d.report('ignored',tmp_path)
    monkeypatch.setattr(d.instrument,'authorize',lambda *a,**k: (_ for _ in ()).throw(ValueError('no job')))
    monkeypatch.setattr(d,'load',lambda *args:(tmp_path,plan,tmp_path/'fresh',parent))
    with pytest.raises(ValueError,match='no job'):d.run('ignored',tmp_path,'a',None)


def test_report_rejects_real_cuda_without_effective_setting_or_fresh_process(tmp_path, monkeypatch):
    import json
    from pilot_eval import sampling_sft_deterministic_score as d
    fixture=[dict(problem_id=14,prompt_ids=[1],response_ids=[2])]
    (tmp_path/'fixture.json').write_text(json.dumps(fixture))
    plan=dict(runtime_hash='runtime',fixture='fixture.json',fixture_sha256='fixture-hash')
    values=[dict(token_ids=[2],positions=[1],logprobs=[-.6])]
    def result(pid, flag):
        return dict(runtime_hash='runtime',fixture_sha256='fixture-hash',environment={d.ENV_NAME:'0'},
                    host='host',pid=pid,worker=dict(execution_kind='real-cuda',vllm_effective_multiprocessing=flag),
                    first=values,second=values)
    lane_data={'a':result(1,False),'b':result(2,True)}
    folder=tmp_path/'diag';folder.mkdir()
    for lane in ('a','b'):
        (folder/lane).mkdir(); (folder/lane/'complete.json').write_text('{}')
    monkeypatch.setattr(d,'load',lambda *args:(tmp_path,plan,folder,{}))
    monkeypatch.setattr(d,'lane_result',lambda folder,plan,lane:lane_data[lane])
    with pytest.raises(ValueError,match='effective'):d.report('ignored',tmp_path)
    lane_data['b']=result(1,False)
    with pytest.raises(ValueError,match='process'):d.report('ignored',tmp_path)

import json
import pytest


def test_blind_package_is_deterministic_key_is_separate_and_unblind_waits_for_settlement(tmp_path):
    from pilot_eval.trace_grading import prepare_grading, analyze_grades
    lists=[{'condition':'sft','tokens':[' total',' apples']},{'condition':'grpo','tokens':['####',' dollars']}]
    package=prepare_grading(tmp_path/'a',lists,['total'],salt='fixture-secret',synthetic=True)
    other=prepare_grading(tmp_path/'b',lists,['total'],salt='fixture-secret',synthetic=True)
    assert (tmp_path/'a/grading/lists.jsonl').read_bytes()==(tmp_path/'b/grading/lists.jsonl').read_bytes()
    public=[json.loads(line) for line in (tmp_path/'a/grading/lists.jsonl').read_text().splitlines()]
    assert all(set(row)=={'list_id','tokens'} for row in public)
    assert not (tmp_path/'a/grading/key.json').exists()
    assert (tmp_path/'a/grading_key/key.json').exists()
    a=[dict(list_id=row['list_id'],labels=[1,0]) for row in public]
    b=[dict(list_id=row['list_id'],labels=[0,0]) for row in public]
    result=analyze_grades(package,a,b)
    assert result['state']=='awaiting-settlement' and 'by_condition' not in result
    assert len(result['disagreements'])==2
    settlements={f"{row['list_id']}:0":1 for row in public}
    final=analyze_grades(package,a,b,settlement={'reviewed':True,'reviewer':'Prasad','notes':'Reviewed every disagreement.','labels':settlements})
    assert final['state']=='settled' and set(final['by_condition'])=={'sft','grpo'}
    assert final['by_condition']['sft']['settled_relevance']==.5
    assert prepare_grading(tmp_path/'a',lists,['total'],salt='fixture-secret',synthetic=True)==package
    with pytest.raises(ValueError,match='IDs'):
        analyze_grades(package,a,a+[a[0]])

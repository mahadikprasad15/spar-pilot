import json
from pathlib import Path
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
    resumed=analyze_grades(package,a,b)
    assert resumed['state']=='settled'
    assert (Path(resumed['grade_run_path'])/'grader_claude.jsonl').is_file()
    assert prepare_grading(tmp_path/'a',lists,['total'],salt='fixture-secret',synthetic=True)==package
    with pytest.raises(ValueError,match='IDs'):
        analyze_grades(package,a,a+[a[0]])


def test_agreement_hand_cases_and_naming_require_isolated_session(tmp_path):
    import math
    from pilot_eval.trace_grading import agreement,prepare_grading,analyze_naming
    assert agreement([0,1,0,1],[0,1,0,1])['kappa']==1
    assert agreement([0,0,1,1],[0,1,0,1])['kappa']==0
    assert agreement([0,0],[0,1])['kappa']==0
    assert math.isnan(agreement([0,0],[0,0])['kappa'])
    package=prepare_grading(tmp_path/'package',[{'condition':'sft','tokens':[' total']}],['total'],salt='fixture',synthetic=True)
    row=json.loads((package/'grading/lists.jsonl').read_text().splitlines()[0])
    grades=[{'list_id':row['list_id'],'choice':1}]
    with pytest.raises(ValueError,match='isolated'):
        analyze_naming(package,grades,session_record={'fresh_session':False})
    result=analyze_naming(package,grades,session_record={'fresh_session':True,'naming_before_relevance':True,'session_id':'fixture','grader':'fixture'})
    assert result['by_condition']['sft']['target_choice_rate']==1
    assert result['nominal_chance']==.125
    assert (Path(result['artifact_path'])/'choices.jsonl').is_file()


def test_approved_grading_inventory_has_106_unique_conditions():
    from pilot_eval.trace_grading import expected_grading_conditions
    conditions=expected_grading_conditions()
    assert len(conditions)==len(set(conditions))==106
    assert len([c for c in conditions if c.startswith('null/')])==40
    assert len([c for c in conditions if c.startswith('delta/')])==36
    assert len([c for c in conditions if c.startswith('activation/')])==30

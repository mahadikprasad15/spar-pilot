import pytest
from pilot_eval.behaviour_assisted_report import summarize


def record(method, step, answer, correct):
    return dict(method=method,step=step,item_id='q',generated_text='response',gold='#### 18',
                scores={'flexible_v3':dict(correct=correct)})


def test_confirmed_correction_updates_shared_base_without_double_counting():
    pairs=[dict(method=m,item_id='q',before=record(m,0,'x',False),after=record(m,64,'x',True)) for m in ['sft','grpo']]
    refs=[dict(method=m,step=0,item_id='q',record=p['before']) for m,p in zip(['sft','grpo'],pairs)]
    mapping=[dict(response_id='r',references=refs)]
    assistance={'rows':[dict(response_id='r',index=1,human_label=dict(status='numeric',answer=None),proposal={'answer':'18'})]}
    result=summarize(pairs,mapping,assistance,[{'labels':[dict(response_id='r',status='numeric',answer='18')]}])
    assert result['unique_confirmed_numeric']==1
    assert len(result['changes'])==2
    for m in ['sft','grpo']:
        assert result['methods'][m]['original']['gain']==1
        assert result['methods'][m]['confirmed_corrections_only']['gain']==0
    assert all(not p['before']['scores']['flexible_v3']['correct'] for p in pairs)


def test_unconfirmed_proposals_are_not_used_and_conflicts_rejected():
    pairs=[dict(method='grpo',item_id='q',before=record('grpo',0,'x',False),after=record('grpo',64,'x',True))]
    mapping=[dict(response_id='r',references=[dict(method='grpo',step=0,item_id='q',record=pairs[0]['before'])])]
    assistance={'rows':[dict(response_id='r',index=1,human_label=dict(status='numeric',answer=None),proposal={'answer':'18'})]}
    result=summarize(pairs,mapping,assistance,[])
    assert result['methods']['grpo']['confirmed_corrections_only']['gain']==1
    with pytest.raises(ValueError,match='conflicting'):
        summarize(pairs,mapping,assistance,[{'labels':[dict(response_id='r',status='numeric',answer='18')]},
                                          {'labels':[dict(response_id='r',status='numeric',answer='19')]}])

from pilot_eval import sampling_sft_mixed_handoff as handoff
from pilot_eval import sampling_sft_handoff as original


def test_handoff_reuses_completion_rule_and_never_claims_missing_external_evidence(tmp_path,monkeypatch):
    assert handoff.completion is original.completion
    root=tmp_path
    for n in ('source','generation','review','gate','data','selection','k0','k2','k5'):
        (root/n).write_text('{}')
    selection={'source_plan':'source','generation_plan':'generation','review_plan':'review',
        'gate_plan':'gate','training_data_plan':'data','training_selection':'selection',
        'training_plans':{'0':'k0','2':'k2','5':'k5'}}
    monkeypatch.setattr(handoff.source,'status',lambda *a:{'audit':{'source_ready':True}})
    monkeypatch.setattr(handoff.generation,'load',lambda *a:(root,{'source_plan':'source'},root))
    monkeypatch.setattr(handoff.generation,'validate_for_scoring',lambda *a:None)
    monkeypatch.setattr(handoff.review,'load',lambda *a:(root,{'generation_plan':'generation'},root))
    monkeypatch.setattr(handoff.review,'verify',lambda *a:{'coverage_blocked':False})
    gate_plan={'review_plan':'review','generation_plan':'generation'}
    monkeypatch.setattr(handoff.gate,'load',lambda *a:(root,gate_plan,root))
    monkeypatch.setattr(handoff.gate,'verify',lambda *a:{'execution_kind':'real-cuda'})
    monkeypatch.setattr(handoff.data,'load',lambda *a:(root,{'review_plan':'review','generation_plan':'generation'},root))
    monkeypatch.setattr(handoff.data,'verify',lambda *a:{})
    monkeypatch.setattr(handoff.data,'status',lambda *a:{'readiness':{'t15_passed':True,'t16_passed':True,'training_ready':True}})
    monkeypatch.setattr(handoff.training,'selection',lambda *a:({'arms':[0,2,5]},{'config_sha256':handoff._hash(gate_plan)}))
    monkeypatch.setattr(handoff.training,'load',lambda p,*a:(root,{'k':int(str(p)[-1]),'data_plan':'data','selection':'selection'},root))
    monkeypatch.setattr(handoff.training,'verify',lambda *a:{'successful_steps':64,'execution_kind':'real-cuda'})
    result=handoff.assess(root,selection)
    assert result['state']=='partial'
    assert not result['stages']['lifecycle_complete'] and not result['stages']['budget_settled']
    assert result['training_authorized'] is False

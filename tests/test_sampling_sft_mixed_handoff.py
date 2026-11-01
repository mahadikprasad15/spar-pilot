from pilot_eval import sampling_sft_mixed_handoff as handoff
from pilot_eval import sampling_sft_handoff as original
import json
import pytest


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


def test_saved_handoff_detects_changed_parent(tmp_path,monkeypatch):
    selected={'source_plan':'source','generation_plan':'generation','review_plan':'review',
        'gate_plan':'gate','training_data_plan':'data','training_selection':'selection',
        'training_plans':{'0':'k0','2':'k2','5':'k5'}}
    for path in ['source','generation','review','gate','data','selection','k0','k2','k5']:
        (tmp_path/path).write_text('{}')
    selection=tmp_path/'selection-input.json';selection.write_text(json.dumps(selected))
    monkeypatch.setattr(handoff,'assess',lambda *a:{'state':'partial','training_authorized':False})
    report=handoff.build(selection,tmp_path,'fixture')
    assert handoff.verify(report,tmp_path)['state']=='partial'
    (tmp_path/'k5').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='parent'):
        handoff.verify(report,tmp_path)


def test_dynamic_inventory_detects_nested_artifacts_and_live_writer(tmp_path):
    run=tmp_path/'run';(run/'rounds/0/blocks/000').mkdir(parents=True)
    (run/'rounds/0/blocks/000/scientific.jsonl').write_text('record\n')
    inv=handoff.dynamic_inventory(tmp_path,[run])
    assert 'run/rounds/0/blocks/000/scientific.jsonl' in inv
    (run/'writer.json').write_text('{}')
    with pytest.raises(ValueError,match='writer'):
        handoff.dynamic_inventory(tmp_path,[run])


def test_budget_waiver_does_not_fake_settlement():
    assert not handoff.budget_settled({'unreconciled_gpu_jobs':['x'],
        'jobs':{'x':{'kind':'gpu','launch_recorded':True,'state':'reserved','budget_limits_waived':True}}})
    assert handoff.budget_settled({'unreconciled_gpu_jobs':[],
        'jobs':{'x':{'kind':'gpu','launch_recorded':True,'state':'settled','budget_limits_waived':True}}})


def test_dynamic_inventory_rejects_symlink_escape(tmp_path):
    folder=tmp_path/'run';folder.mkdir();(folder/'link').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='symlink'):
        handoff.dynamic_inventory(tmp_path,[folder])


def test_lifecycle_requires_every_launched_job_with_matching_copies(tmp_path,monkeypatch):
    ledger={'jobs':{'a':{'kind':'gpu','launch_recorded':True,'state':'settled'},
                    'b':{'kind':'gpu','launch_recorded':True,'state':'settled'}},'unreconciled_gpu_jobs':[]}
    evidence=[{'job_id':'a','launched':True,'work_complete':True,'copies_verified':True}]
    assert not handoff.lifecycle_complete(ledger,evidence)
    evidence.append({'job_id':'b','launched':True,'work_complete':True,'copies_verified':True})
    assert handoff.lifecycle_complete(ledger,evidence)
    evidence[1]['copies_verified']=False
    assert not handoff.lifecycle_complete(ledger,evidence)


def test_full_inventory_rejects_old_fixed_job_manifest_as_backup(tmp_path):
    full={'run/science/000':{'sha256':'a','bytes':1},'run/science/001':{'sha256':'b','bytes':1}}
    partial={'files':{'run/science/000':{'sha256':'a','bytes':1}}}
    with pytest.raises(ValueError,match='full dynamic inventory'):
        handoff.validate_full_copy_inventory(full,partial)


def test_dynamic_inventory_manifest_seals_all_nested_files(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'block.jsonl').write_text('science\n')
    path=handoff.create_full_manifest(tmp_path,[source],tmp_path/'manifest.json','worker')
    value=json.loads(path.read_text())
    assert value['files']==handoff.dynamic_inventory(tmp_path,[source])
    assert value['sha256']==handoff._hash({k:v for k,v in value.items() if k!='sha256'})
    (source/'block.jsonl').write_text('changed\n')
    with pytest.raises(ValueError,match='changed'):
        handoff.verify_full_manifest(path,tmp_path)

import json
import pytest
from pilot_eval import sampling_sft_mixed_generation as mixed


def test_mixed_plan_binds_each_original_arm_without_resealing(tmp_path, monkeypatch):
    root=tmp_path
    old=root/'old.json'; new=root/'new.json'; source=root/'source.json'
    old.write_text('{}'); new.write_text('{}'); source.write_text(json.dumps({'run_path':'runs/source'}))
    legacy={'source_plan':'source.json','problems':[{'id':0}], 'runtime_hash':'r',
            'algorithm_hash':'a', 'settings':{'runtime':{}}, 'arms':[0,2,5]}
    provenance={'arms':{'0':{'arm_complete_sha256':'a'},'2':{'arm_complete_sha256':'b'},
                        '5':{'arm_complete_sha256':'c'}},'training_allowed':False}
    monkeypatch.setattr(mixed.bridge,'verify',lambda *a:provenance)
    monkeypatch.setattr(mixed.old,'load',lambda *a:(root,legacy,root/'old-arm'))
    monkeypatch.setattr(mixed.new,'load',lambda *a:(root,{},root/'new-arm'))
    config=mixed.prepare(old,new,root,'mixed-fixture')
    _,plan,folder=mixed.load(config,root)
    assert plan['provenance']==provenance
    assert mixed.arm_directory(root,plan,0)==root/'old-arm/k0'
    assert mixed.arm_directory(root,plan,5)==root/'new-arm/k5'
    assert mixed.verify(config,root)['training_allowed'] is False
    provenance['arms']['5']['arm_complete_sha256']='changed'
    with pytest.raises(ValueError,match='provenance'):mixed.load(config,root)

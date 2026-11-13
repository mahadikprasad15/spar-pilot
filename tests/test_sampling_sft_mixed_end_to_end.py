"""Synthetic bridge fixture: no GPU or production qualification is claimed."""
import json
import shutil

from pilot_eval import sampling_sft_generation as old
from pilot_eval import sampling_sft_generation_v2 as new
from pilot_eval import sampling_sft_generation_bridge as bridge
from pilot_eval import sampling_sft_mixed_generation as mixed
from pilot_eval import sampling_sft_mixed_review as review
from pilot_eval import sampling_sft_mixed_gate as gate
from pilot_eval import sampling_sft_mixed_training_data as data
from test_sampling_sft_generation import Dependencies as GenerationDependencies, source_template
from test_sampling_sft_review import collected
from test_sampling_sft_gate import reviewed, Dependencies as OracleDependencies


def test_synthetic_mixed_arms_through_review_gate_and_stored_ids(tmp_path,reviewed,monkeypatch):
    shutil.copytree(reviewed,tmp_path,dirs_exist_ok=True)
    root=tmp_path;legacy=root/'plans/generation/sampling-sft.generation.json'
    buffered=new.prepare(root/'plans/pilot/sampling-sft.pilot.json',root,'mixed-fixture-k5')
    new.run(buffered,root,5,dependencies=GenerationDependencies())
    def fixture_bridge(a,b,artifact_root):
        _,left,lf=old.load(a,artifact_root);_,right,rf=new.load(b,artifact_root)
        bridge.check_contract(left,right)
        for k,module,plan,folder in ((0,old,left,lf),(2,old,left,lf),(5,new,right,rf)):
            result=module.verify_arm(artifact_root,plan,k,folder/f'k{k}')
            assert result['state']=='completed' and result['execution_kind']=='cpu-fixture'
        return {'scope':'explicit CPU fixture provenance','training_allowed':False}
    monkeypatch.setattr(bridge,'verify',fixture_bridge)
    plan=mixed.prepare(legacy,buffered,root,'mixed-e2e')
    review_plan=review.prepare(plan,root,'mixed-e2e-review')
    draft=root/'mixed-e2e-labels.json';review.export(review_plan,root,draft)
    value=json.loads(draft.read_text());value['reviewer']='fixture-human'
    for row in value['labels']:row.update(status='ambiguous',answer=None,evidence='',note='fixture only')
    draft.write_text(json.dumps(value))
    review.import_labels(review_plan,root,draft,'human',confirm_human=True)
    coverage=review.freeze(review_plan,root)
    assert coverage['human_review_complete'] and not coverage['training_allowed']
    policy=root/'mixed-e2e-policy.json';policy.write_text(json.dumps(
        dict(version='unicode-words-v1',approved=True,reviewer='fixture',notes='fixture only')))
    gate_plan=gate.prepare(review_plan,root,policy,'mixed-e2e-gate')
    gate.score(gate_plan,root,dependencies=OracleDependencies())
    result=gate.report(gate_plan,root)
    assert result['execution_kind']=='cpu-fixture' and not result['training_authorized']
    data_plan=data.prepare(review_plan,root,'mixed-e2e-data')
    _,config,_=data.load(data_plan,root)
    _,arms=data.expected(root,config)
    assert all(len(arms[str(k)]['batches'])==64 for k in (0,2,5))
    assert all(row['labels'][:row['prompt_tokens']]==[-100]*row['prompt_tokens']
               for arm in arms.values() for row in arm['rows'])

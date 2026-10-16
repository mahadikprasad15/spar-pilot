"""Saved readout consumer: reveal barriers, literal ranks, no decoder inference."""
import json
from pathlib import Path
import numpy as np
import pytest
from test_trace_production import frozen,CaptureEngine,Dependencies


def evidence(root,*,reviewed=True):
    from pilot_eval.trace_production import run_capture
    from pilot_eval.trace_readout_assets import prepare_assets
    from pilot_eval.trace_vocabulary import freeze_references,approve_references
    from pilot_eval.training import file_hash
    path,_=frozen(root);run_capture(path,root,dependencies=Dependencies(CaptureEngine()))
    class Assets:
        def load(self,config):
            return dict(execution_kind='synthetic-cpu',norm=np.ones(2,dtype=np.float32),
                head=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,1]],dtype=np.float32),epsilon=1e-6,
                decoded={0:' first',1:' second',2:' minus',3:' negative',4:' both'},excluded_ids=[],origin_files={})
    assets=prepare_assets(path,root,'assets',dependencies=Assets())
    refs={name:dict(strings=['both']) for name in ['ref_questions','ref_gold','ref_grpo']}
    references=freeze_references(root/'references',refs,parameters={'fixture':True},input_files={'production':path})
    (references.parent/'top30.md').write_text('independent reference fixture')
    if reviewed:approve_references(references,reviewer='fixture',notes='Reference inspection',exposure_status='not-seen')
    review=root/'readout-review.json'
    review.write_text(json.dumps(dict(approved=True,reviewer='fixture',notes='Predeclared readouts',
        capture_sha256=file_hash(path),references_sha256=file_hash(references),
        primary_family='fixture family; no inferential claims',
        requests=[dict(name='sft-primary',condition='sft-64',role='delta',site='block-14-output',positions=[0,1,2,3,4]),
                  dict(name='grpo-primary',condition='grpo-64',role='delta',site='block-14-output',positions=[0,1,2,3,4]),
                  dict(name='base-control',condition='base',role='base',site='block-14-output',positions=[0,1,2,3,4]),
                  dict(name='zero',condition='base',role='delta',site='block-14-output',positions=[0,1,2,3,4]),
                  dict(name='random-reference',condition='random-64',role='delta',site='block-14-output',positions=[0,1,2,3,4]),
                  dict(name='gaussian',condition='sft-64',role='random-direction',site='block-14-output',positions=[0,1,2,3,4],seed=7)],
        lens_names=['logit'],partial_mode_approved=True)))
    return path,assets,references,review


def test_readout_refuses_unreviewed_references_before_loading_arrays(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts
    path,assets,references,review=evidence(tmp_path,reviewed=False)
    with pytest.raises(ValueError,match='reference review'):prepare_readouts(path,assets,references,review,tmp_path,'readouts')
    assert not (tmp_path/'runs/pilot-5/readouts/readouts/complete.json').exists()


def test_readout_ranks_saved_primary_vectors_and_keeps_controls_distinct(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts,run_readouts,verify_readouts
    path,assets,references,review=evidence(tmp_path)
    plan=prepare_readouts(path,assets,references,review,tmp_path,'readouts')
    result=run_readouts(plan,tmp_path)
    assert result['readouts_complete'] and result['decoder_forwards']==0
    directory=tmp_path/json.loads(plan.read_text())['run_path']
    sft=json.loads((directory/'units/sft-primary/result.json').read_text())
    assert sft['readouts']['logit']['positive']['raw_top_tokens'][0]['token_id']==4
    assert sft['readouts']['logit']['negative']['raw_top_tokens'][0]['token_id']==2
    np.testing.assert_allclose(np.load(directory/'units/sft-primary/vector.npy'),[.25,.25])
    assert sft['coherence']==pytest.approx(1.)
    zero=json.loads((directory/'units/zero/result.json').read_text())
    assert zero['state']=='undefined' and zero['reason']=='zero-vector'
    roles=[json.loads((directory/'units'/n/'result.json').read_text())['role'] for n in ['base-control','random-reference','gaussian']]
    assert roles==['base','delta','random-direction']
    assert verify_readouts(plan,tmp_path)==result
    assert run_readouts(plan,tmp_path)==result
    (directory/'units/sft-primary/vector.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash'):verify_readouts(plan,tmp_path)


def test_readouts_resume_only_missing_units_and_report_unsupported_control(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts,run_readouts
    path,assets,references,review=evidence(tmp_path)
    data=json.loads(review.read_text())
    data['requests'].append(dict(name='unsupported-base-window',condition='base',role='base',site='block-14-output',positions=[127]))
    review.write_text(json.dumps(data))
    plan=prepare_readouts(path,assets,references,review,tmp_path,'resume-readouts')
    first=run_readouts(plan,tmp_path,max_new_units=2)
    assert not first['readouts_complete'] and first['new_units']==2
    directory=tmp_path/json.loads(plan.read_text())['run_path']
    original=(directory/'units/sft-primary/result.json').read_bytes()
    second=run_readouts(plan,tmp_path)
    assert second['new_units']==5 and second['reused_units']==2
    assert (directory/'units/sft-primary/result.json').read_bytes()==original
    unsupported=json.loads((directory/'units/unsupported-base-window/result.json').read_text())
    assert unsupported['state']=='unavailable' and unsupported['readouts']=={}


def test_cpu_readout_weight_loader_handles_bfloat16_without_torch(tmp_path):
    from pilot_eval.trace_readout_assets import read_weight
    header=json.dumps({'weight':dict(dtype='BF16',shape=[1,2],data_offsets=[0,4])}).encode()
    path=tmp_path/'tiny.safetensors';path.write_bytes(len(header).to_bytes(8,'little')+header+bytes.fromhex('c03f00c0'))
    np.testing.assert_array_equal(read_weight(path,'weight'),[[1.5,-2.]])
    bad=json.dumps({'weight':dict(dtype='BF16',shape=[1000,1000],data_offsets=[0,4])}).encode()
    path.write_bytes(len(bad).to_bytes(8,'little')+bad+bytes(4))
    with pytest.raises(ValueError,match='byte|shape'):read_weight(path,'weight')

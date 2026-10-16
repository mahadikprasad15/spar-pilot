"""Saved production capture, using an independent tiny CPU model boundary."""
import json
from pathlib import Path
import numpy as np
import pytest
from test_trace_profile import prepared, Deps, Engine


def frozen(root, *, approved=True):
    from pilot_eval.trace_profile import run_profile
    from pilot_eval.trace_production_plan import propose_capture, freeze_capture
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    profile=prepared(root);run_profile(profile,root,dependencies=Deps(Engine()))
    proposal=propose_capture(profile,root,'production')
    data=json.loads(proposal.read_text())
    # Independently authored fixture qualification; never accepted by real CUDA.
    qualification=root/'fixture-qualification.json'
    qualification.write_text(json.dumps(dict(protocol='pilot5-capture-qualification-v1',
        qualified=True,execution_kind='synthetic-cpu',profile_sha256=file_hash(profile),
        model_revision=data['model_revision'],batch_size=4,storage_dtype='float32',
        checks={k:dict(passed=True,evidence='independent fixture only') for k in
            ['zero_identity','base_integrity','numeric_lens','batching','retention','ranking','bootstrap']})))
    (qualification.parent/(qualification.name+'.seal.json')).write_text(json.dumps(dict(sha256=file_hash(qualification))))
    review=root/'capture-review.json';review.write_text(json.dumps(dict(approved=approved,
        reviewer='fixture-only',notes='Synthetic qualification and budget; never scientific evidence.',
        proposal_sha256=file_hash(proposal),conditions=['base','sft-64','grpo-64','random-64'],
        batch_size=4,storage_dtype='float32',scope='endpoints',
        qualification_path=str(qualification),qualification_sha256=file_hash(qualification),
        lens_mode='logit-only',partial_mode_approved=True,lens_qualification_path=None,
        predictions={'sft':'fixture prediction','grpo':'fixture prediction'},
        budget=dict(approved=True,max_seconds=10000,max_durable_bytes=100000000),
        primary=dict(form='raw',site='block-14-output',step=64,positions=[0,1,2,3,4],
            pooling='positions-within-example-then-examples'))))
    return freeze_capture(proposal,review,root),profile


class CaptureEngine:
    def __init__(self,stop=None):self.calls=[];self.stop=stop;self.closed=False
    def integrity(self):return 'fixture-base'
    def capture(self,rows,condition):
        from pilot_eval.trace_capture_workflow import SITES
        self.calls.append((condition,[r['id'] for r in rows]))
        if len(self.calls)==self.stop:raise RuntimeError('fixture disconnect')
        b=np.stack([np.asarray(r['input_ids'],dtype=np.float32) for r in rows])[:,:,None]
        b=np.concatenate([b,b+1],axis=2)
        shift={'base':0,'sft-64':.25,'grpo-64':.5,'random-64':-.125}[condition]
        return {s:b+i+shift for i,s in enumerate(SITES)}
    def close(self):self.closed=True


class Dependencies:
    def __init__(self,engine):self.instance=engine
    def runtime(self):return dict(execution_kind='synthetic-cpu',device='fixture')
    def engine(self,config,root):return self.instance


def test_capture_requires_review_and_fixture_cannot_qualify_real_execution(tmp_path):
    from pilot_eval.trace_production_plan import freeze_capture, load_execution
    with pytest.raises(ValueError,match='approval'):frozen(tmp_path,approved=False)
    review=tmp_path/'capture-review.json';data=json.loads(review.read_text());data['approved']=True;review.write_text(json.dumps(data))
    proposal=tmp_path/'plans/production/traces.production.proposal.json'
    path=freeze_capture(proposal,review,tmp_path)
    assert load_execution(path,tmp_path)['execution_kind']=='synthetic-cpu'
    from pilot_eval.trace_production import run_capture
    class Real(Dependencies):
        def runtime(self):return dict(execution_kind='real-cuda',device='fixture-forbidden')
    e=CaptureEngine()
    with pytest.raises(ValueError,match='execution kind'):run_capture(path,tmp_path,dependencies=Real(e))
    assert not e.calls


def test_production_resume_matches_uninterrupted_sums_and_durable_release(tmp_path):
    from pilot_eval.trace_production import run_capture,verify_capture,load_aggregate
    path,_=frozen(tmp_path)
    first=CaptureEngine(stop=6)
    with pytest.raises(RuntimeError,match='disconnect'):run_capture(path,tmp_path,dependencies=Dependencies(first))
    assert first.closed
    directory=tmp_path/json.loads(path.read_text())['run_path']
    assert (directory/'units/000000/complete.json').exists()
    assert not (directory/'complete.json').exists()
    second=CaptureEngine();result=run_capture(path,tmp_path,dependencies=Dependencies(second))
    assert len(second.calls)==49*4
    assert result['durable_verified'] and result['gpu_release_safe']
    assert result['examples']==200 and result['base_moment_count']==25400
    assert result['new_batches']==49 and result['reused_batches']==1
    sums=load_aggregate(path,tmp_path)
    np.testing.assert_allclose(sums['sft-64.position_sum.block-14-output'],np.full((128,2),50))
    np.testing.assert_allclose(sums['grpo-64.position_sum.block-14-output'],np.full((128,2),100))
    assert verify_capture(path,tmp_path)==result
    class NoGPU:
        def runtime(self):raise AssertionError('completed capture must verify on CPU')
    assert run_capture(path,tmp_path,dependencies=NoGPU())==result
    shard=directory/'units/000000/arrays.npz';shard.write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash'):verify_capture(path,tmp_path)


def test_production_disk_failure_preserves_no_premature_release(tmp_path):
    from pilot_eval.trace_production import run_capture
    path,_=frozen(tmp_path)
    class NoSpace(Dependencies):
        def free_bytes(self,path):return 0
    e=CaptureEngine()
    with pytest.raises(OSError,match='space'):run_capture(path,tmp_path,dependencies=NoSpace(e))
    assert not e.calls
    directory=tmp_path/json.loads(path.read_text())['run_path']
    assert not (directory/'complete.json').exists()
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'


def test_production_refuses_unqualified_gpu_change_before_model(tmp_path):
    from pilot_eval.trace_production import run_capture
    path,_=frozen(tmp_path)
    class OtherGPU(Dependencies):
        def runtime(self):return dict(execution_kind='synthetic-cpu',device='different GPU')
    engine=CaptureEngine()
    with pytest.raises(ValueError,match='qualified runtime'):run_capture(path,tmp_path,dependencies=OtherGPU(engine))
    assert not engine.calls


def test_public_production_cli_verifies_complete_fixture_without_model(tmp_path):
    import subprocess,sys
    from pilot_eval.trace_production import run_capture
    path,_=frozen(tmp_path);run_capture(path,tmp_path,dependencies=Dependencies(CaptureEngine()))
    output=subprocess.run([sys.executable,'-m','pilot_eval','traces-production-verify','--config',str(path),
        '--output-root',str(tmp_path)],capture_output=True,text=True)
    assert output.returncode==0,output.stdout+output.stderr
    assert json.loads(output.stdout)['gpu_release_safe'] is True

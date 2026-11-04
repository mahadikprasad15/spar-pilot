"""Offline source and gauge-invariant product comparisons."""
import numpy as np
import pytest


def test_rank_one_products_are_gauge_invariant():
    from pilot_eval.offline_grpo_internals import product_metrics
    a=np.array([[1.,2.]]);b=np.array([[3.],[4.]])
    result=product_metrics((a,b),(-2*a,-b/2))
    assert result['cosine']==pytest.approx(1.)
    assert result['norm_left']==pytest.approx(np.linalg.norm(b@a))
    assert result['norm_right']==pytest.approx(np.linalg.norm(b@a))


def test_zero_products_have_undefined_cosine():
    from pilot_eval.offline_grpo_internals import product_metrics
    assert product_metrics((np.array([[1.]]),np.array([[0.]])),(np.array([[1.]]),np.array([[1.]])))['cosine'] is None


def test_initial_push_groups_include_dead_draws_and_window_one():
    from pilot_eval.offline_grpo_internals import _push_units
    windows=[]
    for step in range(64):
        window=[]
        for group in range(8):
            q=f'q{step*8+group}'
            window.append(dict(id=q,input_ids=[3],draws=[dict(reward_v3=0,generated_token_ids=[4,99]) for _ in range(8)]))
        windows.append(window)
    units=_push_units(windows)
    assert len(units)==65 and units[-1]['key']=='window-1'
    assert units[-1]['problem_ids']==[q['id'] for q in windows[0]]
    assert units[-1]['token_count']==128 and units[-1]['dead_groups']==8
    assert all(s['advantage']==0 for s in units[-1]['sequences'])


def test_public_prepare_load_rejects_changed_review(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_internals as mod,stage0_spectral,stage0_qualification,activation_prepare,stage0_drift_math
    from pilot_eval.cli import main
    from pilot_eval.workflow import _hash
    from pilot_eval.training import file_hash
    runtime=dict(execution_kind='synthetic-cpu-fixture',session_id='fixture')
    model=dict(model='m',model_revision='r',tokenizer_revision='r',dtype='float32',base_sha256='base')
    source=dict(training_path='training.json',training_sha256='x',training_complete_sha256='x',model_identity=model,
                pool_path='pool',pool_sha256='x',runtime=runtime,prompt_contract={},checkpoints={})
    windows=[[{'id':f'q{i}'} for i in range(8)] for _ in range(64)]
    monkeypatch.setattr(mod,'source',lambda path,root:(source,windows))
    spectral=dict(run_path='spectral-run',weights_dir='weights',inventory={'modules':{'module':{}}})
    audit={'sources':{'sft':{'identity':model}}};cache=dict(module_results=[])
    monkeypatch.setattr(stage0_spectral,'_load',lambda path,root:(spectral,audit))
    monkeypatch.setattr(stage0_spectral,'_verify_cache',lambda *args:cache)
    qual=dict(model_identity=model,decoding={},tokenizer_files={})
    qdir=tmp_path/'qual';qdir.mkdir();(qdir/'complete.json').write_text('{}')
    review_data=dict(diagnostic_sha256=file_hash(qdir/'complete.json'))
    review=qdir/'reviews'/_hash(review_data)/'qualification.json';review.parent.mkdir(parents=True)
    review.write_text(json.dumps(dict(qualified=True,config_sha256=_hash(qual),runtime=runtime,review=review_data)))
    monkeypatch.setattr(stage0_qualification,'_load',lambda path,root:(qual,qdir))
    monkeypatch.setattr(stage0_qualification,'_verify',lambda *args:dict(runtime=runtime))
    rows=[dict(id=f'q{i}',corpus='gsm8k' if i<150 else 'fineweb',input_ids=[1,2],attention_mask=[1,1],masks=dict(question=[True,False],solution=[False,True],user=[False,False])) for i in range(300)]
    for name in ['training','spectral','qualification']:(tmp_path/f'{name}.json').write_text('{}')
    (tmp_path/'runtime.json').write_text(json.dumps(runtime))
    prepared=dict(items_path='items.json',items_sha256=_hash(rows),source_evidence={'base_sha256':'base'},**{k:model[k] for k in ['model','model_revision','tokenizer_revision','dtype']})
    (tmp_path/'inputs.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(activation_prepare,'load_prepared',lambda *args:(prepared,rows))
    args=['--output-root',str(tmp_path)]
    assert main(['offline-grpo-internals-prepare',*args,'--training',str(tmp_path/'training.json'),'--spectral',str(tmp_path/'spectral.json'),'--qualification',str(tmp_path/'qualification.json'),'--inputs',str(tmp_path/'inputs.json'),'--runtime',str(tmp_path/'runtime.json'),'--name','fixture'])==0
    path=tmp_path/'plans/fixture/offline.internals.json'
    assert mod.load(path,tmp_path)[0]['source']==source
    review.write_text('{}')
    with pytest.raises(ValueError,match='source changed'):
        mod.load(path,tmp_path)


def test_public_drift_collection_resume_checks_actual_mapping(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_internals as mod,stage0_evaluation
    from pilot_eval.cli import main
    from test_stage0_drift import fixture_measurement
    runtime=dict(execution_kind='synthetic-cpu-fixture',session_id='fixture')
    refs={str(i):dict(path=f'checkpoint-{i}') for i in [0,8,16,32,64]}
    config=dict(run_path='internals',source=dict(model_identity={},prompt_contract={},checkpoints=refs),
                runtime=runtime,execution_kind='synthetic-cpu-fixture',modules=['module'])
    row=dict(id='q',corpus='gsm8k',input_ids=[1,2,3],attention_mask=[1,1,1],masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3))
    monkeypatch.setattr(mod,'load',lambda *args:(config,tmp_path/'internals',[],[row],None,None))
    monkeypatch.setattr(mod,'factors',lambda path:{'module':(np.array([[1.]]),np.array([[1.]]))})
    monkeypatch.setattr(mod,'_reserve',lambda *args:(dict(reservation_id='fixture'),60))
    class Backend:
        def runtime(self):return runtime
        def engine(self,config,root):self.config=config;return self
        def measure(self,rows,variant,*,diagnostic):return fixture_measurement(rows,self.config,variant,diagnostic)
        def check_integrity(self,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
        def close(self):pass
    path=tmp_path/'plan.json';path.write_text('{}');args=['--output-root',str(tmp_path),'--config',str(path),'--kind','drift']
    assert main(['offline-grpo-internals-collect',*args,'--max-new-units','1'],dependencies=Backend())==0
    for attempt in (tmp_path/'internals/drift/attempts').iterdir():
        for name in ['final-integrity.json','timing.json','timing.complete.json']:(attempt/name).unlink()
    assert main(['offline-grpo-internals-collect',*args],dependencies=Backend())==0
    assert main(['offline-grpo-internals-verify',*args])==0
    result=json.loads((tmp_path/'internals/drift/units/off8-000/result.json').read_text())
    assert result['metadata']['kl_direction']=='variant||untuned'
    arr=tmp_path/'internals/drift/units/off8-000/arrays.npz';arr.write_bytes(b'bad')
    assert main(['offline-grpo-internals-verify',*args])==1


def test_public_push_collection_validates_negative_gradient(tmp_path,monkeypatch):
    from pilot_eval import offline_grpo_internals as mod,stage0_evaluation
    from pilot_eval.cli import main
    from test_stage0_pushes import Gradients
    runtime=dict(execution_kind='synthetic-cpu-fixture',session_id='session-a',device='tiny CPU')
    config=dict(runtime=runtime,execution_kind='synthetic-cpu-fixture',run_path='internals')
    unit=dict(key='window-1',case='offline',objective='grpo',question_count=8,partial=False,
              sequences=[dict(advantage=0.,input_ids=[1,2],target_ids=[2])],token_count=1)
    collector=dict(modules=['module'],module_shapes={'module':dict(a=[1,64],b=[64,1])})
    monkeypatch.setattr(mod,'load',lambda *args:(config,tmp_path/'internals',[],[],None,None))
    monkeypatch.setattr(mod,'_collector',lambda *args:(collector,[unit]))
    monkeypatch.setattr(mod,'_reserve',lambda *args:(dict(reservation_id='fixture'),60))
    path=tmp_path/'plan.json';path.write_text('{}');args=['--output-root',str(tmp_path),'--config',str(path),'--kind','push']
    assert main(['offline-grpo-internals-collect',*args],dependencies=Gradients())==0
    assert main(['offline-grpo-internals-verify',*args])==0
    folder=tmp_path/'internals/push/units/window-1'
    with np.load(folder/'arrays.npz') as data:
        np.testing.assert_array_equal(data['push_0'],-data['gradient_b_0'])


def test_public_geometry_dense_norm_and_intruder_controls(tmp_path,monkeypatch):
    from pilot_eval import offline_grpo_internals as mod,stage0_weights
    from pilot_eval.stage0_spectral_math import decompose
    from pilot_eval.cli import main
    matrix=np.diag(np.linspace(2.,1.,40));basis,ranks=decompose(matrix)
    spectral_dir=tmp_path/'spectral';spectral_dir.mkdir();np.savez(spectral_dir/'basis.npz',**basis)
    row=dict(module='module',arrays_path='basis.npz',**ranks)
    spectral=dict(run_path='spectral',inventory={'modules':{'module':{'path':'weights','key':'module'}}})
    cache=dict(module_results=[row]);refs={str(i):dict(path=f'cp{i}') for i in [0,8,16,32,64]}
    config=dict(modules=['module'],source=dict(checkpoints=refs),online=None)
    monkeypatch.setattr(mod,'load',lambda *args:(config,tmp_path/'internals',[],[],spectral,cache))
    monkeypatch.setattr(mod,'factors',lambda path:{'module':(np.ones((1,40),dtype=np.float32),np.ones((40,1),dtype=np.float32)*(.0 if str(path).endswith('cp0') else .01))})
    monkeypatch.setattr(stage0_weights.LocalWeights,'matrix',lambda self,entry:matrix)
    path=tmp_path/'plan.json';path.write_text('{}');args=['--output-root',str(tmp_path),'--config',str(path)]
    assert main(['offline-grpo-internals-geometry',*args],dependencies=None)==0
    result=__import__('json').loads((tmp_path/'internals/geometry/units/000/result.json').read_text())
    assert result['norms']['0']==0 and result['norms']['64']==pytest.approx(.4)
    assert 'threshold_sweep' in result['geometry']['64']['levels']['50']


def test_report_reconstructs_observed_kl_and_matched_window(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_internals as mod,offline_grpo_training
    from pilot_eval.workflow import _hash
    from pilot_eval.training import file_hash
    from pilot_eval.run import _write_json
    runtime=dict(execution_kind='synthetic-cpu-fixture',session_id='fixture')
    refs={str(i):dict(path=f'cp{i}') for i in [0,8,16,32,64]}
    training_path=tmp_path/'train.json';training_path.write_text(json.dumps(dict(run_path='train',optimizer_groups=[])))
    config=dict(modules=['module'],execution_kind=runtime['execution_kind'],source=dict(checkpoints=refs,training_path='train.json'),online=None)
    base=tmp_path/'internals';folder=base/'geometry/units/000';folder.mkdir(parents=True)
    geom=dict(module='module',norms={str(i):float(i) for i in [0,8,16,32,64]},online_product_comparisons={'8':dict(cosine=.5),'16':dict(cosine=.25),'64':dict(cosine=.1)})
    _write_json(folder/'result.json',geom);(folder/'geometry.npz').write_bytes(b'geometry')
    _write_json(folder/'complete.json',dict(config_sha256=_hash(config),files={n:file_hash(folder/n) for n in ['result.json','geometry.npz']}))
    _write_json(base/'geometry/complete.json',dict(config_sha256=_hash(config),units={'0':file_hash(folder/'complete.json')}))
    for kind in ['push','drift']:_write_json(base/kind/'complete.json',{})
    row=dict(id='q',corpus='fineweb',input_ids=[1,2],attention_mask=[1,1],masks=dict(question=[False,False],solution=[False,False],user=[True,True]))
    windows=[[dict(id='q',draws=[dict(draw_id='draw',generated_token_ids=[2],completion_mask=[1])])] for _ in range(64)]
    monkeypatch.setattr(mod,'load',lambda *args:(config,base,windows,[row],None,None))
    monkeypatch.setattr(mod,'verify_collection',lambda *args:dict(complete=True))
    monkeypatch.setattr(mod,'_verified_checkpoints',lambda *args:{64:tmp_path/'cp64'})
    monkeypatch.setattr(mod,'factors',lambda path:{'module':(np.array([[1.]]),np.array([[1.]]))})
    drift_unit=dict(key='d',variant='off8',ids=['q'],diagnostic=False)
    push_units=[dict(key='batch'),dict(key='window-1')]
    monkeypatch.setattr(mod,'_collector',lambda c,k,w,r,root:({},[drift_unit] if k=='drift' else push_units))
    def read(c,kind,directory,unit,root,collector,rows,cache):
        if kind=='push':return {'push_0':np.array([[1.]]),'gradient_a_0':np.array([[0.]])},{}
        return dict(kl=np.array([.2]),write_sums=np.array([[[[0.,0.,0.]],[[0.,0.,0.]],[[1.,4.,1.]]]])),dict(metadata=dict(predictions=[dict(example=0,view='user')]))
    monkeypatch.setattr(mod,'_read_collection',read)
    logs=[]
    for step in range(1,65):
        diagnostic=tmp_path/f'diagnostic-{step}.json';
        from pilot_eval.offline_grpo_diagnostics import summarize
        mu=[[float(step)]];pi=[[0.]]
        diagnostic.write_text(json.dumps(dict(protocol='offline-logratio-v1',timing='before-update',draw_ids=['draw'],question_ids=['q'],generated_token_ids=[[2]],masks=[[1]],log_mu=mu,log_pi=pi,summary=summarize(mu,pi))))
        evidence={'logratio':{'path':diagnostic.name}}
        if step==1:
            push=tmp_path/'window.json';push.write_text(json.dumps(dict(independent_push=[0.,1.],parameter_names=['module.lora_A.default.weight','module.lora_B.default.weight'],criterion=dict(atol=1e-6,rtol=1e-4))))
            evidence['window1_push']={'path':push.name}
        logs.append(dict(diagnostic_evidence=evidence))
    monkeypatch.setattr(offline_grpo_training,'_accepted_history',lambda *args:([],[],logs))
    plan=tmp_path/'plan.json';plan.write_text('{}')
    result=mod.report(plan,tmp_path)
    assert result['drift'][0]['kl_token_mean']==pytest.approx(.2)
    assert result['drift'][0]['module_effective_write_ratio']['module']==pytest.approx(.5)
    assert result['preupdate']['9']['sequence_mean']==9 and result['adapter_frobenius_norms']['64']==64
    points=result['descriptive_similarities'][0]['points']
    assert [(p['checkpoint'],p['preupdate_window'],p['learner_updates']) for p in points]==[(8,9,8),(16,17,16)]
    assert any('checkpoint 64' in line for line in result['limitations'])
    assert result['push_alignment'][0]['offline']['cosine']==pytest.approx(1.)
    assert mod.verify_report(plan,tmp_path)['complete']
    (base/'report/report.md').write_text('tampered')
    with pytest.raises(ValueError,match='report seal changed'):mod.verify_report(plan,tmp_path)

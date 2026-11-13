"""Offline principal-cut contracts; CPU-only."""
import numpy as np
import pytest


def test_eight_conditions_match_existing_cut_definitions():
    from pilot_eval.offline_grpo_cuts import make_edits, CONDITIONS
    a=np.ones((1,40),dtype=np.float32);b=np.ones((40,1),dtype=np.float32)
    u=np.eye(40,dtype=np.float32)[:,:3];v=u.copy()
    edits,checks=make_edits(a,b,u,v)
    assert len(CONDITIONS)==8 and set(edits)==set(CONDITIONS)
    assert np.linalg.norm(u.T@edits['output'][1])<1e-6
    assert np.linalg.norm(v.T@edits['input'][0].T)<1e-6
    assert checks['output-magnitude']['update_norm']==pytest.approx(checks['output']['update_norm'],rel=1e-6)
    assert checks['input-magnitude']['update_norm']==pytest.approx(checks['input']['update_norm'],rel=1e-6)


def test_public_agreement_evaluation_resume_and_tamper(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_cuts as cuts
    from pilot_eval.cli import main
    plan=dict(name='fixture',run_path='run',execution_kind='synthetic-cpu-fixture',runtime={'session_id':'fixture'},
        decoding=dict(max_new_tokens=4,eos_token_id=99,pad_token_id=98),items=[dict(id=f'q{i}',prompt=f'prompt{i}',gold='#### 2') for i in range(150)])
    directory=tmp_path/'run'
    variants={name:dict(adapter_path=f'adapters/{name}',source_arm='offline',operation=name) for name in cuts.CONDITIONS}
    for name in cuts.CONDITIONS:
        target=tmp_path/'adapters'/name;target.mkdir(parents=True)
        for file in ['adapter_model.safetensors','adapter_config.json','complete.json']:(target/file).write_text(name)
    monkeypatch.setattr(cuts,'load',lambda *args:(plan,directory,None,None))
    calls={'verify_build':0}
    def verified_build(*args,**kwargs):
        calls['verify_build']+=1
        return dict(variants=variants)
    monkeypatch.setattr(cuts,'verify_build',verified_build)
    monkeypatch.setattr(cuts,'_reservation',lambda *args:(dict(reservation_id='fixture'),60))
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):return [7]
    class Engine:
        tokenizer=Tokenizer()
        def runtime(self):return plan['runtime']
        def engine(self,config,root):self.config=config;self.inputs={};return self
        def prepare_inputs(self,rows):
            self.inputs={r['id']:[3] for r in rows}
            return [dict(id=r['id'],input_ids=[3],attention_mask=[1]) for r in rows]
        def compare(self,rows,variant,tokens):
            error=.01 if variant=='input' else 0.
            return dict(max_absolute_error=error,examples=[dict(id=row['id'],errors=[error]) for row in rows])
        def generate(self,rows,variant,seed):
            return [dict(text='#### 2',token_ids=[7],generated_token_ids=[7,99],stop_reason='eos') for _ in rows]
        def check_integrity(self,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
        def close(self):pass
    path=tmp_path/'plan.json';path.write_text('{}')
    args=['--output-root',str(tmp_path),'--config',str(path)]
    assert main(['offline-grpo-cuts-agreement',*args,'--max-new-units','2'],dependencies=Engine())==0
    for attempt in (directory/'attempts').iterdir():
        for name in ['final-integrity.json','timing.json','timing.complete.json']:(attempt/name).unlink()
    assert main(['offline-grpo-cuts-agreement',*args],dependencies=Engine())==0
    result=cuts.verify_stage(path,tmp_path,'agreement')
    assert result['batch_sizes']['input']==1 and result['batch_sizes']['output']==16
    assert main(['offline-grpo-cuts-evaluate',*args,'--max-new-units','2'],dependencies=Engine())==0
    assert main(['offline-grpo-cuts-evaluate',*args],dependencies=Engine())==0
    calls['verify_build']=0
    assert cuts.verify_stage(path,tmp_path,'evaluation')['responses']==1200
    assert calls['verify_build']==1
    calls['verify_build']=0
    assert main(['offline-grpo-cuts-report',*args])==0
    assert calls['verify_build']==1
    result=json.loads((directory/'report/results.json').read_text())
    assert len(result['transitions'])==1200 and not result['original_endpoint_reused']
    assert main(['offline-grpo-cuts-verify',*args,'--stage','report'])==0
    payload=directory/'evaluation/units/output-0000/result.json';payload.write_text('{}')
    with pytest.raises(ValueError,match='seal changed'):
        cuts.verify_stage(path,tmp_path,'evaluation')


def test_public_build_resume_reconstructs_all_condition_payloads(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_cuts as cuts, stage0_sources
    from pilot_eval.cli import main
    from pilot_eval.stage0_edits import _save_tensors
    from pilot_eval.trace_readout_assets import read_weight
    a=np.ones((1,40),dtype=np.float32);b=np.ones((40,1),dtype=np.float32)
    source=tmp_path/'checkpoint';source.mkdir();(source/'adapter_config.json').write_text('{}')
    _save_tensors(source/'adapter_model.safetensors',{'module.lora_A.weight':a,'module.lora_B.weight':b})
    spectral_dir=tmp_path/'spectral';spectral_dir.mkdir()
    basis=np.eye(40,dtype=np.float32);np.savez(spectral_dir/'basis.npz',u=basis,v=basis)
    cache=dict(module_results=[dict(module='module',k50=3,arrays_path='basis.npz')])
    spectral=dict(run_path='spectral');directory=tmp_path/'run'
    plan=dict(name='fixture',source={'checkpoints':{'64':{'path':'checkpoint'}}},modules=['module'])
    monkeypatch.setattr(cuts,'load',lambda *args:(plan,directory,spectral,cache))
    def factors(path):
        target=path/'adapter_model.safetensors'
        return {'module':(read_weight(target,'module.lora_A.weight'),read_weight(target,'module.lora_B.weight'))}
    monkeypatch.setattr(stage0_sources,'factors',factors)
    path=tmp_path/'plan.json';path.write_text('{}');args=['--output-root',str(tmp_path),'--config',str(path)]
    assert main(['offline-grpo-cuts-build',*args,'--max-new-units','0'])==0
    assert main(['offline-grpo-cuts-build',*args])==0
    calls=[]
    original_make=cuts.make_edits
    def tracked_make(*args):
        calls.append(True)
        return original_make(*args)
    monkeypatch.setattr(cuts,'make_edits',tracked_make)
    result=cuts.verify_build(path,tmp_path)
    assert not calls
    cuts.verify_build(path,tmp_path,deep=True)
    assert len(calls)==1
    assert set(result['variants'])==set(cuts.CONDITIONS)
    (directory/'build/adapters/input/adapter_model.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError,match='payload changed'):
        cuts.verify_build(path,tmp_path)


def test_public_prepare_and_load_bind_reference_and_sources(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_cuts as cuts,offline_grpo_internals,offline_grpo_behaviour
    from pilot_eval.cli import main
    for name in ['internals','behaviour','reference']:(tmp_path/f'{name}.json').write_text('{}')
    identity=dict(model='m',model_revision='r',tokenizer_revision='r',dtype='float32',base_sha256='base')
    source=dict(training_sha256='train',model_identity=identity,prompt_contract={},checkpoints={})
    runtime=dict(session_id='session',execution_kind='synthetic-cpu-fixture')
    internal=dict(source=source,runtime=runtime,weights_dir='weights',decoding={},modules=['module'],spectral_path='spectral',execution_kind='synthetic-cpu-fixture')
    items=[dict(id=f'q{i}',prompt='q',gold='#### 2') for i in range(150)]
    behaviour=dict(training_sha256='train',items=items)
    cache=dict(module_results=[dict(module='module',k50=3)])
    monkeypatch.setattr(offline_grpo_internals,'load',lambda *args:(internal,None,None,None,{},cache))
    monkeypatch.setattr(offline_grpo_behaviour,'verify',lambda *args:(behaviour,[]))
    monkeypatch.setattr(cuts,'_reference',lambda *args:(dict(model_identity=identity),{'module':3}))
    args=['--output-root',str(tmp_path),'--internals',str(tmp_path/'internals.json'),'--behaviour',str(tmp_path/'behaviour.json'),
          '--reference',str(tmp_path/'reference.json'),'--reference-root',str(tmp_path),'--name','test']
    assert main(['offline-grpo-cuts-prepare',*args])==0
    plan=tmp_path/'plans/test/offline.cuts.json'
    assert cuts.load(plan,tmp_path)[0]['conditions']==list(cuts.CONDITIONS)
    assert json.loads(plan.read_text())['reference_root']=='.'
    (tmp_path/'reference.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='reference changed'):cuts.load(plan,tmp_path)


def test_historical_reference_accepts_only_bound_basis_omission(tmp_path):
    import hashlib
    import json
    from pilot_eval import offline_grpo_cuts as cuts
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    directory=tmp_path/'historical'
    plan=dict(protocol='sampling-principal-cuts-v1')
    plan_path=tmp_path/'plans/historical.json'
    save(plan_path,plan)
    implementation=cuts._identity()
    code={'sampling_stage0_cuts.py':'06ba73389170885440ddaa06f395cc7a5fc34acf2b1b322631d0315f05fb9f9a',
          'sampling_stage0_cut_agreement.py':'be42c96985fb00a2110baa9bf3a5461305b4e5b8668b31d523abe1c5c4381b9b',
          'stage0_edit_math.py':implementation['edit_math'],'stage0_spectral_math.py':implementation['spectral_math']}
    variants={f'{arm}-{name}':dict(operation=name) for arm in ['k0','k2','k5'] for name in cuts.CONDITIONS}
    config=dict(protocol='sampling-principal-cut-evaluation-v2',run_path='historical',implementation=code,
                plan_sha256=_hash(plan),variants=variants)
    save(directory/'config.json',config)
    save(directory/'cut-sources.json',dict(protocol='sampling-principal-cut-sources-v1',plan_sha256=_hash(plan),plan_path='plans/historical.json'))
    save(directory/'inputs/items.json',[])
    save(directory/'meta/final-integrity.json',dict(base_unchanged=True,adapters_unchanged=True))
    (directory/'results').mkdir(parents=True);(directory/'results/responses.jsonl').write_text('')
    save(directory/'results/results.json',{})
    save(directory/'complete.json',dict(config_sha256=_hash(config),items_sha256=file_hash(directory/'inputs/items.json'),
        final_integrity_sha256=file_hash(directory/'meta/final-integrity.json'),responses_sha256=file_hash(directory/'results/responses.jsonl'),
        results_sha256=file_hash(directory/'results/results.json')))
    build_config=dict(identity=_hash(plan),modules=['module'])
    save(directory/'build/build-config.json',build_config)
    unit=directory/'build/units'/hashlib.sha256(b'module').hexdigest();unit.mkdir(parents=True)
    (unit/'factors.npz').write_bytes(b'fixture')
    save(unit/'checks.json',dict(spectrum=dict(k50=3)))
    omitted_digest='a'*64
    save(unit/'complete.json',dict(config_sha256=_hash(build_config),module='module',files={
        'factors.npz':file_hash(unit/'factors.npz'),'checks.json':file_hash(unit/'checks.json'),'bases.npz':omitted_digest}))
    omission=str((unit/'bases.npz').relative_to(directory))
    save(directory/'meta/omitted-files.json',dict(files={omission:omitted_digest},reason='fixture'))
    _,ranks=cuts._reference(directory/'config.json',tmp_path)
    assert ranks=={'module':3}
    save(directory/'meta/omitted-files.json',dict(files={omission:'b'*64}))
    with pytest.raises(ValueError,match='matching sealed omission'):
        cuts._reference(directory/'config.json',tmp_path)
    plan_path.unlink()
    with pytest.raises(FileNotFoundError):
        cuts._reference(directory/'config.json',tmp_path)

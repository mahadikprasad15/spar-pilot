"""GRPO fixed-input preparation through the approved public workflow seam."""
import json
import pytest
from pilot_eval.cli import main
from pilot_eval.activation_prepare import load_prepared
from test_grpo_training import frozen, train, Dependencies, Engine
from pilot_eval.activation_prepare import PROJECTIONS


class MatchedEngine(Engine):
    def check_integrity(self):
        return dict(base_before="b" * 64, base_after="b" * 64, base_unchanged=True)

    def save_checkpoint(self, directory, step):
        super().save_checkpoint(directory, step)
        import numpy as np
        from safetensors.numpy import save_file
        factors = {}
        for layer in range(28):
            for projection in PROJECTIONS:
                stem = f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
                factors[stem+'.lora_A.weight'] = np.array([[3.,4.]],dtype=np.float32)
                factors[stem+'.lora_B.weight'] = np.array([[0.],[float(step)],[-float(step)]],dtype=np.float32)
        save_file(factors,str(directory/'adapter_model.safetensors'))
        (directory / "adapter_config.json").write_text(json.dumps(dict(
            r=1, lora_alpha=1, lora_dropout=0, target_modules=PROJECTIONS)))


class MatchedDependencies(Dependencies):
    def load_training(self, *args):
        return MatchedEngine(self)


def source(root):
    assert train(root, frozen(root), MatchedDependencies()) == 0
    return root / "plans/arm/grpo.training.json"


def test_grpo_preparation_reuses_frozen_sequences_without_tokenizing(tmp_path):
    training = source(tmp_path)
    train = json.loads(training.read_text())
    frozen = json.loads((tmp_path / train['frozen_path']).read_text())
    plan = json.loads((tmp_path / frozen['source_path']).read_text())
    inputs = tmp_path / plan['sources']['measurement']['path']
    # This source is an existing Pilot 3 execution, not an SFT training config.
    args = ['grpo-writes-prepare', '--config', str(training), '--measurement-config', str(inputs),
            '--name', 'grpo-writes', '--output-root', str(tmp_path)]
    assert main(args) == 0
    path = tmp_path / 'plans/grpo-writes/activation.prepared.json'
    config, rows = load_prepared(path, tmp_path)
    old, original = load_prepared(inputs, tmp_path)
    assert rows == original
    assert config['items_sha256'] == old['items_sha256']
    assert config['source_contract']['kind'] == 'grpo'
    assert config['source_contract']['training_path'] == str(training.relative_to(tmp_path))
    assert config['source_evidence']['checkpoints']['64']['path'].startswith(train['run_path'])
    before = path.read_bytes()
    assert main(args) == 0
    assert path.read_bytes() == before


def test_random_control_matches_every_module_norm_without_changing_source_or_rng(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file, load_file
    from pilot_eval.grpo_random_control import construct_control
    source_dir = tmp_path / 'trained'
    source_dir.mkdir()
    factors = {}
    for layer in range(28):
        for projection in PROJECTIONS:
            stem = f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            factors[stem+'.lora_A.weight'] = np.array([[3.,4.]],dtype=np.float32)
            factors[stem+'.lora_B.weight'] = np.array([[0.],[2.],[-1.]],dtype=np.float32)
    factors['base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight'] *= 0
    save_file(factors,str(source_dir/'adapter_model.safetensors'))
    (source_dir/'adapter_config.json').write_text(json.dumps(dict(r=1,lora_alpha=1,lora_dropout=0,
                                                              target_modules=PROJECTIONS)))
    before={p:p.read_bytes() for p in source_dir.iterdir()}
    np.random.seed(19)
    expected=np.random.random(3)
    np.random.seed(19)
    result=construct_control(source_dir,tmp_path/'control')
    assert np.array_equal(np.random.random(3),expected)
    saved=load_file(str(tmp_path/'control/adapter_model.safetensors'))
    assert len(result['modules'])==196
    for key,A in factors.items():
        if '.lora_A.' not in key:continue
        B=factors[key.replace('.lora_A.','.lora_B.')]
        a,b=saved[key],saved[key.replace('.lora_A.','.lora_B.')]
        assert np.linalg.norm(a)==pytest.approx(1.,rel=1e-6)
        # Independent literal: ||[3,4]|| * ||[0,2,-1]|| = sqrt(125).
        target=0. if not np.any(B) else np.sqrt(125.)
        assert np.linalg.norm(b@a)==pytest.approx(target,rel=1e-6)
    construct_control(source_dir,tmp_path/'other')
    other=load_file(str(tmp_path/'other/adapter_model.safetensors'))
    assert all(np.array_equal(saved[k],other[k]) for k in saved)
    assert all(p.read_bytes()==value for p,value in before.items())


def test_public_random_control_preparation_retains_real_zero_and_separate_identity(tmp_path):
    training=source(tmp_path)
    config=json.loads(training.read_text())
    frozen=json.loads((tmp_path/config['frozen_path']).read_text())
    plan=json.loads((tmp_path/frozen['source_path']).read_text())
    assert main(['grpo-writes-prepare','--config',str(training),'--measurement-config',
                 str(tmp_path/plan['sources']['measurement']['path']), '--name','writes','--output-root',str(tmp_path)])==0
    original=tmp_path/'plans/writes/activation.prepared.json'
    args=['grpo-writes-control','--config',str(original),'--name','random-control','--output-root',str(tmp_path)]
    assert main(args)==0
    path=tmp_path/'plans/random-control/activation.prepared.json'
    prepared,rows=load_prepared(path,tmp_path)
    expected,original_rows=load_prepared(original,tmp_path)
    assert rows==original_rows
    assert prepared['checkpoint_steps']==[0,64]
    assert prepared['source_contract']['kind']=='random-rank1-control'
    assert prepared['source_evidence']['checkpoints']['0']==expected['source_evidence']['checkpoints']['0']
    assert prepared['source_evidence']['checkpoints']['64']!=expected['source_evidence']['checkpoints']['64']
    assert main(args)==0


def test_control_profile_and_resumable_measurement_use_only_zero_and_control(tmp_path):
    training=source(tmp_path)
    train_config=json.loads(training.read_text())
    frozen=json.loads((tmp_path/train_config['frozen_path']).read_text())
    plan=json.loads((tmp_path/frozen['source_path']).read_text())
    from pilot_eval.grpo_writes import prepare_grpo_writes,prepare_random_writes
    prepared=prepare_grpo_writes(training,tmp_path/plan['sources']['measurement']['path'],tmp_path,'writes')
    control=prepare_random_writes(prepared,tmp_path,'random')
    from pilot_eval.activation_profile import profile_activation, freeze_execution
    from test_activation_measurement import MeasurementDependencies, MeasurementEngine
    class CalibrationEngine(MeasurementEngine):
        def capture_reference(self, rows, *, padded_width=None):
            return super().capture_reference(rows)
    class CalibrationDependencies(MeasurementDependencies):
        def activation_engine(self, config, root):
            self.engine_loads += 1
            return CalibrationEngine(self)
    deps=CalibrationDependencies()
    from pilot_eval.activation_calibration import prepare_calibration,collect_calibration,freeze_rule,validate_rule
    calibration=prepare_calibration(control,tmp_path,name='random-calibration',dependencies=deps)
    collect_calibration(calibration,tmp_path,dependencies=deps)
    freeze_rule(calibration,tmp_path,review_notes='Reviewed independent envelope.')
    validate_rule(calibration,tmp_path,dependencies=deps)
    profile=profile_activation(control,tmp_path,dependencies=deps,calibration=calibration)
    assert profile['identity']['checkpoint_steps']==[0,64]
    execution=freeze_execution(control,tmp_path,profile=profile['profile_path'],batch_size=16,
                               name='random-execution',review_notes='Reviewed independent control profile.')
    from pilot_eval.activation_measurement import measure_activation,verify_measurement
    interrupted=MeasurementDependencies(fail_at=3)
    with pytest.raises(RuntimeError,match='controlled'):
        measure_activation(execution,tmp_path,dependencies=interrupted)
    resumed=MeasurementDependencies()
    result=measure_activation(execution,tmp_path,dependencies=resumed)
    assert result['completed_combinations']==38
    assert {s for _,s in resumed.measure_calls}=={0,64}
    assert verify_measurement(execution,tmp_path)==result
    assert measure_activation(execution,tmp_path,dependencies=resumed)==result
    assert main(['activation-report','--config',str(execution),'--name','control-report','--output-root',str(tmp_path)])==0
    report=json.loads((tmp_path/'reports/control-report/results/results.json').read_text())
    assert report['checkpoint_steps']==[0,64]
    assert {r['step'] for r in report['measurements']}=={0,64}
    assert report['variant_labels']['64'].startswith('random control')
    for label,path in [('grpo',prepared),('sft',tmp_path/plan['sources']['measurement']['path'])]:
        cal=prepare_calibration(path,tmp_path,name=label+'-calibration',dependencies=deps)
        collect_calibration(cal,tmp_path,dependencies=deps)
        freeze_rule(cal,tmp_path,review_notes='Reviewed independent arm calibration.')
        validate_rule(cal,tmp_path,dependencies=deps)
        measured_profile=profile_activation(path,tmp_path,dependencies=deps,calibration=cal)
        execution_path=freeze_execution(path,tmp_path,profile=measured_profile['profile_path'],batch_size=16,
                                       name=label+'-execution',review_notes='Reviewed matching batches.')
        measure_activation(execution_path,tmp_path,dependencies=MeasurementDependencies())
        assert main(['activation-report','--config',str(execution_path),'--name',label+'-report','--output-root',str(tmp_path)])==0
    args=['grpo-writes-report','--grpo-report',str(tmp_path/'reports/grpo-report'),
          '--control-report',str(tmp_path/'reports/control-report'),'--sft-report',str(tmp_path/'reports/sft-report'),
          '--name','writes-comparison','--output-root',str(tmp_path)]
    assert main(args)==0
    compared=json.loads((tmp_path/'reports/writes-comparison/results/results.json').read_text())
    assert compared['random_realizations']==1
    assert len(compared['cosines'])==504
    assert all(row['cosine']==pytest.approx(1.) for row in compared['cosines'])
    assert main(args)==0
    (tmp_path/'reports/control-report/results/mean-vectors.npz').write_bytes(b'corrupt')
    assert main(args)==1


@pytest.mark.parametrize('fault',['missing-module','nonfinite','extra-head','wrong-rank'])
def test_random_control_rejects_invalid_source_before_publishing(tmp_path,fault):
    import numpy as np
    from safetensors.numpy import save_file
    from pilot_eval.grpo_random_control import construct_control
    folder=tmp_path/'source';folder.mkdir()
    values={}
    for layer in range(28):
        for projection in PROJECTIONS:
            stem=f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            values[stem+'.lora_A.weight']=np.ones((1,2),dtype=np.float32)
            values[stem+'.lora_B.weight']=np.ones((3,1),dtype=np.float32)
    key=next(iter(values))
    if fault=='missing-module':del values[key]
    elif fault=='nonfinite':values[key][0,0]=np.nan
    elif fault=='extra-head':values['lm_head.lora_A.weight']=np.ones((1,2),dtype=np.float32)
    else:values[key]=np.ones((2,2),dtype=np.float32)
    save_file(values,str(folder/'adapter_model.safetensors'))
    (folder/'adapter_config.json').write_text(json.dumps(dict(r=1,lora_alpha=1,lora_dropout=0)))
    with pytest.raises(ValueError):construct_control(folder,tmp_path/'control')
    assert not (tmp_path/'control/complete.json').exists()


def test_real_random_control_passes_all_196_instrument_hooks_and_restores_model(tmp_path):
    torch=pytest.importorskip('torch',exc_type=ImportError)
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model
    from pilot_eval.activation_engine import ActivationEngine
    from pilot_eval.sft_backend import frozen_weight_hash
    from pilot_eval.grpo_random_control import construct_control
    torch.set_num_threads(1);torch.manual_seed(2)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=8,hidden_size=8,
        intermediate_size=16,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=32,pad_token_id=0,attn_implementation='eager')),
        LoraConfig(r=1,lora_alpha=1,lora_dropout=0,target_modules=PROJECTIONS,task_type='CAUSAL_LM'))
    zero=tmp_path/'zero';trained=tmp_path/'trained'
    model.save_pretrained(zero,save_embedding_layers=False)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.fill_(.01)
    model.save_pretrained(trained,save_embedding_layers=False)
    construct_control(trained,tmp_path/'random')
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    base=frozen_weight_hash(model)
    engine=ActivationEngine(model=model,expected_base_hash=base)
    rows=[dict(id='fixed:1',input_ids=[1,2,3],attention_mask=[1,1,1],
               masks={'question':[True,False,False],'solution':[False,True,True],'user':[False,False,False]})]
    try:
        reference=engine.capture_reference(rows)
        assert engine.measure(zero,reference,step=0)['validation']['exact_zero']
        measured=engine.measure(tmp_path/'random',reference,step=64)
        assert measured['validation']['module_count']==196
        assert measured['validation']['rank1_passed'] and measured['validation']['reference_invariant']
        assert measured['arrays']['block_delta_norm_sum'].sum()>0
    finally:engine.close()
    assert frozen_weight_hash(model)==base
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_grpo_writes_rejects_base_mismatch_and_changed_frozen_tokens(tmp_path):
    from test_grpo_training import Engine as OldEngine
    class Mismatched(MatchedDependencies):
        def load_training(self,*args):return OldEngine(self)
    path=frozen(tmp_path)
    assert train(tmp_path,path,Mismatched())==0
    training=tmp_path/'plans/arm/grpo.training.json'
    frozen_config=json.loads(path.read_text())
    plan=json.loads((tmp_path/frozen_config['source_path']).read_text())
    args=['grpo-writes-prepare','--config',str(training),'--measurement-config',
          str(tmp_path/plan['sources']['measurement']['path']),'--name','writes','--output-root',str(tmp_path)]
    assert main(args)==1
    assert not (tmp_path/'plans/writes/prepare-complete.json').exists()
    measurement=json.loads((tmp_path/plan['sources']['measurement']['path']).read_text())
    inputs=tmp_path/measurement['items_path']
    rows=json.loads(inputs.read_text());rows[0]['input_ids'][0]+=1;inputs.write_text(json.dumps(rows))
    assert main(args)==1
    assert not (tmp_path/'plans/writes/prepare-complete.json').exists()


def test_random_control_cannot_silently_round_a_nonzero_target_to_zero(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    from pilot_eval.grpo_random_control import construct_control
    source_dir=tmp_path/'tiny-norm';source_dir.mkdir()
    values={}
    for layer in range(28):
        for projection in PROJECTIONS:
            stem=f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            values[stem+'.lora_A.weight']=np.full((1,2),1e-30,dtype=np.float32)
            values[stem+'.lora_B.weight']=np.full((3,1),1e-30,dtype=np.float32)
    save_file(values,str(source_dir/'adapter_model.safetensors'))
    (source_dir/'adapter_config.json').write_text(json.dumps(dict(r=1,lora_alpha=1,lora_dropout=0)))
    with pytest.raises(ValueError,match='norm'):
        construct_control(source_dir,tmp_path/'control')
    assert not (tmp_path/'control/complete.json').exists()


def test_grpo_profile_records_loading_warmup_saving_and_source_verification_cost(tmp_path):
    training=source(tmp_path)
    train_config=json.loads(training.read_text())
    frozen=json.loads((tmp_path/train_config['frozen_path']).read_text())
    plan=json.loads((tmp_path/frozen['source_path']).read_text())
    from pilot_eval.grpo_writes import prepare_grpo_writes
    prepared=prepare_grpo_writes(training,tmp_path/plan['sources']['measurement']['path'],tmp_path,'timed-writes')
    from pilot_eval.activation_profile import profile_activation
    from test_activation_measurement import MeasurementDependencies
    result=profile_activation(prepared,tmp_path,dependencies=MeasurementDependencies())
    assert result['source_verification_wall_seconds']>=0
    assert result['profile_wall_seconds']>0
    for row in result['measurements']:
        assert row['loading_wall_seconds']>=0 and row['warmup_wall_seconds']>=0
        assert row['saving_wall_seconds']>=0 and row['candidate_wall_seconds']>0
        assert row['saved_summary_bytes']>0

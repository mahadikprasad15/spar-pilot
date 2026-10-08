"""Supplemental measurement contract: independent small distributions and masks."""
import numpy as np
import pytest


def test_next_target_views_and_full_vocabulary_kl_have_known_answers():
    from pilot_eval.token_measurement_math import prediction_positions, full_vocabulary_kl, summarize_kl
    rows = [dict(id='one', input_ids=[9, 4, 5, 6], attention_mask=[1]*4,
                 masks={'question':[False,True,False,False],
                        'solution':[False,False,True,True], 'user':[False]*4}),
            dict(id='two', input_ids=[9, 7], attention_mask=[1,1],
                 masks={'question':[False]*2,'solution':[False]*2,'user':[False,True]})]
    positions = prediction_positions(rows)
    assert positions == [dict(example=0,context=0,target=1,view='question',token_id=4),
                         dict(example=0,context=1,target=2,view='solution',token_id=5),
                         dict(example=0,context=2,target=3,view='solution',token_id=6),
                         dict(example=1,context=0,target=1,view='user',token_id=7)]
    p, q = np.log([[.8,.2],[.5,.5]]), np.log([[.5,.5],[.8,.2]])
    values = full_vocabulary_kl(p,q)
    np.testing.assert_allclose(values, [.19274475702175753,.22314355131420976],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(full_vocabulary_kl(p,p),[0.,0.])
    # Different lengths make the two weighting definitions observably distinct.
    summary=summarize_kl([dict(example=0,view='solution')]*2+[dict(example=1,view='solution')],
                         np.array([0.,0.,.6]),['one','two'])
    cell=next(r for r in summary if r['view']=='solution')
    assert cell['token_mean']==pytest.approx(.2)
    assert cell['example_mean']==pytest.approx(.3)
    assert cell['tokens']==3 and cell['examples']==2


def tiny_instrument(tmp_path,layer_count=2):
    torch=pytest.importorskip('torch',exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from pilot_eval.activation_engine import ActivationEngine
    from pilot_eval.activation_prepare import PROJECTIONS
    from pilot_eval.sft_backend import frozen_weight_hash
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=8,
        intermediate_size=12,num_hidden_layers=layer_count,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=32,pad_token_id=0,attn_implementation='eager')),
        LoraConfig(r=1,lora_alpha=1,lora_dropout=0,target_modules=PROJECTIONS))
    base=frozen_weight_hash(model)
    model.save_pretrained(tmp_path/'zero',save_embedding_layers=False)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.fill_(.03)
    model.save_pretrained(tmp_path/'trained',save_embedding_layers=False)
    engine=ActivationEngine(model=model,expected_layers=layer_count,expected_base_hash=base)
    rows=[dict(id='one',input_ids=[1,2,3,4],attention_mask=[1]*4,
               masks={'question':[False,True,False,False],'solution':[False,False,True,True],'user':[False]*4}),
          dict(id='two',input_ids=[1,5,6],attention_mask=[1]*3,
               masks={'question':[False]*3,'solution':[False]*3,'user':[False,True,True]})]
    return torch,engine,rows


@pytest.mark.parametrize("layer_count",[2,28])
def test_real_supplement_reconstructs_branches_and_matches_independent_full_logits(tmp_path,layer_count):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    from pilot_eval.token_measurement_math import full_vocabulary_kl
    torch,instrument,rows=tiny_instrument(tmp_path,layer_count)
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1_000_000)
    # A normal full model call is allowed only in this tiny independent oracle.
    tokens=torch.tensor([[1,2,3,4],[1,5,6,0]])
    attention=torch.tensor([[1,1,1,1],[1,1,1,0]])
    with torch.no_grad():
        with instrument.model.disable_adapter():
            q=instrument.model(input_ids=tokens,attention_mask=attention).logits.numpy()
        p=instrument.model(input_ids=tokens,attention_mask=attention).logits.numpy()
    seen={};handles=[]
    for layer,projection,module in instrument.modules:
        def observe(m,args,output,key=(layer,projection)):
            seen[key]=(args[0].detach().numpy().copy(),output.detach().numpy().copy(),
                       m.weight.detach().numpy().copy())
        handles.append(module.lora_A['default'].register_forward_hook(observe))
    try:
        with engine.integrity_scope():
            result=engine.measure(tmp_path/'trained',rows,step=64)
        positions=result['metadata']['predictions']
        expected=full_vocabulary_kl(np.array([p[r['example'],r['context']] for r in positions]),
                                    np.array([q[r['example'],r['context']] for r in positions]))
        from pilot_eval.activation_profile import AGREEMENT
        # Full-head and chunked-head FP32 matmuls follow the existing policy.
        np.testing.assert_allclose(result['arrays']['kl'],expected,**AGREEMENT)
        assert result['arrays']['coefficients'].shape==(5,7*layer_count)
        for mi,module in enumerate(result['metadata']['modules']):
            x,c,a=seen[module['layer'],module['projection']]
            for ci,position in enumerate(result['metadata']['coefficient_positions']):
                wanted=np.dot(a.astype(np.float64)[0],x[position['example'],position['position']].astype(np.float64))
                assert result['arrays']['coefficients'][ci,mi]==pytest.approx(wanted,abs=1e-6,rel=1e-5)
        assert result['validation']['rank1_passed'] and result['validation']['reference_invariant']
        assert result['validation']['max_logit_contexts']<=2
        with engine.integrity_scope():
            zero=engine.measure(tmp_path/'zero',rows,step=0)
        assert np.count_nonzero(zero['arrays']['kl'])==0
        assert np.count_nonzero(zero['arrays']['coefficients'])>0
        assert zero['validation']['zero_contribution']
    finally:
        for handle in handles:handle.remove()
        engine.close()


def test_supplement_rejects_workspace_before_output_head_allocation(tmp_path):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    _,instrument,rows=tiny_instrument(tmp_path)
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1)
    try:
        with pytest.raises(ValueError,match='workspace'):
            engine.measure(tmp_path/'trained',rows,step=64)
    finally:engine.close()


def make_supplement_sources(root):
    import json
    from test_grpo_writes import source
    from pilot_eval.grpo_writes import prepare_grpo_writes
    from pilot_eval.activation_profile import profile_activation, freeze_execution
    from test_activation_measurement import MeasurementDependencies
    training=source(root)
    train=json.loads(training.read_text());frozen=json.loads((root/train['frozen_path']).read_text())
    plan=json.loads((root/frozen['source_path']).read_text())
    sft_prepared=root/plan['sources']['measurement']['path']
    grpo_prepared=prepare_grpo_writes(training,sft_prepared,root,'grpo-inputs')
    outputs=[]
    for label,path in [('sft',sft_prepared),('grpo',grpo_prepared)]:
        profile=profile_activation(path,root,dependencies=MeasurementDependencies())
        outputs.append(freeze_execution(path,root,profile=profile['profile_path'],batch_size=16,
                       name=label+'-execution',review_notes='Reviewed fixture source.'))
    return outputs


class ControlledTokens:
    """Controlled model boundary; saved-workflow code remains real."""
    def __init__(self,deps,settings):
        assert deps.live_engines==0, 'two full models would coexist'
        deps.live_engines+=1
        self.deps=deps;self.settings=settings
    def integrity_scope(self):
        from contextlib import nullcontext
        self.deps.integrity_scopes+=1
        return nullcontext({'base_unchanged':True})
    def close(self):
        self.deps.closed+=1
        self.deps.live_engines-=1
    def reference(self,rows):return {}
    def release_reference(self,reference):reference.clear()
    def measure(self,checkpoint,rows,*,step,context_chunk=None,reference=None):
        from pilot_eval.activation_prepare import PROJECTIONS,VIEWS
        from pilot_eval.token_measurement_math import prediction_positions
        self.deps.calls.append((tuple(r['id'] for r in rows),step))
        if len(self.deps.calls)==self.deps.fail_at:raise KeyboardInterrupt('controlled interruption')
        positions=[dict(example=i,position=t,view=v,token_id=row['input_ids'][t])
                   for i,row in enumerate(rows) for t in range(len(row['input_ids']))
                   for v in VIEWS if row['masks'][v][t]]
        predicted=prediction_positions(rows)
        modules=[dict(layer=i,projection=p,A_sha256='a'*64,B_sha256='b'*64,
                      scale=1.,A_norm=1.,B_norm=float(step)) for i in range(28) for p in PROJECTIONS]
        return dict(arrays=dict(coefficients=np.ones((len(positions),196),np.float32),
                               kl=np.full(len(predicted),step/1000,np.float64)),
                    metadata=dict(example_ids=[r['id'] for r in rows],coefficient_positions=positions,
                        predictions=predicted,modules=modules,step=step,
                        coefficient_convention='raw-Ax-adapted-input',kl_direction='tuned||untuned',
                        kl_units='nats',logit_precision='float32',reduction_precision='float64'),
                    validation=dict(rank1_passed=True,reference_invariant=True,module_count=196,
                                    zero_contribution=True if step==0 else None,rank1_max_fraction=0.,
                                    max_logit_contexts=min(context_chunk or self.settings['context_chunk'],len(predicted)),
                                    bounded_workspace_estimate=100))


class TokenDependencies:
    def __init__(self,fail_at=None):
        self.fail_at=fail_at;self.calls=[];self.closed=0;self.live_engines=0;self.integrity_scopes=0
    def runtime(self):
        from test_activation_measurement import MeasurementDependencies
        return MeasurementDependencies().runtime()
    def token_engine(self,prepared,root,settings):return ControlledTokens(self,settings)
    def synchronize(self):pass
    def peak_memory(self):return {'allocated_bytes':0,'reserved_bytes':0}
    def reset_peak_memory(self):pass


def test_public_supplement_workflow_freezes_profiles_resumes_and_reports_without_touching_sources(tmp_path):
    import json
    from pilot_eval.token_measurement_workflow import prepare_tokens,profile_tokens,freeze_tokens,measure_tokens,verify_tokens,report_tokens
    sft,grpo=make_supplement_sources(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    prepared=prepare_tokens(sft,grpo,tmp_path,'tokens',context_chunk=8,workspace_bytes=1_000_000)
    deps=TokenDependencies()
    profile_tokens(prepared,tmp_path,dependencies=deps)
    execution=freeze_tokens(prepared,tmp_path,review_notes='Reviewed supplemental cost and new checks.')
    with pytest.raises(KeyboardInterrupt,match='controlled'):
        measure_tokens(execution,tmp_path,dependencies=TokenDependencies(fail_at=8))
    run=tmp_path/json.loads(execution.read_text())['run_path']
    assert not (run/'complete.json').exists()
    resumed=TokenDependencies()
    result=measure_tokens(execution,tmp_path,dependencies=resumed)
    # 19 frozen batches × 5 checkpoints × 2 arms; one sealed batch survives.
    assert result['completed_units']==190
    assert resumed.integrity_scopes==37
    assert resumed.live_engines==0
    assert len(resumed.calls)==185
    assert verify_tokens(execution,tmp_path)['completed_units']==190
    assert measure_tokens(execution,tmp_path,dependencies=resumed)==result
    assert len(resumed.calls)==185
    report=report_tokens(execution,tmp_path,'token-report')
    assert report['report_complete'] and len(report['kl_trajectory'])==30
    assert next(r for r in report['kl_trajectory'] if r['arm']=='grpo' and r['step']==64 and r['view']=='solution')['token_mean']==pytest.approx(.064)
    assert all(p.read_bytes()==data for p,data in before.items())
    corrupt=next(run.glob('shards/*/*/arrays.npz'))
    corrupt.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash'):
        verify_tokens(execution,tmp_path)


def test_supplement_commands_are_exposed_without_importing_gpu_libraries(capsys):
    from pilot_eval.cli import main
    with pytest.raises(SystemExit) as result:main(['--help'])
    assert result.value.code==0
    help_text=capsys.readouterr().out
    assert all(name in help_text for name in ['tokens-prepare','tokens-profile','tokens-freeze',
                                              'tokens-measure','tokens-verify','tokens-report'])


def test_coefficients_rescale_but_distribution_and_branch_do_not(tmp_path):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    from pilot_eval.activation_profile import AGREEMENT
    torch,instrument,rows=tiny_instrument(tmp_path)
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1_000_000)
    try:
        with engine.integrity_scope():first=engine.measure(tmp_path/'trained',rows,step=64)
        with torch.no_grad():
            for _,_,module in instrument.modules:
                module.lora_A['default'].weight.mul_(2)
                module.lora_B['default'].weight.mul_(.5)
        instrument.model.save_pretrained(tmp_path/'equivalent',save_embedding_layers=False)
        with engine.integrity_scope():second=engine.measure(tmp_path/'equivalent',rows,step=64,context_chunk=1)
        np.testing.assert_allclose(second['arrays']['coefficients'],2*first['arrays']['coefficients'],**AGREEMENT)
        np.testing.assert_allclose(second['arrays']['kl'],first['arrays']['kl'],**AGREEMENT)
        assert all(b['B_norm']==pytest.approx(a['B_norm']/2) for a,b in zip(first['metadata']['modules'],second['metadata']['modules']))
    finally:engine.close()


def test_wrong_signed_coefficient_is_detected_independently(tmp_path):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    _,instrument,rows=tiny_instrument(tmp_path)
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1_000_000)
    handle=instrument.modules[0][2].lora_A['default'].register_forward_hook(lambda m,args,output:-output)
    try:
        with pytest.raises(ValueError,match='signed input coefficient'):
            with engine.integrity_scope():engine.measure(tmp_path/'trained',rows,step=64)
    finally:
        handle.remove();engine.close()


def test_valid_adapter_replacement_is_rejected_against_frozen_source_hash(tmp_path):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    from pilot_eval.training import file_hash
    _,instrument,rows=tiny_instrument(tmp_path)
    original=file_hash(tmp_path/'trained/adapter_model.safetensors')
    # Both files are valid rank-1 checkpoints. Shape/finite checks alone would
    # accept the wrong checkpoint; the frozen source identity must stop it.
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1_000_000,
        checkpoint_hashes={str((tmp_path/'trained').resolve()):original})
    (tmp_path/'trained/adapter_model.safetensors').write_bytes((tmp_path/'zero/adapter_model.safetensors').read_bytes())
    try:
        with pytest.raises(ValueError,match='checkpoint source hash'):
            with engine.integrity_scope():engine.measure(tmp_path/'trained',rows,step=64)
    finally:engine.close()

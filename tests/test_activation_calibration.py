"""Calibration is separate from validation and cannot authorize corrupt measurements."""
import json
import numpy as np
import pytest
from test_activation_validation import prepared
from test_activation_profile import ProfileDependencies, ProfileEngine


def test_calibration_metric_uses_vector_norm_and_measured_resolution():
    from pilot_eval.activation_calibration import comparison_metrics, fit_rule, check_agreement
    from pilot_eval.activation_profile import ARRAYS
    counts = np.ones((2, 3, 28))
    modules = np.ones((2, 3, 28, 7))
    a = {key: (modules.copy() if key.startswith('module_') else counts.copy()) for key in ARRAYS}
    a['block_base_sum'] = np.zeros((2, 3, 28, 2)); a['block_base_sum'][..., 0] = 1
    a['block_delta_sum'] = a['block_base_sum'].copy()
    b = {key: value.copy() for key, value in a.items()}
    b['block_delta_sum'][..., 1] = 1e-6
    metrics = comparison_metrics({8:a}, {8:b})
    assert metrics['errors']['block_delta_vector'] == pytest.approx(1e-6)
    rule = fit_rule([metrics], margin=3)
    assert check_agreement({8:a}, {8:b}, rule)['passed']
    wrong = {key: value.copy() for key, value in a.items()}
    wrong['block_delta_sum'] *= -1
    assert not check_agreement({8:a}, {8:wrong}, rule)['passed']
    wrong['block_delta_sum'] *= -2
    wrong['block_delta_norm_sum'] *= 2
    assert not check_agreement({8:a}, {8:wrong}, rule)['passed']
    wrong['block_count'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='count'):
        comparison_metrics({8:a}, {8:wrong})


class CalibrationEngine(ProfileEngine):
    def capture_reference(self, rows, *, padded_width=None):
        if padded_width:
            self.deps.padded_calls.append(padded_width)
        return super().capture_reference(rows)

class CalibrationDeps(ProfileDependencies):
    def __init__(self):
        super().__init__()
        self.padded_calls = []
    def activation_engine(self, config, root):
        return CalibrationEngine(self)


def test_calibration_flow_freezes_disjoint_cohorts_and_requires_independent_validation(tmp_path):
    from pilot_eval.activation_calibration import prepare_calibration, collect_calibration, freeze_rule, validate_rule, load_validated_rule
    config = prepared(tmp_path)
    before = config.read_bytes()
    deps = CalibrationDeps()
    plan = prepare_calibration(config, tmp_path, name='noise-v2', dependencies=deps)
    frozen = json.loads(plan.read_text())
    assert len(frozen['cohorts']['calibration']) == len(frozen['cohorts']['validation']) == 16
    assert not set(frozen['cohorts']['calibration']) & set(frozen['cohorts']['validation'])
    assert not set(frozen['excluded_profile_ids']) & set(sum(frozen['cohorts'].values(), []))
    with pytest.raises(ValueError, match='freeze'):
        validate_rule(plan, tmp_path, dependencies=deps)
    collect_calibration(plan, tmp_path, dependencies=deps)
    calls = len(deps.batch_calls)
    collect_calibration(plan, tmp_path, dependencies=deps)
    assert len(deps.batch_calls) == calls
    assert deps.padded_calls
    with pytest.raises(ValueError, match='review'):
        freeze_rule(plan, tmp_path, review_notes='')
    freeze_rule(plan, tmp_path, review_notes='Reviewed calibration evidence and the 3x engineering margin.')
    validated = validate_rule(plan, tmp_path, dependencies=deps)
    assert validated['validated_batches'] == [1, 2, 4, 8, 16]
    rule = load_validated_rule(plan, tmp_path, prepared=json.loads(config.read_text()), runtime=deps.runtime())
    assert rule['protocol'] == 'pilot3-agreement-v2'
    assert config.read_bytes() == before
    target = tmp_path / frozen['run_path'] / 'calibration/batch-2/step-8.npz'
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        load_validated_rule(plan, tmp_path, prepared=json.loads(config.read_text()), runtime=deps.runtime())


def test_real_instrument_supports_controlled_right_padding_without_counting_it(tmp_path):
    torch = pytest.importorskip('torch')
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import get_peft_model, LoraConfig
    from pilot_eval.activation_engine import ActivationEngine, InstrumentFailure
    from pilot_eval.activation_prepare import PROJECTIONS
    from pilot_eval.sft_backend import frozen_weight_hash
    torch.set_num_threads(1)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=4,
        intermediate_size=8,num_hidden_layers=2,num_attention_heads=1,num_key_value_heads=1,
        max_position_embeddings=16,pad_token_id=0,attn_implementation='eager')),
        LoraConfig(r=1,lora_alpha=1,lora_dropout=0,target_modules=PROJECTIONS,task_type='CAUSAL_LM'))
    checkpoint = tmp_path/'adapter'; model.save_pretrained(checkpoint)
    engine = ActivationEngine(model=model,expected_layers=2,expected_base_hash=frozen_weight_hash(model))
    rows=[{'id':'one','input_ids':[1,2],'attention_mask':[1,1],
           'masks':{'question':[True,False],'solution':[False,True],'user':[False,False]}}]
    try:
        reference=engine.capture_reference(rows,padded_width=4)
        assert reference['inputs']['input_ids'].tolist()==[[1,2,0,0]]
        assert reference['inputs']['attention_mask'].tolist()==[[1,1,0,0]]
        assert reference['inputs']['position_ids'].tolist()==[[0,1,1,1]]
        assert reference['masks']['question'].tolist()==[[True,False,False,False]]
        result=engine.measure(checkpoint,reference,step=0)
        assert result['validation']['reference_invariant'] and result['validation']['exact_zero']
        np.testing.assert_array_equal(result['arrays']['block_count'],[[[1,1],[1,1],[0,0]]])
        with pytest.raises(InstrumentFailure,match='padding width'):
            engine.capture_reference(rows,padded_width=1)
    finally: engine.close()


def test_calibrated_profile_freeze_and_execution_verify_retain_provenance(tmp_path):
    from pilot_eval.activation_calibration import prepare_calibration,collect_calibration,freeze_rule,validate_rule
    from pilot_eval.activation_profile import profile_activation,freeze_execution
    from pilot_eval.activation_measurement import load_execution
    config=prepared(tmp_path); deps=CalibrationDeps()
    plan=prepare_calibration(config,tmp_path,name='noise-v2',dependencies=deps)
    collect_calibration(plan,tmp_path,dependencies=deps)
    freeze_rule(plan,tmp_path,review_notes='Reviewed noise envelope.')
    validate_rule(plan,tmp_path,dependencies=deps)
    profile=profile_activation(config,tmp_path,dependencies=deps,calibration=plan)
    assert profile['identity']['agreement']['protocol']=='pilot3-agreement-v2'
    assert profile['profile_path'].split('/')[-1].startswith('profile-v2-')
    execution=freeze_execution(config,tmp_path,profile=profile['profile_path'],batch_size=8,
                               name='production-v2',review_notes='Reviewed independent validation.')
    saved,_,_=load_execution(execution,tmp_path)
    assert saved['calibration_path']==str(plan.relative_to(tmp_path))
    assert saved['agreement']['protocol']=='pilot3-agreement-v2'


def test_heldout_drift_is_rejected_and_weak_directions_are_flagged():
    from pilot_eval.activation_calibration import comparison_metrics, fit_rule, check_agreement
    from pilot_eval.activation_profile import ARRAYS
    a={k:np.ones((1,3,28,7) if k.startswith('module_') else (1,3,28)) for k in ARRAYS}
    a['block_base_sum']=np.ones((1,3,28,2))
    a['block_delta_sum']=np.full((1,3,28,2),1e-8)
    b={k:v.copy() for k,v in a.items()}; b['block_delta_sum'][...,0]+=1e-6
    metric=comparison_metrics({8:a},{8:b})
    rule=fit_rule([metric],margin=3)
    assert check_agreement({8:a},{8:b},rule)['coverage']['directions']>0
    b['block_delta_sum'][...,0]+=1e-4
    assert not check_agreement({8:a},{8:b},rule)['passed']
    bad=dict(rule); bad['thresholds']=dict(rule['thresholds']); bad['thresholds']['block_delta_vector']=99
    with pytest.raises(ValueError,match='envelope'):
        check_agreement({8:a},{8:b},bad)


def test_calibration_cli_prepares_and_requires_review(tmp_path):
    from pilot_eval.cli import main
    config=prepared(tmp_path); deps=CalibrationDeps()
    assert main(['activation-calibration-prepare','--config',str(config),'--name','cli-noise',
                 '--output-root',str(tmp_path)],dependencies=deps)==0
    plan=tmp_path/'plans/cli-noise/activation.calibration.json'
    assert main(['activation-calibrate','--config',str(plan),'--output-root',str(tmp_path)],dependencies=deps)==0
    assert main(['activation-calibration-freeze','--config',str(plan),'--review-notes','Reviewed',
                 '--output-root',str(tmp_path)],dependencies=deps)==0
    assert main(['activation-calibration-validate','--config',str(plan),'--output-root',str(tmp_path)],dependencies=deps)==0


def test_negative_controls_reject_a_rule_too_wide_to_detect_wrong_science(tmp_path):
    from pilot_eval.activation_calibration import _negative_controls, comparison_metrics, fit_rule
    from pilot_eval.activation_profile import ARRAYS, STEPS
    a={k:np.ones((1,3,28,7) if k.startswith('module_') else (1,3,28)) for k in ARRAYS}
    a['block_base_sum']=np.ones((1,3,28,2)); a['block_delta_sum']=a['block_base_sum'].copy()
    reference={s:{k:v.copy() for k,v in a.items()} for s in STEPS}
    for k in ARRAYS:
        if 'delta' in k: reference[0][k]*=0
    rule=fit_rule([comparison_metrics(reference,reference)],margin=3)
    assert all(_negative_controls(reference,rule).values())
    for k in rule['envelope']: rule['envelope'][k]=10.;rule['thresholds'][k]=30.
    rule['resolution']={k:30. for k in rule['resolution']}
    with pytest.raises(ValueError,match='negative control'):
        _negative_controls(reference,rule)


def test_report_direction_resolution_distinguishes_weak_and_resolved():
    from pilot_eval.activation_report import direction_resolution
    rule={'resolution':{'block_delta_vector':1e-4}}
    assert direction_resolution(1e-5,rule)['direction_resolved'] is False
    assert direction_resolution(1e-3,rule)['direction_resolved'] is True
    assert direction_resolution(None,rule)['direction_resolved'] is None
    assert direction_resolution(1e-3,{})['direction_resolved'] is None


def test_padding_validation_failure_persists_rejection_and_never_seals(tmp_path):
    from pilot_eval.activation_calibration import prepare_calibration, collect_calibration, freeze_rule, validate_rule
    class BrokenPaddingEngine(CalibrationEngine):
        def capture_reference(self, rows, *, padded_width=None):
            self.padded=bool(padded_width)
            return super().capture_reference(rows,padded_width=padded_width)
        def measure(self,checkpoint,reference,*,step):
            result=super().measure(checkpoint,reference,step=step)
            if self.deps.break_padding and self.padded and step>0:
                result['arrays']['block_delta_norm_sum']+=100*(result['arrays']['block_count']>0)
            return result
    class Deps(CalibrationDeps):
        break_padding=False
        def activation_engine(self,config,root): return BrokenPaddingEngine(self)
    deps=Deps(); config=prepared(tmp_path)
    plan=prepare_calibration(config,tmp_path,name='padding-control',dependencies=deps)
    collect_calibration(plan,tmp_path,dependencies=deps)
    freeze_rule(plan,tmp_path,review_notes='Reviewed')
    deps.break_padding=True
    with pytest.raises(ValueError,match='padding validation'):
        validate_rule(plan,tmp_path,dependencies=deps)
    target=tmp_path/json.loads(plan.read_text())['run_path']/'validated'
    assert not (target/'complete.json').exists()
    assert json.loads((target/'results.json').read_text())['status']=='failed'

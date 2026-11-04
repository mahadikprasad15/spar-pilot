import pytest

from pilot_eval.offline_grpo_diagnostics import summarize


def test_sequence_sums_ratio_of_totals_and_finite_negatives():
    result = summarize([[0.0, -2.0], [-1.0]], [[-1.0, -1.0], [-2.0]])
    assert result['sequence_sums'] == [0.0, 1.0]
    assert result['sequence_mean'] == 0.5
    assert result['token_rate'] == pytest.approx(1 / 3)
    negative = summarize([[-2.0]], [[-1.0]])
    assert negative['sequence_mean'] == -1
    assert negative['token_rate'] == -1


def test_reject_misaligned_and_nonfinite():
    with pytest.raises(ValueError): summarize([[0.0]], [[0.0, 1.0]])
    with pytest.raises(ValueError): summarize([[float('nan')]], [[0.]])


def test_preflight_receipt_requires_bound_runtime_and_real_observations(tmp_path):
    import json
    from pilot_eval.offline_grpo_training import _verify_preflight
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    config = {'runtime': {'device': 'fixture'}}
    folder=tmp_path/'diagnostics'; folder.mkdir()
    row=dict(eos=True,capped=False,generated_token_ids=[1,2],generation_log_probs=[-1.,-2.],
             teacher_forced_log_probs=[-1.,-2.],differences=[0.,0.])
    receipt=dict(protocol='offline-grpo-diagnostic-preflight-v1',config_sha256=_hash(config),
        execution_kind='real-cuda',runtime=config['runtime'],prompt_count=1,fidelity_tolerance=.5,
        fidelity={'long_min_tokens':2,'max_new_tokens':4,'eos_token_id':2,'observations':[row]},
        smoke={'finite_gradients':True,'same_input_gradient_match':True,'optimizer_steps':0,
               'adapter_unchanged':True,'base_unchanged':True})
    path=folder/'preflight.json';path.write_text(json.dumps(receipt))
    (folder/'preflight.complete.json').write_text(json.dumps({'config_sha256':_hash(config),'sha256':file_hash(path)}))
    assert _verify_preflight(tmp_path,config)==receipt
    receipt['fidelity']['observations'][0]['differences'][0]=1.
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='seal'):_verify_preflight(tmp_path,config)
    (folder/'preflight.complete.json').write_text(json.dumps({'config_sha256':_hash(config),'sha256':file_hash(path)}))
    with pytest.raises(ValueError, match='differences'):_verify_preflight(tmp_path,config)
    receipt['fidelity']['observations'][0]['differences'][0]=0.
    receipt['fidelity']['observations'][0]['capped']=True
    path.write_text(json.dumps(receipt))
    (folder/'preflight.complete.json').write_text(json.dumps({'config_sha256':_hash(config),'sha256':file_hash(path)}))
    with pytest.raises(ValueError, match='cap'):_verify_preflight(tmp_path,config)


def test_fidelity_gate_uses_only_an_explicit_positive_tolerance():
    from pilot_eval.offline_grpo_diagnostics import fidelity_gate
    fidelity={'observations':[dict(differences=[0.,-.25]),dict(differences=[.5])]}
    assert fidelity_gate(fidelity,.5)==.5
    with pytest.raises(ValueError,match='exceeds'):fidelity_gate(fidelity,.4)
    for bad in [None,0,-1.,float('nan'),True]:
        with pytest.raises(ValueError,match='tolerance'):fidelity_gate(fidelity,bad)


def test_sealed_preflight_above_its_tolerance_is_rejected(tmp_path):
    import json
    from pilot_eval.offline_grpo_training import _verify_preflight
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    config={'runtime':{'device':'fixture'}}
    folder=tmp_path/'diagnostics';folder.mkdir()
    row=dict(eos=True,capped=False,generated_token_ids=[1,2],generation_log_probs=[-1.,-1.],
             teacher_forced_log_probs=[-2.,-1.],differences=[1.,0.])
    receipt=dict(protocol='offline-grpo-diagnostic-preflight-v1',config_sha256=_hash(config),
        execution_kind='real-cuda',runtime=config['runtime'],prompt_count=1,fidelity_tolerance=.5,
        fidelity={'long_min_tokens':2,'max_new_tokens':4,'eos_token_id':2,'observations':[row]},
        smoke={'finite_gradients':True,'same_input_gradient_match':True,'optimizer_steps':0,
               'adapter_unchanged':True,'base_unchanged':True})
    path=folder/'preflight.json';path.write_text(json.dumps(receipt))
    (folder/'preflight.complete.json').write_text(json.dumps({'config_sha256':_hash(config),'sha256':file_hash(path)}))
    with pytest.raises(ValueError,match='exceeds'):_verify_preflight(tmp_path,config)


def test_fidelity_sampling_overrides_the_model_generation_config():
    import inspect
    from pilot_eval.offline_grpo_diagnostics import measure_generation_scores
    source = inspect.getsource(measure_generation_scores)
    assert 'repetition_penalty=1.0' in source and 'no_repeat_ngram_size=0' in source

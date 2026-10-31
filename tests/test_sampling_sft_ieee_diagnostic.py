import pytest


def test_source_inspection_rejects_ambiguous_or_missing_precision_mechanism(tmp_path):
    from pilot_eval.sampling_sft_ieee_diagnostic import inspect_sources
    v=tmp_path/'vllm'; t=tmp_path/'triton'
    (v/'attention/ops').mkdir(parents=True); (t/'language').mkdir(parents=True)
    (v/'attention/ops/triton_unified_attention.py').write_text('x = tl.dot(Q, K)\ny = tl.dot(P, V)\n')
    (t/'language/core.py').write_text('input_precision = knobs.language.fp32_default or "tf32"\n')
    (t/'knobs.py').write_text('fp32_default = env_opt_str("TRITON_F32_DEFAULT")\n')
    result=inspect_sources(v,t,'0.11.1','3.5.0')
    assert result['unqualified_dots']==2
    assert len(result['files'])==3
    (t/'language/core.py').write_text('pass\n')
    with pytest.raises(ValueError, match='precision mechanism'): inspect_sources(v,t,'0.11.1','3.5.0')
    (t/'language/core.py').write_text('input_precision = knobs.language.fp32_default or "tf32"\n')
    (v/'attention/ops/triton_unified_attention.py').write_text('x = tl.dot(Q, K, input_precision="tf32")\n')
    with pytest.raises(ValueError, match='unqualified'): inspect_sources(v,t,'0.11.1','3.5.0')


def test_child_rejects_unsealed_plan_before_loading_gpu(tmp_path, monkeypatch):
    from pilot_eval.sampling_sft_ieee_diagnostic import configure_child
    monkeypatch.delenv('TRITON_F32_DEFAULT', raising=False)
    with pytest.raises(ValueError, match='precision plan'):
        configure_child(tmp_path/'missing.json', tmp_path)
    import os
    assert os.environ.get('TRITON_F32_DEFAULT') is None


def test_precision_cli_rejects_missing_plan_before_job_launch(tmp_path, capsys):
    from pilot_eval.cli import main
    code=main(['sampling-sft-instrument-run','--output-root',str(tmp_path),
               '--config',str(tmp_path/'missing-config'),'--job',str(tmp_path/'missing-job'),
               '--precision-plan',str(tmp_path/'missing-plan'),'--precision-sha256','x'])
    assert code==1
    assert 'precision plan' in capsys.readouterr().err.lower()


def test_child_and_worker_evidence_bind_environment_and_sources(tmp_path, monkeypatch):
    import json, os, sys, types
    from pilot_eval import sampling_sft_ieee_diagnostic as ieee
    from pilot_eval import sampling_sft_engine as engine
    from pilot_eval.training import file_hash
    for key in ['TRITON_F32_DEFAULT','TRITON_CACHE_DIR','VLLM_CACHE_ROOT',
                'SAMPLING_SFT_IEEE_PLAN','SAMPLING_SFT_IEEE_HASH']:
        monkeypatch.delenv(key,raising=False)
    sources={'vllm_version':'0.11.1','triton_version':'3.5.0','files':{}}
    monkeypatch.setattr(ieee,'installed_sources',lambda:sources)
    monkeypatch.setattr(engine,'open_engine',engine.open_engine)
    cache=tmp_path/'cache';cache.mkdir()
    plan=tmp_path/'precision.json'
    plan.write_text(json.dumps(dict(precision='ieee',triton_cache=str(cache),sources=sources,cache_initially_empty=True,
        diagnostic_code_sha256=file_hash(ieee.__file__))))
    monkeypatch.setattr(ieee.SamplingSFTWorkerExtension,'sampling_sft_worker_evidence',lambda self:{'gpu':'fake'})
    monkeypatch.setitem(sys.modules,'triton',types.SimpleNamespace(knobs=types.SimpleNamespace(
        language=types.SimpleNamespace(fp32_default='ieee'))))
    # Child must configure before importing Triton.
    sys.modules.pop('triton')
    ieee.configure_child(plan,tmp_path,file_hash(plan))
    assert os.environ['TRITON_F32_DEFAULT']=='ieee'
    assert os.environ['TRITON_CACHE_DIR']==str(cache)
    sys.modules['triton']=types.SimpleNamespace(knobs=types.SimpleNamespace(
        language=types.SimpleNamespace(fp32_default='ieee')))
    evidence=ieee.IEEEWorkerExtension().sampling_sft_worker_evidence()['triton_precision']
    assert evidence['plan_sha256']==file_hash(plan)
    assert evidence['effective_knob']=='ieee' and evidence['sources']==sources
    assert evidence['vllm_cache']==str(tmp_path/'vllm-cache')
    sys.modules['triton'].knobs.language.fp32_default='tf32'
    with pytest.raises(ValueError,match='Worker precision'):
        ieee.IEEEWorkerExtension().sampling_sft_worker_evidence()


def test_execution_identity_separately_binds_precision_plan(tmp_path):
    from pilot_eval.sampling_sft_ieee_diagnostic import execution_identity
    plan=tmp_path/'precision.json';plan.write_text('{"precision":"ieee"}')
    result=execution_identity('legacy-runtime',plan)
    assert result['legacy_runtime_hash']=='legacy-runtime'
    assert result['precision_plan_sha256'] != 'legacy-runtime'
    assert result['production_qualified'] is False


def test_child_rejects_prepopulated_compiler_cache(tmp_path, monkeypatch):
    import json
    from pilot_eval import sampling_sft_ieee_diagnostic as ieee
    from pilot_eval.training import file_hash
    sources={'test':'sources'}
    monkeypatch.setattr(ieee,'installed_sources',lambda:sources)
    cache=tmp_path/'cache';cache.mkdir();(cache/'old-kernel').write_text('stale')
    plan=tmp_path/'precision.json'
    plan.write_text(json.dumps(dict(precision='ieee',triton_cache=str(cache),sources=sources,
        cache_initially_empty=True,diagnostic_code_sha256=file_hash(ieee.__file__))))
    with pytest.raises(ValueError,match='isolated Triton cache'):
        ieee.configure_child(plan,tmp_path,file_hash(plan))

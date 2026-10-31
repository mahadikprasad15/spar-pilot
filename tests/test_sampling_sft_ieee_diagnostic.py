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

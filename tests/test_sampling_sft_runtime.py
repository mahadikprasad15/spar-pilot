import json
from pilot_eval import sampling_sft_runtime as runtime
from test_sampling_sft_instrument import setup, source_template


def test_prepare_explicit_settings_preserves_source_and_refuses_overwrite(tmp_path,source_template,monkeypatch):
    _,old=setup(tmp_path,source_template)
    before=json.loads(old.read_text());before['runtime'].pop('triton_precision');old.write_text(json.dumps(before))
    sources={'vllm_version':'0.11.1','triton_version':'fixture','unqualified_dots':2,
        'files':{k:{'path':'/fixture/'+k,'sha256':'a'*64} for k in ['vllm_attention','triton_core','triton_knobs']}}
    monkeypatch.setattr(runtime,'installed_sources',lambda:sources)
    target=tmp_path/'plans/ieee/settings.json'
    assert runtime.main(['--output-root',str(tmp_path),'--input',str(old),'--output',str(target)])==0
    new=json.loads(target.read_text())
    assert new['runtime']['triton_precision']==dict(mode='ieee',cache_policy='isolated-per-engine',sources=sources)
    assert json.loads(old.read_text())==before
    assert runtime.main(['--output-root',str(tmp_path),'--input',str(target),'--output',str(old)])==1
    assert json.loads(old.read_text())==before

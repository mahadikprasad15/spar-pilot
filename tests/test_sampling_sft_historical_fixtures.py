import json
import pytest
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash
from pilot_eval.sampling_sft_qualification import historical_fixtures


def test_sealed_historical_capture_can_be_reused_without_reinterpreting_implementation(tmp_path):
    folder=tmp_path/'runs/source/instrument/old';folder.mkdir(parents=True)
    path=tmp_path/'plans/old/sampling-sft.instrument.json';path.parent.mkdir(parents=True)
    source=tmp_path/'plans/source.json';source.write_text('{}')
    plan=dict(protocol='sampling-sft-bounded-instrument-v1',source_plan='plans/source.json',
              settings=dict(mode='fixtures',runtime={'old':True}),runtime_hash=_hash({'old':True}),
              implementation={'old_release':'hash'},parents={},run_path=str(folder.relative_to(tmp_path)))
    path.write_text(json.dumps(plan))
    (folder/'plan-seal.json').write_text(json.dumps({'config_sha256':_hash(plan)}))
    (folder/'fixtures.json').write_text('[]')
    (folder/'requests.jsonl').write_text('{}\n')
    (folder/'complete.json').write_text(json.dumps(dict(config_sha256=_hash(plan),files={
        'fixtures.json':file_hash(folder/'fixtures.json'),'requests.jsonl':file_hash(folder/'requests.jsonl')})))
    with pytest.raises(ValueError,match='source'):
        historical_fixtures(tmp_path,path,source)
    audit=tmp_path/'runs/source/results/source-audit.json';audit.parent.mkdir(parents=True);audit.write_text('{}')
    source.write_text(json.dumps({'run_path':'runs/source'}))
    plan['parents']={'plans/source.json':file_hash(source),str(audit.relative_to(tmp_path)):file_hash(audit)}
    path.write_text(json.dumps(plan))
    (folder/'plan-seal.json').write_text(json.dumps({'config_sha256':_hash(plan)}))
    seal=json.loads((folder/'complete.json').read_text());seal['config_sha256']=_hash(plan)
    (folder/'complete.json').write_text(json.dumps(seal))
    assert historical_fixtures(tmp_path,path,source)==folder/'fixtures.json'
    (folder/'requests.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='seal'):historical_fixtures(tmp_path,path,source)

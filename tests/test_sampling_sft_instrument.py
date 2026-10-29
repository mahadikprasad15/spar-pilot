"""Public saved bounded workflow; injected dependencies cannot qualify a GPU."""
import json
import shutil

import pytest

from pilot_eval.cli import main
from test_sampling_sft_sources import fixture_sources, prepare, command
from test_sampling_sft_sampler import Engine, Tokenizer, generation


@pytest.fixture(scope='module')
def source_template(tmp_path_factory):
    root=tmp_path_factory.mktemp('sampler-source')
    selection,_,_=fixture_sources(root)
    plan=prepare(root,selection)
    assert command(root,'audit','--config',plan)==0
    return root


def setup(tmp_path,source_template):
    shutil.copytree(source_template,tmp_path,dirs_exist_ok=True)
    runtime=dict(vllm_version='0.11.1',torch_version='2.9.0',cuda='12.8',driver='fixture',
        gpu='NVIDIA L40S',dtype='float32',tf32=False,worker_numeric_flags=dict(matmul_tf32=False,
        cudnn_tf32=False,matmul_precision='highest'),max_num_seqs=32,max_num_batched_tokens=8192,
        gpu_memory_utilization=0.85,prefix_caching=True,max_model_len=16384,
        top_k=-1,min_p=0.0,presence_penalty=0.0,frequency_penalty=0.0,actual_stop_ids=[3,4])
    settings=dict(runtime=runtime,mode='bounded',k=0,problem_ids=[0],
                  local_model='inputs/model',local_tokenizer='inputs/tokenizer')
    # CPU fixtures use the same local-file inventory contract without loading weights.
    for directory in ['model','tokenizer']:
        path=tmp_path/'inputs'/directory;path.mkdir();(path/'config.json').write_text('{}')
    path=tmp_path/'settings.json';path.write_text(json.dumps(settings))
    return tmp_path/'plans/sample/sampling-sft.plan.json',path


def cli(root,operation,*args,deps=None):
    return main(['sampling-sft-instrument-'+operation,'--output-root',str(root),*map(str,args)],dependencies=deps)


class Dependencies:
    def __init__(self): self.engine=Engine([generation([3],'stop',3)])
    def open(self,config,root):return self.engine,Tokenizer(),dict(execution_kind='cpu-fixture')


def test_public_bounded_workflow_restores_and_rejects_changed_payload(tmp_path,source_template):
    source,settings=setup(tmp_path,source_template)
    assert cli(tmp_path,'prepare','--config',source,'--settings',settings,'--name','bounded')==0
    plan=tmp_path/'plans/bounded/sampling-sft.instrument.json'
    deps=Dependencies()
    assert cli(tmp_path,'run','--config',plan,deps=deps)==0
    assert cli(tmp_path,'run','--config',plan,deps=Dependencies())==0
    assert cli(tmp_path,'verify','--config',plan)==0
    folder=tmp_path/'runs/sampling-sft/sample/instrument/bounded'
    result=json.loads((folder/'results.json').read_text())
    assert result['execution_kind']=='cpu-fixture' and not result['gpu_qualified']
    assert len(result['traces'])==1 and result['traces'][0]['end_state']=='finished'
    assert (folder/'requests.jsonl').exists() and (folder/'blocks/000.json').exists()
    (folder/'requests.jsonl').write_text('tampered')
    assert cli(tmp_path,'verify','--config',plan)==1


def test_no_implicit_engine_defaults_and_no_unreserved_real_import(tmp_path,source_template):
    source,settings=setup(tmp_path,source_template)
    data=json.loads(settings.read_text());del data['runtime']['top_k'];settings.write_text(json.dumps(data))
    assert cli(tmp_path,'prepare','--config',source,'--settings',settings,'--name','bad')==1
    data['runtime']['top_k']=-1;settings.write_text(json.dumps(data))
    assert cli(tmp_path,'prepare','--config',source,'--settings',settings,'--name','bounded')==0
    plan=tmp_path/'plans/bounded/sampling-sft.instrument.json'
    # Refuse before importing torch/vLLM or starting a model.
    assert cli(tmp_path,'run','--config',plan)==1
    assert not (tmp_path/'runs/sampling-sft/sample/instrument/bounded/started.json').exists()


def test_interrupted_or_failed_attempt_is_not_silently_reexecuted(tmp_path,source_template):
    source,settings=setup(tmp_path,source_template)
    assert cli(tmp_path,'prepare','--config',source,'--settings',settings,'--name','bounded')==0
    plan=tmp_path/'plans/bounded/sampling-sft.instrument.json'
    deps=Dependencies();deps.engine=Engine([generation([]),generation([])])
    assert cli(tmp_path,'run','--config',plan,deps=deps)==1
    assert cli(tmp_path,'run','--config',plan,deps=Dependencies())==1
    folder=tmp_path/'runs/sampling-sft/sample/instrument/bounded'
    assert len(json.loads((folder/'results.json').read_text())['errors'])==2
    assert cli(tmp_path,'status','--config',plan)==0


def test_profile_is_bounded_records_batch_dependence_and_cannot_freeze_mock_gpu(tmp_path,source_template):
    source,settings=setup(tmp_path,source_template)
    data=json.loads(settings.read_text());data.update(mode='profile',k=2,problem_ids=list(range(64)))
    settings.write_text(json.dumps(data))
    assert cli(tmp_path,'prepare','--config',source,'--settings',settings,'--name','profile')==0
    plan=tmp_path/'plans/profile/sampling-sft.instrument.json'
    deps=Dependencies();deps.engine=Engine([generation([3],'stop',3)]*(64*4))
    assert cli(tmp_path,'run','--config',plan,deps=deps)==0
    result=json.loads((tmp_path/'runs/sampling-sft/sample/instrument/profile/results.json').read_text())
    assert len(result['batch_dependence'])==64
    assert all(r['different_tokens']==0 for r in result['batch_dependence'])
    assert result['profile']['generated_tokens']==192
    assert result['profile']['generation_seconds']>=0
    assert 'prefill_tokens_per_second' in result['profile']
    review=tmp_path/'profile-review.json';review.write_text(json.dumps(dict(candidates=[str(plan)],
        reviewer='fixture',notes='Cannot qualify with fake evidence.')))
    assert cli(tmp_path,'select','--config',plan,'--review',review)==1

import pytest
import json
from pilot_eval import sampling_sft_mixed_jobs as jobs
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash
from pilot_eval.cli import main


def test_mixed_training_dispatch_rejects_legacy_and_wrong_stage(monkeypatch,tmp_path):
    called=[]
    monkeypatch.setattr(jobs.training,'authorize_job',lambda root,job:called.append(job))
    job={'request':{'kind':'gpu','mode':'runpod','stage':'training',
                    'command':['python','-m','pilot_eval','sampling-sft-mixed-training-run']}}
    jobs.authorize_training(tmp_path,job)
    assert called==[job]
    for command in ('sampling-sft-training-run','sampling-sft-mixed-gate-score'):
        invalid={'request':{**job['request'],'command':['python','-m','pilot_eval',command]}}
        with pytest.raises(ValueError):jobs.authorize_training(tmp_path,invalid)
    with pytest.raises(ValueError):jobs.authorize_training(tmp_path,{'request':{**job['request'],'stage':'sampling'}})


def test_mixed_lifecycle_rejects_sealed_legacy_job(tmp_path):
    folder=tmp_path/'job';folder.mkdir()
    job={'implementation':{'sampling_sft_jobs.py':'legacy'},'request':{'job_id':'x'},
         'request_sha256':_hash({'job_id':'x'}),'directory':'job'}
    path=folder/'job.plan.json';path.write_text(json.dumps(job))
    (folder/'job-seal.json').write_text(json.dumps({'plan_file_sha256':file_hash(path),'job_sha256':_hash(job)}))
    with pytest.raises(ValueError,match='identity'):
        jobs._load(path,tmp_path)


def test_mixed_export_cannot_close_without_independent_receipts(tmp_path,monkeypatch):
    folder=tmp_path/'job';folder.mkdir();path=folder/'job.plan.json';path.write_text('{}')
    monkeypatch.setattr(jobs,'_load',lambda *a:(tmp_path,{'request':{'mode':'runpod'}},folder))
    monkeypatch.setattr(jobs,'_work',lambda *a:{'files':{'result':'hash'}})
    monkeypatch.setattr(jobs.copies,'check_receipts',lambda *a:(_ for _ in ()).throw(ValueError('missing independent receipts')))
    with pytest.raises(ValueError,match='independent receipts'):
        jobs.close(path,tmp_path,tmp_path/'network',tmp_path/'download')


@pytest.mark.parametrize('command',['sampling-sft-mixed-training-run','sampling-sft-mixed-job-prepare',
    'sampling-sft-mixed-generation-prepare','sampling-sft-mixed-review-prepare',
    'sampling-sft-mixed-gate-prepare','sampling-sft-mixed-data-prepare'])
def test_mixed_commands_are_registered(command):
    with pytest.raises(SystemExit) as result:main([command,'--help'])
    assert result.value.code==0

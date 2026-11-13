import json

import pytest

from pilot_eval import sampling_sft_buffered_lifecycle as lifecycle


def test_prepares_scoped_buffered_kill_job_without_launch(tmp_path, monkeypatch):
    root = tmp_path
    config = root/'pilot.json'; config.write_text('{}')
    source = root/'source.json'; source.write_text(json.dumps({'run_path':'runs/test'}))
    plan = {'source_plan':'source.json', 'run_path':'runs/test/pilot/v2'}
    monkeypatch.setattr(lifecycle.pilot, 'load', lambda *a:(root,plan,root/plan['run_path']))
    seen = {}
    def prepare(source_path, output_root, request_path):
        seen.update(json.loads(request_path.read_text()))
        return root/'job.json'
    monkeypatch.setattr(lifecycle.jobs, 'prepare', prepare)
    result = lifecycle.prepare(config, root, 'pilot-kill', 'v2-kill', 'volume', '0.60', '1')
    assert result['state'] == 'prepared'
    assert seen['command'][3] == 'sampling-sft-buffered-pilot-run'
    assert seen['command'][seen['command'].index('--interrupt-after-block')+1] == '2'
    assert seen['outputs'] == ['runs/test/pilot/v2/resumed/blocks/001/complete.json',
                               'runs/test/pilot/v2/resumed/writer.json']
    with pytest.raises(ValueError):
        lifecycle.prepare(config, root, 'unknown', 'bad', 'volume', '0.60', '1')


def test_execute_rejects_mismatched_command_before_launch(tmp_path, monkeypatch):
    root = tmp_path
    config = root/'pilot.json'; config.write_text('{}')
    job = root/'job'/'job.plan.json'; job.parent.mkdir(); job.write_text('{}')
    plan = {'source_plan':'source.json', 'run_path':'runs/test/pilot/v2'}
    monkeypatch.setattr(lifecycle.pilot, 'load', lambda *a:(root,plan,root/plan['run_path']))
    monkeypatch.setattr(lifecycle.jobs, '_load', lambda *a:(root,{'source_plan':'source.json',
        'request':{'command':['python','-m','pilot_eval','sampling-sft-generation-run']}},job.parent))
    monkeypatch.setattr(lifecycle.jobs, 'run', lambda *a: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='match'):
        lifecycle.execute(config, root, 'pilot-reference', job, 'reviewer')


def test_execute_rejects_missing_scientific_output_inventory(tmp_path, monkeypatch):
    root=tmp_path; config=root/'pilot.json';config.write_text('{}')
    job=root/'job'/'job.plan.json';job.parent.mkdir();job.write_text('{}')
    plan={'source_plan':'source.json','run_path':'runs/test/pilot/v2'}
    argv=['python','-m','pilot_eval','sampling-sft-buffered-pilot-run','--config',str(config),
          '--lane','uninterrupted','--job',str(job),'--output-root',str(root)]
    request={'command':argv,'mode':'runpod','kind':'gpu','stage':'sampling','outputs':[]}
    monkeypatch.setattr(lifecycle.pilot,'load',lambda *a:(root,plan,root/plan['run_path']))
    monkeypatch.setattr(lifecycle.jobs,'_load',lambda *a:(root,{'source_plan':'source.json','request':request},job.parent))
    monkeypatch.setattr(lifecycle.jobs,'run',lambda *a:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='output inventory'):
        lifecycle.execute(config,root,'pilot-reference',job,'reviewer')


def test_execute_fails_closed_on_failed_workload(tmp_path, monkeypatch):
    from pilot_eval import sampling_sft_pipeline as pipeline
    root=tmp_path;config=root/'pilot.json';config.write_text('{}')
    job=root/'job'/'job.plan.json';job.parent.mkdir();job.write_text('{}')
    plan={'source_plan':'source.json','run_path':'runs/test/pilot/v2'}
    argv=['python','-m','pilot_eval','sampling-sft-buffered-pilot-run','--config',str(config),
          '--lane','uninterrupted','--job',str(job),'--output-root',str(root)]
    outputs=[f'runs/test/pilot/v2/uninterrupted/{x}' for x in ['complete.json','results.json','responses.jsonl']]
    request={'command':argv,'mode':'runpod','kind':'gpu','stage':'sampling','outputs':outputs}
    monkeypatch.setattr(lifecycle.pilot,'load',lambda *a:(root,plan,root/plan['run_path']))
    monkeypatch.setattr(lifecycle.jobs,'_load',lambda *a:(root,{'source_plan':'source.json','request':request},job.parent))
    monkeypatch.setattr(pipeline,'create_job_approval',lambda *a:root/'approval.json')
    monkeypatch.setattr(lifecycle.jobs,'run',lambda *a:{'work_complete':True,'workload_success':False})
    with pytest.raises(ValueError,match='workload failed'):
        lifecycle.execute(config,root,'pilot-reference',job,'reviewer')


def test_kill_stage_rejects_non_sigkill_exit(tmp_path, monkeypatch):
    from pilot_eval import sampling_sft_pipeline as pipeline
    root=tmp_path;config=root/'pilot.json';config.write_text('{}')
    job=root/'job'/'job.plan.json';job.parent.mkdir();job.write_text('{}')
    (job.parent/'execution.json').write_text('{"returncode": 0}')
    plan={'source_plan':'source.json','run_path':'runs/test/pilot/v2'}
    argv=['python','-m','pilot_eval','sampling-sft-buffered-pilot-run','--config',str(config),
          '--lane','resumed','--interrupt-after-block','2','--job',str(job),'--output-root',str(root)]
    outputs=[f'runs/test/pilot/v2/resumed/{x}' for x in ['blocks/001/complete.json','writer.json']]
    request={'command':argv,'mode':'runpod','kind':'gpu','stage':'sampling','outputs':outputs}
    monkeypatch.setattr(lifecycle.pilot,'load',lambda *a:(root,plan,root/plan['run_path']))
    monkeypatch.setattr(lifecycle.pilot,'verify_lane',lambda *a:{'execution_kind':'real-cuda'})
    monkeypatch.setattr(lifecycle.jobs,'_load',lambda *a:(root,{'source_plan':'source.json','request':request},job.parent))
    monkeypatch.setattr(pipeline,'create_job_approval',lambda *a:root/'approval.json')
    monkeypatch.setattr(lifecycle.jobs,'run',lambda *a:{'work_complete':True,'workload_success':True})
    with pytest.raises(ValueError,match='SIGKILL'):
        lifecycle.execute(config,root,'pilot-kill',job,'reviewer')

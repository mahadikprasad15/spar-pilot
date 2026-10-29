"""Exact scientific replay and pilot alarms at the public saved workflow seam."""
import json
import subprocess
import sys
from pathlib import Path

from pilot_eval.cli import main
from test_sampling_sft_instrument import source_template, setup, cli as instrument_cli

from pilot_eval.sampling_sft_sampler import sample
from test_sampling_sft_sampler import Engine, Tokenizer, generation, problem


class PilotEngine(Engine):
    def __init__(self):super().__init__([])
    def generate(self,requests):
        self.requests.extend(requests)
        return [generation([65]*r['max_tokens']) if r['block']<2 else generation([3],'stop',3)
                for r in requests]


def test_core_resumes_json_rng_state_and_identical_logical_requests_after_block_two():
    uninterrupted=sample([problem()],5,PilotEngine(),Tokenizer(),[3])
    blocks=[]
    first=sample([problem()],5,PilotEngine(),Tokenizer(),[3],stop_after_blocks=2,
        on_block=lambda b,states,rng:blocks.append(dict(next_block=b+1,states=states,rng=rng)))
    saved=json.loads(json.dumps(blocks[-1]))
    assert first['state']=='paused' and saved['next_block']==2
    second=sample([problem()],5,PilotEngine(),Tokenizer(),[3],resume=saved)
    assert second['traces']==uninterrupted['traces']
    assert first['steps']+second['steps']==uninterrupted['steps']
    assert first['requests']+second['requests']==uninterrupted['requests']


class Dependencies:
    def __init__(self,fail_block=None):self.fail_block=fail_block
    def open(self,config,root):return PilotEngine(),Tokenizer(),dict(execution_kind='cpu-fixture')
    def after_record(self,record):
        if record.get('kind')=='generate' and record['requests'][0]['block']==self.fail_block:
            raise RuntimeError('Injected partial block interruption')


def cli(root,operation,*args,deps=None):
    return main(['sampling-sft-pilot-'+operation,'--output-root',str(root),*map(str,args)],dependencies=deps)


def prepared(root,template):
    source,settings=setup(root,template)
    assert instrument_cli(root,'prepare','--config',source,'--settings',settings,'--name','bounded')==0
    instrument=root/'plans/bounded/sampling-sft.instrument.json'
    assert cli(root,'prepare','--instrument',instrument,'--name','pilot')==0
    return root/'plans/pilot/sampling-sft.pilot.json'


def test_public_pilot_discards_only_partial_science_and_preserves_sealed_bytes(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--lane','uninterrupted',deps=Dependencies())==0
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Dependencies(fail_block=2))==1
    folder=tmp_path/'runs/sampling-sft/sample/pilot/pilot/resumed'
    saved={p:p.read_bytes() for p in (folder/'blocks').rglob('*') if p.is_file()}
    assert len(list((folder/'blocks').glob('*/complete.json')))==2
    assert (folder/'partial/scientific.jsonl').exists()
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Dependencies())==0
    assert all(p.read_bytes()==value for p,value in saved.items())
    assert list((folder/'abandoned').glob('*/scientific.jsonl'))
    assert cli(tmp_path,'compare','--config',plan)==0
    comparison=json.loads((folder.parent/'replay.json').read_text())
    assert comparison['scientific_equal'] and not comparison['gpu_qualified']
    assert comparison['compared_blocks']==3
    # Restoration is numeric by problem ID, not lexicographic JSON key order.
    result=json.loads((folder/'results.json').read_text())
    assert [r['problem_id'] for r in result['traces']]==sorted(r['problem_id'] for r in result['traces'])


def test_pilot_writer_and_real_preflight_are_explicit(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed')==1
    folder=tmp_path/'runs/sampling-sft/sample/pilot/pilot/resumed';folder.mkdir(parents=True,exist_ok=True)
    (folder/'writer.json').write_text(json.dumps(dict(token='owner',host='other-host',pid=123)))
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Dependencies())==1
    assert cli(tmp_path,'recover-writer','--config',plan,'--lane','resumed','--owner-token','wrong','--review-notes','checked')==1
    assert cli(tmp_path,'recover-writer','--config',plan,'--lane','resumed','--owner-token','owner','--review-notes','checked dead worker')==0
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed','--stop-after-blocks','2',deps=Dependencies())==0
    assert not (folder/'complete.json').exists()


def test_alarm_report_uses_real_denominators_and_never_claims_full_generation(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    for lane in ['uninterrupted','resumed']:
        assert cli(tmp_path,'run','--config',plan,'--lane',lane,deps=Dependencies())==0
    assert cli(tmp_path,'compare','--config',plan)==0
    forecast=tmp_path/'forecast.json';forecast.write_text(json.dumps(dict(rate_usd_per_hour='1.09',estimated_hours='0.5',setup_hours='0.1',
        retry_policy='reserve-all-four-attempts',notes='Conservative measured pilot projection; no job approval.')))
    assert cli(tmp_path,'report','--config',plan,'--forecast',forecast)==0
    report=json.loads((tmp_path/'runs/sampling-sft/sample/pilot/pilot/report.json').read_text())
    assert report['n_problems']==32 and report['initial_correct']==0
    assert report['alarms']['initial_correctness']['triggered']
    assert report['alarms']['initial_correctness']['threshold']==0.9
    assert report['alarms']['unfinished']['denominator']==32
    assert report['production_allowed'] is False


def test_real_cpu_child_sigkill_requires_owner_recovery_and_fresh_process_resume(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--lane','uninterrupted',deps=Dependencies())==0
    script='''
import sys
sys.path.insert(0, 'tests')
from test_sampling_sft_pilot import Dependencies
from pilot_eval.cli import main
raise SystemExit(main(sys.argv[1:], dependencies=Dependencies()))
'''
    argv=['sampling-sft-pilot-run','--output-root',str(tmp_path),'--config',str(plan),'--lane','resumed']
    killed=subprocess.run([sys.executable,'-c',script,*argv,'--interrupt-after-block','2'],
                          cwd=Path(__file__).parents[1],capture_output=True,text=True)
    assert killed.returncode==-9
    folder=tmp_path/'runs/sampling-sft/sample/pilot/pilot/resumed'
    owner=json.loads((folder/'writer.json').read_text())
    original={p:p.read_bytes() for p in (folder/'blocks').rglob('*') if p.is_file()}
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Dependencies())==1
    assert cli(tmp_path,'recover-writer','--config',plan,'--lane','resumed','--owner-token',owner['token'],
               '--review-notes','Child exit -9 confirmed; local PID dead.')==0
    resumed=subprocess.run([sys.executable,'-c',script,*argv],cwd=Path(__file__).parents[1],capture_output=True,text=True)
    assert resumed.returncode==0,resumed.stderr
    assert all(p.read_bytes()==value for p,value in original.items())
    assert cli(tmp_path,'compare','--config',plan)==0
    receipt=json.loads((folder.parent/'replay.json').read_text())
    assert receipt['kill_after_block2_recorded'] and receipt['restart_from_block2']
    assert not receipt['gpu_qualified']


def test_replay_difference_stops_without_tolerance_and_changed_runtime_is_rejected(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--lane','uninterrupted',deps=Dependencies())==0
    class Changed(Dependencies):
        def open(self,config,root):
            engine,tokenizer,runtime=super().open(config,root)
            original=engine.score
            def changed(requests):
                rows=original(requests)
                for row in rows:row['logprobs']=[-0.999]*len(row['logprobs'])
                return rows
            engine.score=changed
            return engine,tokenizer,runtime
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Changed())==0
    assert cli(tmp_path,'compare','--config',plan)==1
    receipt=json.loads((tmp_path/'runs/sampling-sft/sample/pilot/pilot/replay.json').read_text())
    assert not receipt['scientific_equal'] and len(receipt['requalification_order'])==3
    assert 'logprobs' in receipt['first_difference']['path']
    value=json.loads(plan.read_text());value['settings']['runtime']['max_num_seqs']=64
    plan.write_text(json.dumps(value))
    assert cli(tmp_path,'status','--config',plan)==1


def test_normalized_p90_matches_the_declared_inclusive_quantile():
    from pilot_eval.sampling_sft_pilot import length_statistics
    values=list(range(1,33))
    assert length_statistics(values)==dict(median=16.5,p90=28.9)


def test_tampered_block_or_environment_never_replays_as_compatible(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed','--stop-after-blocks','2',deps=Dependencies())==0
    folder=tmp_path/'runs/sampling-sft/sample/pilot/pilot/resumed'
    env=next(folder.glob('env.*.json'));env.write_text('{"execution_kind":"real-cuda"}')
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Dependencies())==1


def test_engine_replay_metadata_is_operational_not_scientific(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--lane','uninterrupted',deps=Dependencies())==0
    class Flaky(Dependencies):
        def open(self,config,root):
            engine,tokenizer,runtime=super().open(config,root)
            original=engine.generate;first=[True]
            def generate(requests):
                if first[0]:first[0]=False;raise RuntimeError('transient fixture failure')
                return original(requests)
            engine.generate=generate
            return engine,tokenizer,runtime
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=Flaky())==0
    assert cli(tmp_path,'compare','--config',plan)==0


def test_completed_blocks_finalize_without_reloading_a_model(tmp_path,source_template):
    plan=prepared(tmp_path,source_template)
    class EndFailure(Dependencies):
        def after_checkpoint(self,block):
            if block==2:raise RuntimeError('Interruption after final block seal')
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=EndFailure())==1
    class NoModel(Dependencies):
        def open(self,*args):raise AssertionError('Completed work must not reload a model')
    assert cli(tmp_path,'run','--config',plan,'--lane','resumed',deps=NoModel())==0

"""Full512 reconciliation and recovery through the public CPU-fixture seam."""
import json

import pytest

from pilot_eval.cli import main
from pilot_eval.sampling_sft_sampler import request_seed
from pilot_eval.sampling_sft_generation import plan_approval, reservation_fits
from test_sampling_sft_instrument import source_template
from test_sampling_sft_pilot import prepared as pilot_plan
from test_sampling_sft_sampler import Engine, Tokenizer, generation


def cli(root,operation,*args,deps=None):
    return main(['sampling-sft-generation-'+operation,'--output-root',str(root),*map(str,args)],dependencies=deps)


def prepare(root,template):
    pilot=pilot_plan(root,template)
    assert cli(root,'prepare','--pilot',pilot,'--name','generation')==0
    return root/'plans/generation/sampling-sft.generation.json'


def directory(root):return root/'runs/sampling-sft/sample/generation/generation'


class GenerationEngine(Engine):
    def __init__(self,config,kind='ordinary'):
        super().__init__([])
        self.problems={p['problem_id']:p for p in config['problems']}
        self.kind=kind;self.transient=False

    def generate(self,requests):
        self.requests.extend(requests)
        if self.kind=='transient' and not self.transient:
            self.transient=True;raise RuntimeError('one same-seed protocol replay')
        outputs=[]
        for r in requests:
            pid=r['problem_id'];step=r['step']
            if self.kind=='broken' and pid==0:outputs.append(generation([]));continue
            initial=r['seed']==request_seed(0,pid,r['block'],step)
            if pid==2 and initial and step==-1:
                outputs.append(generation([65]*r['max_tokens']));continue
            gold=self.problems[pid]['gold'].rsplit('####',1)[-1].strip()
            answer='-999999' if pid==0 or (pid in [1,4] and initial) else gold
            if step!=-1:outputs.append(generation([3],'stop',3));continue
            if pid==4 and initial:outputs.append(generation([3],'stop',3));continue
            outputs.append(generation([ord(c) for c in '#### '+answer]+[3],'stop',3))
        return outputs


class Dependencies:
    def __init__(self,kind='ordinary',fail_block=None):
        self.kind=kind;self.fail_block=fail_block;self.engines=[]
    def open(self,config,root):
        engine=GenerationEngine(config,self.kind);self.engines.append(engine)
        return engine,Tokenizer(),dict(execution_kind='cpu-fixture')
    def after_record(self,record):
        if record.get('kind')=='generate' and record['requests'][0]['block']==self.fail_block:
            raise RuntimeError('partial-block fixture interruption')


def read_lines(path):return [json.loads(s) for s in path.read_text().splitlines()]


def test_full_arms_reconcile_retries_and_never_claim_real_qualification(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    for k in [0,2,5]:
        assert cli(tmp_path,'run','--config',plan,'--k',k,deps=Dependencies())==0
        assert cli(tmp_path,'verify','--config',plan,'--k',k)==0
        folder=directory(tmp_path)/f'k{k}'
        attempts=read_lines(folder/'attempts.jsonl');dispositions=read_lines(folder/'dispositions.jsonl')
        assert len(dispositions)==512 and len({d['problem_id'] for d in dispositions})==512
        by_pid={i:[a for a in attempts if a['problem_id']==i] for i in range(512)}
        assert len(by_pid[0])==4 and all(a['end_state']=='finished' for a in by_pid[0])
        assert [a['retry_index'] for a in by_pid[1]]==[0,1]
        assert [a['data_seed'] for a in by_pid[1]]==[0,1001]
        assert by_pid[2][0]['end_state']=='capped' and by_pid[2][1]['eligible']
        assert by_pid[4][0]['response_ids']==[3] and not by_pid[4][0]['eligible']
        assert all(len(by_pid[i])==1 for i in range(5,512))
        for d in dispositions:
            eligible=[a for a in by_pid[d['problem_id']] if a['eligible']]
            assert d['selected_attempt']==(eligible[0]['attempt_id'] if eligible else None)
        summary=json.loads((folder/'results.json').read_text())
        assert summary['covered']==511 and summary['uncovered']==1
        assert summary['execution_kind']=='cpu-fixture' and not summary['training_allowed']
        assert summary['attempt_count']==len(attempts)
        assert sum(summary['end_state_counts'].values())==len(attempts)
        assert summary['response_tokens']==sum(len(a['response_ids']) for a in attempts)
    assert cli(tmp_path,'verify','--config',plan)==0
    assert (directory(tmp_path)/'complete.json').exists()


def test_partial_round_resume_preserves_sealed_bytes_and_skips_covered_work(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=Dependencies(fail_block=1))==1
    folder=directory(tmp_path)/'k0/rounds/0'
    saved={p:p.read_bytes() for p in (folder/'blocks').rglob('*') if p.is_file()}
    assert len(list((folder/'blocks').glob('*/complete.json')))==1
    assert (folder/'partial/scientific.jsonl').exists()
    deps=Dependencies()
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=deps)==0
    assert all(p.read_bytes()==b for p,b in saved.items())
    assert list((folder/'abandoned').glob('*/scientific.jsonl'))
    # After resume, previously completed responses are never regenerated.
    first=deps.engines[0].requests[0]
    assert first['problem_id']==2 and first['block']==1
    assert all(r['problem_id']<5 for r in deps.engines[0].requests)
    class NoModel(Dependencies):
        def open(self,*args):raise AssertionError('sealed arm must not reload a model')
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=NoModel())==0


def test_engine_replay_is_not_a_model_retry_and_exhausted_engine_blocks_completion(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=Dependencies(kind='transient'))==0
    attempts=read_lines(directory(tmp_path)/'k0/attempts.jsonl')
    assert len([a for a in attempts if a['problem_id']==5])==1
    assert cli(tmp_path,'run','--config',plan,'--k','2',deps=Dependencies(kind='broken'))==1
    folder=directory(tmp_path)/'k2'
    assert not (folder/'complete.json').exists() and not (folder/'dispositions.jsonl').exists()
    failures=read_lines(folder/'rounds/0/failed-attempts.jsonl')
    assert len(failures)==512 and all(a['end_state']=='engine_error' for a in failures)
    assert cli(tmp_path,'run','--config',plan,'--k','2',deps=Dependencies())==1


def test_real_generation_and_approval_refuse_fixture_pilot_before_model_import(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--k','0')==1
    assert not (directory(tmp_path)/'k0/events.jsonl').exists()
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(reviewer='researcher',notes='No qualification yet',
        generation_approved=True,plan_sha256='not-the-plan')))
    assert cli(tmp_path,'freeze','--config',plan,'--review',review)==1


def test_modified_science_and_competing_writer_reject_resume(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=Dependencies(fail_block=1))==1
    folder=directory(tmp_path)/'k0'
    (folder/'writer.json').write_text(json.dumps(dict(token='held',host='external',pid=321)))
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=Dependencies())==1
    assert cli(tmp_path,'recover-writer','--config',plan,'--k','0','--owner-token','held','--review-notes','Confirmed dead fixture')==0
    target=folder/'rounds/0/blocks/000/scientific.jsonl';target.write_text('changed')
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=Dependencies())==1


def test_pause_reports_verified_provisional_coverage_without_claiming_completion(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--k','0','--stop-after-blocks','1',deps=Dependencies())==0
    folder=directory(tmp_path)/'k0'
    meta=json.loads((folder/'meta/status.json').read_text())
    assert meta['completed']==508 and meta['state']=='paused'
    from pilot_eval.sampling_sft_generation import status
    value=status(plan,tmp_path)['arms']['0']
    assert value['provisional_covered']==508 and value['attempt_count']==0
    assert not (folder/'complete.json').exists()


def test_approval_is_bound_to_plan_and_report_and_budget_math_is_exact(tmp_path,source_template):
    from pilot_eval.workflow import _hash
    plan=prepare(tmp_path,source_template);config=json.loads(plan.read_text())
    # Missing report was observed by the plan, so do not mutate its parent: use an
    # independent pure approval function, then test actual freeze refusal above.
    review=dict(reviewer='researcher',notes='reviewed',generation_approved=True,
                plan_sha256=_hash(config),pilot_report_sha256='r'*64)
    plan_approval(config,review,'r'*64)
    review['plan_sha256']='different'
    with pytest.raises(ValueError):plan_approval(config,review,'r'*64)
    ledger=dict(cap_exceeded=False,committed_usd='9.999999',stage_totals={'sampling':{'committed_usd':'9.999999'}})
    assert reservation_fits(ledger,'0.000001')
    assert not reservation_fits(ledger,'0.000002')


def test_last_round_finalization_uses_sealed_state_without_model_reload(tmp_path,source_template):
    plan=prepare(tmp_path,source_template)
    class EndFailure(Dependencies):
        def after_checkpoint(self,k,index,block):
            if index==3:raise RuntimeError('after final round block seal')
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=EndFailure())==1
    class NoModel(Dependencies):
        def open(self,*args):raise AssertionError('all remaining states are already sealed')
    assert cli(tmp_path,'run','--config',plan,'--k','0',deps=NoModel())==0
    assert cli(tmp_path,'verify','--config',plan,'--k','0')==0


def test_exhausted_model_attempt_is_retried_without_mixing_engine_errors(tmp_path,source_template,monkeypatch):
    monkeypatch.setattr('random.Random.randint',lambda self,a,b:0)
    plan=prepare(tmp_path,source_template)
    class Exhaust(Dependencies):
        def open(self,config,root):
            engine=GenerationEngine(config)
            original=engine.generate
            def generate(requests):
                outputs=original(requests)
                for i,r in enumerate(requests):
                    initial=r['seed']==request_seed(0,r['problem_id'],r['block'],r['step'])
                    if r['problem_id']==2 and initial and r['step']>=1:
                        outputs[i]=generation([66]*r['max_tokens'])
                return outputs
            old_score=engine.score
            def score(requests):
                rows=old_score(requests)
                for r,row in zip(requests,rows):
                    if r['problem_id']==2:
                        ids=r['response_ids'];value=-1.0 if ids[-1]==3 else -0.5 if len(ids)==1 else -2.0
                        row['logprobs']=[value]*len(ids)
                return rows
            engine.generate=generate;engine.score=score
            return engine,Tokenizer(),dict(execution_kind='cpu-fixture')
    assert cli(tmp_path,'run','--config',plan,'--k','2',deps=Exhaust())==0
    attempts=read_lines(directory(tmp_path)/'k2/attempts.jsonl')
    rows=[a for a in attempts if a['problem_id']==2]
    assert rows[0]['end_state']=='block_budget_exhausted' and rows[0]['blocks']==32
    assert rows[1]['eligible'] and rows[1]['retry_index']==1
    assert all(a['end_state']!='engine_error' for a in attempts)


def test_generation_review_is_idempotent_after_ledger_changes(tmp_path,source_template,monkeypatch):
    import pilot_eval.sampling_sft_generation as workflow
    from pilot_eval.workflow import _hash
    from pilot_eval.training import file_hash
    pilot=pilot_plan(tmp_path,source_template)
    pconf=json.loads(pilot.read_text())
    report=tmp_path/pconf['run_path']/'report.json'
    report.write_text(json.dumps(dict(alarms={'budget':{'reservation_usd':'1.000000'}})))
    assert cli(tmp_path,'prepare','--pilot',pilot,'--name','generation')==0
    plan=tmp_path/'plans/generation/sampling-sft.generation.json'
    review=tmp_path/'approval.json';review.write_text(json.dumps(dict(reviewer='fixture',notes='Approval-unit test only',
        generation_approved=True,plan_sha256=_hash(json.loads(plan.read_text())),pilot_report_sha256=file_hash(report))))
    monkeypatch.setattr(workflow,'qualification',lambda *a:json.loads(report.read_text()))
    assert cli(tmp_path,'freeze','--config',plan,'--review',review)==0
    old=(directory(tmp_path)/'generation.review.json').read_bytes()
    # Model a later ledger snapshot; this does not mutate the actual cost ledger.
    saved=workflow.snapshot
    monkeypatch.setattr(workflow,'snapshot',lambda root:{**saved(root),'sha256':'different-snapshot'})
    assert cli(tmp_path,'freeze','--config',plan,'--review',review)==0
    assert (directory(tmp_path)/'generation.review.json').read_bytes()==old

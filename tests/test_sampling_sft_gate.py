"""Saved gate scores/report/selection; model work and CPU analysis stay distinct."""
import json
import shutil

import pytest

from pilot_eval.cli import main
from test_sampling_sft_review import collected,completed_labels
from test_sampling_sft_review import setup as review_plan,cli as review_cli
from test_sampling_sft_generation import source_template
from test_sampling_sft_sampler import Tokenizer


def cli(root,operation,*args,deps=None):
    return main(['sampling-sft-gate-'+operation,'--output-root',str(root),*map(str,args)],dependencies=deps)


@pytest.fixture(scope='module')
def reviewed(tmp_path_factory,collected):
    root=tmp_path_factory.mktemp('gate-reviewed');plan=review_plan(root,collected)
    labels,_=completed_labels(root,plan)
    assert review_cli(root,'import','--config',plan,'--labels',labels,'--origin','human','--confirm-human')==0
    assert review_cli(root,'freeze','--config',plan)==0
    return root


def prepare(root,reviewed):
    shutil.copytree(reviewed,root,dirs_exist_ok=True)
    review=root/'plans/review/sampling-sft.review.json'
    policy=root/'gate-word-policy.json'
    policy.write_text(json.dumps(dict(version='unicode-words-v1',approved=True,reviewer='fixture',notes='Synthetic fixture policy, not real design approval.')))
    assert cli(root,'prepare','--review',review,'--word-policy',policy,'--name','gate')==0
    return root/'plans/gate/sampling-sft.gate.json'


def directory(root):return root/'runs/sampling-sft/sample/gate/gate'


class Oracle:
    def score(self,requests):
        return [dict(token_ids=r['response_ids'],positions=list(range(len(r['prompt_ids']),len(r['prompt_ids'])+len(r['response_ids']))),
            logprobs=[-1.0]*(len(r['response_ids'])-1)+[-3.0]) for r in requests]


class Dependencies:
    def __init__(self,fail=False):self.fail=fail;self.calls=0
    def open(self,config,root):return Oracle(),Tokenizer(),dict(execution_kind='cpu-fixture')
    def after_unit(self,index):
        self.calls+=1
        if self.fail and index==0:raise RuntimeError('fixture interruption after one sealed problem')


def test_score_restore_and_cpu_report_preserve_gold_and_common_cohorts(tmp_path,reviewed):
    plan=prepare(tmp_path,reviewed);folder=directory(tmp_path)
    assert cli(tmp_path,'report','--config',plan)==1
    assert cli(tmp_path,'score','--config',plan,'--max-new-units','1',deps=Dependencies())==0
    saved={p:p.read_bytes() for p in (folder/'scores/units/000').iterdir() if p.is_file()}
    assert cli(tmp_path,'score','--config',plan,deps=Dependencies())==0
    assert all(p.read_bytes()==v for p,v in saved.items())
    assert cli(tmp_path,'report','--config',plan)==0
    assert cli(tmp_path,'verify','--config',plan)==0
    report=json.loads((folder/'results/report.json').read_text())
    assert report['execution_kind']=='cpu-fixture' and not report['training_authorized']
    assert report['own_cohort']['gold']['n_answers']==512
    assert all(report['own_cohort'][k]['n_answers']==511 for k in ['k0','k2','k5'])
    assert all(v['n_answers']==511 for v in report['common_cohort'].values())
    assert report['baseline']=='not measured' and report['relative_sharpening']=='not measured'
    assert 'offset' not in report['own_cohort']['gold']
    assert report['paired_contrasts']['k0-minus-k5']['nll']['estimate']==0
    rows=[json.loads(s) for s in (folder/'results/answers.jsonl').read_text().splitlines()]
    for row in rows:
        assert row['nll']==pytest.approx((len(row['response_ids'])+2)/len(row['response_ids']))
        assert row['logprobs'][-1]==-3 and row['response_ids'][-1] in [3,4]
    assert (folder/'plots/common-nll.svg').exists() and (folder/'plots/manifest.json').exists()
    assert (folder/'report.html').exists() and (folder/'results/summary.csv').exists()
    from pilot_eval.sampling_sft_gate import request
    from pilot_eval.sampling_sft_budget import reserve,initialize
    gen=tmp_path/'plans/sample/sampling-sft.plan.json'
    forecast=tmp_path/'training-forecast.json';forecast.write_text(json.dumps(dict(arms=[0,5],per_arm={str(k):dict(
        rate_usd_per_hour='0.59',estimated_hours='1',setup_hours='0.1') for k in [0,5]},notes='Fixture measured estimate')))
    requested=request(plan,tmp_path,forecast)
    assert requested['selection']['reservation_usd']=='1.888000'
    assert requested['required_acknowledgments']==['k0:omissions-require-T16','k2:omissions-require-T16','k5:omissions-require-T16']
    approval=tmp_path/'training-approval.json'
    value=dict(request_path=requested['request_path'],request_sha256=requested['request_sha256'],
        report_sha256=requested['report_sha256'],budget_sha256=requested['budget_snapshot']['sha256'],
        reviewer='fixture-human',notes='Intent selection only, readiness not yet checked.',gate_approved=True,
        acknowledged_coverage_flags=requested['required_acknowledgments'],review_80_percent=False)
    approval.write_text(json.dumps(value))
    assert cli(tmp_path,'approve','--config',plan,'--approval',approval)==0
    decisions=list((folder/'training-selections').glob('*.json'))
    decision=json.loads(decisions[0].read_text())
    assert decision['arms']==[0,5] and decision['selection_confirmed'] and not decision['training_authorized']
    # A new independent zero-cost lifecycle reservation still changes ledger identity.
    ledger_request=tmp_path/'noop-budget-request.json';ledger_request.write_text(json.dumps(dict(job_id='new-ledger-event',kind='noop',stage='sampling',mode='noop',command=[])))
    reserve(gen,tmp_path,ledger_request)
    assert cli(tmp_path,'approve','--config',plan,'--approval',approval)==1
    (folder/'results/summary.csv').write_text('changed report')
    assert cli(tmp_path,'verify','--config',plan)==1


def test_real_score_route_refuses_unqualified_fixture_and_retains_interruption(tmp_path,reviewed):
    plan=prepare(tmp_path,reviewed)
    assert cli(tmp_path,'score','--config',plan)==1
    assert not (directory(tmp_path)/'scores/events.jsonl').exists()
    assert cli(tmp_path,'score','--config',plan,deps=Dependencies(fail=True))==1
    assert (directory(tmp_path)/'scores/units/000/complete.json').exists()
    assert cli(tmp_path,'score','--config',plan,'--max-new-units','1',deps=Dependencies())==0
    assert (directory(tmp_path)/'scores/units/001/complete.json').exists()


def test_misaligned_scoring_replays_once_and_blocks_report(tmp_path,reviewed):
    plan=prepare(tmp_path,reviewed)
    class Bad:
        def score(self,requests):return [dict(token_ids=[],positions=[],logprobs=[]) for _ in requests]
    class BadDependencies(Dependencies):
        def open(self,*args):return Bad(),Tokenizer(),dict(execution_kind='cpu-fixture')
    assert cli(tmp_path,'score','--config',plan,deps=BadDependencies())==1
    assert not (directory(tmp_path)/'scores/units/000/complete.json').exists()
    errors=[json.loads(s) for s in (directory(tmp_path)/'scores/events.jsonl').read_text().splitlines()]
    assert sum(r['event']=='score-engine-error' for r in errors)==2
    assert cli(tmp_path,'report','--config',plan)==1

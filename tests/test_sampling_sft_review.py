"""Source-bound human eligibility, immutable scoring and coverage products."""
import json
import shutil

import pytest

from pilot_eval.cli import main
from pilot_eval.sampling_sft_generation import verify as verify_generation
from test_sampling_sft_generation import source_template,prepare as generation_plan,Dependencies,GenerationEngine
from test_sampling_sft_generation import cli as generation_cli
from test_sampling_sft_sampler import Tokenizer,generation
from pilot_eval.sampling_sft_sampler import request_seed


def cli(root,operation,*args):
    return main(['sampling-sft-review-'+operation,'--output-root',str(root),*map(str,args)])


@pytest.fixture(scope='module')
def collected(tmp_path_factory,source_template):
    root=tmp_path_factory.mktemp('review-collection');plan=generation_plan(root,source_template)
    class ReviewDependencies(Dependencies):
        def open(self,config,artifact_root):
            engine=GenerationEngine(config);original=engine.generate
            def generate_rows(requests):
                values=original(requests)
                for i,r in enumerate(requests):
                    if r['problem_id']==4 and r['seed']==request_seed(0,4,r['block'],r['step']) and r['step']==-1:
                        gold=engine.problems[4]['gold'].rsplit('####',1)[-1].strip()
                        values[i]=generation([ord(c) for c in f'I counted 999 earlier. My result: {gold} dollars.']+[3],'stop',3)
                return values
            engine.generate=generate_rows
            return engine,Tokenizer(),dict(execution_kind='cpu-fixture')
    for k in [0,2,5]:assert generation_cli(root,'run','--config',plan,'--k',k,deps=ReviewDependencies())==0
    return root


def setup(root,collected):
    shutil.copytree(collected,root,dirs_exist_ok=True)
    gen=root/'plans/generation/sampling-sft.generation.json'
    assert cli(root,'prepare','--generation',gen,'--name','review')==0
    return root/'plans/review/sampling-sft.review.json'


def folder(root):return root/'runs/sampling-sft/sample/eligibility/review'


def draft(root,plan):
    path=root/f'download-{len(list(root.glob("download-*.json")))}.json'
    assert cli(root,'export','--config',plan,'--destination',path)==0
    return path,json.loads(path.read_text())


def completed_labels(root,plan):
    path,value=draft(root,plan)
    packet=json.loads((folder(root)/'reviewer/packet.json').read_text())
    gen=json.loads((root/'plans/generation/sampling-sft.generation.json').read_text())
    by_pid={p['problem_id']:p for p in gen['problems']}
    value['reviewer']='fixture-human'
    for row,item in zip(value['labels'],packet):
        if item['response']:
            gold=by_pid[item['problem_id']]['gold'].rsplit('####',1)[-1].strip()
            row.update(status='numeric',answer=gold,evidence=f'My result: {gold} dollars.',note='Transcribed from response.')
        else:row.update(status='absent',answer=None,evidence='',note='Stop-only response.')
    path.write_text(json.dumps(value))
    return path,value


def test_public_review_roundtrip_freeze_keeps_auto_scores_and_original_slots(tmp_path,collected):
    plan=setup(tmp_path,collected)
    packet=json.loads((folder(tmp_path)/'reviewer/packet.json').read_text())
    assert len(packet)==3 and all(row['problem_id']==4 for row in packet)
    assert all('gold' not in row and 'correct' not in row for row in packet)
    assert cli(tmp_path,'freeze','--config',plan)==1
    path,labels=completed_labels(tmp_path,plan)
    assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==0
    assert cli(tmp_path,'freeze','--config',plan)==0
    assert cli(tmp_path,'verify','--config',plan)==0
    results=json.loads((folder(tmp_path)/'results/coverage.json').read_text())
    assert results['training_allowed'] is False and len(results['common_covered_ids'])==511
    assert results['human_review_complete']
    for k in ['0','2','5']:
        arm=results['arms'][k]
        assert arm['uncovered_count']==1 and arm['coverage_decision']=='conditional-on-T16-and-training-approval'
        assert len(arm['dispositions'])==512 and len(arm['batches'])==64
        assert sum(len(b['slots']) for b in arm['batches'])==512
        assert sum(s['missing'] for b in arm['batches'] for s in b['slots'])==1
        corrected=next(r for r in arm['dispositions'] if r['problem_id']==4)
        assert corrected['selected_attempt']==f'k{k}:p4:r0' and corrected['eligibility_origin']=='confirmed-human'
        original=next(r for r in arm['attempts'] if r['attempt_id']==f'k{k}:p4:r0')
        assert original['automatic_flexible_v3']['status']=='invalid' and not original['eligible']
    assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==1


def test_agent_suggestions_and_partial_reviews_cannot_pass_required_review(tmp_path,collected):
    plan=setup(tmp_path,collected);path,value=completed_labels(tmp_path,plan)
    assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','agent')==0
    assert cli(tmp_path,'freeze','--config',plan)==1
    _,human=draft(tmp_path,plan)
    assert human['base_revision']==0 and all(r['status']=='pending' for r in human['labels'])
    assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human')==1
    value['labels'][1]['status']='pending';value['labels'][1]['answer']=None
    path.write_text(json.dumps(value))
    assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==0
    assert cli(tmp_path,'freeze','--config',plan)==1
    assert cli(tmp_path,'status','--config',plan)==0


def test_stale_revision_unknown_labels_and_unsupported_evidence_are_rejected(tmp_path,collected):
    plan=setup(tmp_path,collected);path,value=completed_labels(tmp_path,plan)
    original=json.loads(json.dumps(value))
    value['labels'][0]['evidence']='invented answer not present'
    path.write_text(json.dumps(value));assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==1
    value=json.loads(json.dumps(original));value['labels'][0]['response_id']='unknown'
    path.write_text(json.dumps(value));assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==1
    path.write_text(json.dumps(original));assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==0
    assert cli(tmp_path,'import','--config',plan,'--labels',path,'--origin','human','--confirm-human')==1
    _,updated=draft(tmp_path,plan);assert updated['base_revision']==1


@pytest.mark.parametrize('missing,decision',[(0,'passes-coverage-only'),(1,'conditional-on-T16-and-training-approval'),
    (5,'conditional-on-T16-and-training-approval'),(6,'blocked-more-than-five-uncovered')])
def test_coverage_limits_and_original_slots(missing,decision):
    from pilot_eval.sampling_sft_review import coverage_arm
    problems=[dict(problem_id=i,source_id=f'p{i}') for i in range(512)]
    batches=[[f'p{i}' for i in range(start,start+8)] for start in range(0,512,8)]
    attempts=[dict(problem_id=i,source_id=f'p{i}',attempt_id=f'a{i}',eligible=i>=missing,
        end_state='finished',automatic_flexible_v3=dict(status='valid',correct=i>=missing)) for i in range(512)]
    result=coverage_arm(0,problems,batches,attempts,{})
    assert result['coverage_decision']==decision
    assert result['uncovered_count']==missing and result['batches'][0]['nonmissing_count']==8-missing
    assert len(result['batches'])==64 and len(result['dispositions'])==512


def test_unsafe_overrides_never_make_unfinished_or_valid_wrong_answers_eligible():
    from pilot_eval.sampling_sft_review import eligibility_override
    problem=dict(gold='work\n#### 12')
    label=dict(status='numeric',answer='12',evidence='12',note='')
    for end,status in [('capped','invalid'),('block_budget_exhausted','invalid'),('finished','valid')]:
        attempt=dict(end_state=end,automatic_flexible_v3={'status':status},text='12',eligible=False)
        with pytest.raises(ValueError):eligibility_override(attempt,label,problem)


def test_changed_generation_payload_rejects_review_restore(tmp_path,collected):
    plan=setup(tmp_path,collected)
    source=tmp_path/'runs/sampling-sft/sample/generation/generation/k0/attempts.jsonl'
    source.write_text('tampered')
    assert cli(tmp_path,'status','--config',plan)==1

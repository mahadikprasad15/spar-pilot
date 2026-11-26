"""Offline dataset score mapping and weighted summaries on CPU."""
import numpy as np
import pytest


def test_duplicate_payload_retains_membership_and_zero_correct():
    from pilot_eval.offline_grpo_dataset import pool_answers, aggregate
    pool=[dict(id='q',input_ids=[3],draws=[dict(draw_id=f'q:draw:{i}',generated_token_ids=[4,99],completion_mask=[1,1],stop_reason='eos',capped=False,reward_v3=0) for i in range(8)])]
    answers,members=pool_answers(pool,99)
    assert len(answers)==1 and len(members)==8
    scores={answers[0]['answer_id']:dict(surprise=np.array([1.,3.]),entropy=np.array([1.,1.]),rank=np.array([1,2]))}
    result=aggregate(members,scores)
    assert result['answers']==8 and result['questions']==1 and result['token_weighted']['nll']==2
    assert result['equal_question']['nll']==2


def test_eos_and_mask_rejected_when_inconsistent():
    from pilot_eval.offline_grpo_dataset import pool_answers
    pool=[dict(id='q',input_ids=[3],draws=[dict(draw_id='d',generated_token_ids=[4,99],completion_mask=[1,0],stop_reason='eos',capped=False,reward_v3=0)])]
    with pytest.raises(ValueError,match='mask'):
        pool_answers(pool,99)


def test_public_cpu_scoring_resume_report_and_tamper(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_dataset as ds
    from pilot_eval.cli import main
    from pilot_eval.training import file_hash
    answer=ds._answer([3],[4,99]);answer2=ds._answer([3],[5,99])
    member=lambda name:dict(problem_id='q',answer_id=answer['answer_id'],draw_id=name,correct=False,capped=False)
    memberships=dict(pool=[member(str(i)) for i in range(8)],gold=[member('g')],k0=[member('k0')],k2=[member('k2')],k5=[dict(member('k5'),answer_id=answer2['answer_id'])])
    receipt=dict(metadata=dict(eos_token_id=99),r0_sha256='placeholder')
    for name in ['pool','sft']:(tmp_path/f'{name}.json').write_text('{}')
    (tmp_path/'sft.json').write_text(json.dumps(dict(protocol='sampling-stage0-dataset-v1',run_path='old')))
    old_meta=tmp_path/'old/meta';old_meta.mkdir(parents=True)
    vocab=dict(logit_vocabulary_size=100,tokenizer_vocabulary_size=100,padded_output_rows=0,includes_padded_rows=True,rank_and_softmax_same_full_vector=True)
    (old_meta/'vocabulary.json').write_text(json.dumps(vocab))
    (old_meta/'runtime.json').write_text(json.dumps(dict(session='historical')))
    old_scores=dict(surprise=np.ones(2,dtype=np.float64),entropy=np.ones(2,dtype=np.float64),rank=np.ones(2,dtype=np.int64))
    monkeypatch.setattr(ds,'_legacy',lambda *args:({}, {}, {answer['answer_id']:old_scores}))
    calls=[]
    sampler=tmp_path/'sampler.json';sampler.write_text(json.dumps(dict(protocol='offline-grpo-fresh-r0-v1',model_identity={'base_sha256':'fixture'},weights_dir='weights',execution_kind='synthetic-cpu-fixture')))
    receipt['r0_sha256']=file_hash(sampler)
    runtime=dict(execution_kind='synthetic-cpu-fixture',session_id='fixture')
    (tmp_path/'runtime.json').write_text(json.dumps(runtime))
    monkeypatch.setattr(ds,'_inputs',lambda *args:([answer,answer2],memberships,receipt))
    monkeypatch.setattr(ds.offline_grpo_pool,'_verified_r0',lambda *args:None)
    monkeypatch.setattr(ds,'_reservation',lambda *args:(dict(reservation_id='fixture'),100))
    class Engine:
        def runtime(self):return runtime
        def engine(self,config,root):return self
        def score(self,answer):
            calls.append(answer['answer_id'])
            return dict(surprise=np.ones(len(answer['target_ids']),dtype=np.float64),entropy=np.ones(len(answer['target_ids']),dtype=np.float64),rank=np.ones(len(answer['target_ids']),dtype=np.int64))
        def check_integrity(self,full=False):return dict(base_unchanged=True)
        def vocabulary(self):return vocab
        def close(self):pass
    args=['--output-root',str(tmp_path)]
    assert main(['offline-grpo-dataset-prepare',*args,'--pool',str(tmp_path/'pool.json'),'--sft',str(tmp_path/'sft.json'),'--sampler',str(sampler),'--runtime',str(tmp_path/'runtime.json'),'--name','test'])==0
    config=tmp_path/'plans/test/offline.dataset.json';deps=Engine()
    assert main(['offline-grpo-dataset-run',*args,'--config',str(config),'--max-new-units','1'],dependencies=deps)==0
    for attempt in (tmp_path/'runs/offline-grpo/test/dataset/attempts').iterdir():
        for name in ['final-integrity.json','timing.json','timing.complete.json']:(attempt/name).unlink()
    assert main(['offline-grpo-dataset-run',*args,'--config',str(config)],dependencies=deps)==0
    assert main(['offline-grpo-dataset-verify',*args,'--config',str(config)])==0
    assert main(['offline-grpo-dataset-report',*args,'--config',str(config)])==0
    result=json.loads((tmp_path/'runs/offline-grpo/test/dataset/results/results.json').read_text())
    assert json.loads(config.read_text())['sft_root']=='.'
    assert calls==[answer2['answer_id']]
    assert result['full']['pool']['answers']==8 and result['full']['pool_incorrect']['answers']==8
    assert result['paired_common']['k5']['questions']==1 and result['full']['pool']['token_weighted']['top1']==1
    (tmp_path/'sft.json').write_text('{"changed": true}')
    assert main(['offline-grpo-dataset-verify',*args,'--config',str(config)])==1


def test_historical_seal_reader_checks_metadata_arrays_and_memberships(tmp_path):
    import json
    from pathlib import Path
    from pilot_eval import offline_grpo_dataset as ds
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    directory=tmp_path/'legacy';folder=directory/'units'/'answers-0000';folder.mkdir(parents=True)
    a=ds._answer([3],[4,99]);answer={k:a[k] for k in ['answer_id','input_ids','positions','prompt_tokens','target_ids']}
    members={k:[dict(problem_id='q',answer_id=a['answer_id'])] for k in ['gold','k0','k2','k5']}
    identity=dict(model='m',model_revision='r',tokenizer_revision='r',base_sha256='base')
    unit=dict(key='answers-0000',ids=[a['answer_id']])
    config=dict(protocol='sampling-stage0-dataset-v1',run_path='legacy',model_identity=identity,scored_eos_token_id=99,
        mask='stored prompt masked; supervised answer tokens through the single final end-of-turn',rank='1 + full-vocabulary logits strictly greater than the target; ties share rank',
        implementation={n:file_hash(Path(ds.__file__).with_name(n)) for n in ['stage0_distance_backend.py','stage0_distance_math.py','stage0_rollout_backend.py']},
        answers_sha256=_hash([answer]),memberships_sha256=_hash(members),units=[unit])
    path=tmp_path/'legacy.json';save(path,config);save(directory/'config.json',config)
    (directory/'inputs').mkdir();(directory/'inputs/answers.jsonl').write_text(json.dumps(answer)+'\n');save(directory/'inputs/memberships.json',members)
    save(directory/'meta/final-integrity.json',dict(base_unchanged=True));save(directory/'meta/runtime.json',dict(session='fixture'))
    save(directory/'meta/vocabulary.json',dict(includes_padded_rows=True,rank_and_softmax_same_full_vector=True,logit_vocabulary_size=100,
        rank_definition='1 + count of strictly greater entries in the full FP64-promoted FP32 logit vector used for softmax'))
    save(folder/'integrity.json',dict(base_unchanged=True))
    np.savez_compressed(folder/'tokens.npz',surprise_0=np.array([1.,2.]),entropy_0=np.array([2.,2.]),rank_0=np.array([1,2]))
    save(folder/'complete.json',dict(config_sha256=_hash(config),unit=unit,runtime_sha256=file_hash(directory/'meta/runtime.json'),sha256=file_hash(folder/'tokens.npz'),integrity_sha256=file_hash(folder/'integrity.json')))
    files=['config.json','inputs/answers.jsonl','inputs/memberships.json','meta/final-integrity.json','meta/runtime.json','meta/vocabulary.json','units/answers-0000/complete.json']
    save(directory/'complete.json',dict(config_sha256=_hash(config),files={f:file_hash(directory/f) for f in files}))
    answers,mapping,scores=ds._legacy(path,tmp_path,identity,99)
    assert mapping['k5'][0]['answer_id']==a['answer_id'] and len(scores[a['answer_id']]['rank'])==2
    save(folder/'integrity.json',dict(base_unchanged=False))
    with pytest.raises(ValueError,match='seal changed'):
        ds._legacy(path,tmp_path,identity,99)

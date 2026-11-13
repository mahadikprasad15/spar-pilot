from pilot_eval import sampling_sft_mixed_training_data as mixed
from pilot_eval import sampling_sft_training_data as original


def test_mixed_training_reuses_original_stored_id_mask_contract():
    assert mixed.supervised_row is original.supervised_row
    problem={'source_id':'s','problem_id':0,'scorer_prompt_ids':[10,11]}
    attempt={'end_state':'finished','response_ids':[12,3],'attempt_id':'k5:p0:r0'}
    row=mixed.supervised_row(problem,attempt,3)
    assert row['input_ids']==[10,11,12,3]
    assert row['labels']==[-100,-100,12,3]
    assert 'sampling_sft_training_data.py' in mixed.implementation()


def test_mixed_training_rows_match_original_for_full_cohort(tmp_path,monkeypatch):
    problems=[{'source_id':str(i),'problem_id':i,'scorer_prompt_ids':[10,11]} for i in range(512)]
    gold_rows=[{'id':str(i),'input_ids':[10,11,12,3],'prompt_tokens':2} for i in range(512)]
    attempts=[{'source_id':str(i),'problem_id':i,'end_state':'finished',
               'response_ids':[12,3],'attempt_id':f'p{i}'} for i in range(512)]
    batches=[{'step':j+1,'nonmissing_count':8,'slots':[
        {'source_id':str(i),'missing':False,'selected_attempt':f'p{i}'} for i in range(j*8,j*8+8)]}
        for j in range(64)]
    coverage={'coverage_blocked':False,'arms':{str(k):{'attempts':attempts,'batches':batches,
        'uncovered_ids':[],'coverage_decision':'passes-coverage-only'} for k in (0,2,5)}}
    gold={'source_config':{'decoding':{'eos_token_id':3}}}
    for module in (mixed,original):
        monkeypatch.setattr(module.review,'verify',lambda *a:coverage)
        monkeypatch.setattr(module.generation,'load',lambda *a:(tmp_path,{'problems':problems},tmp_path))
        monkeypatch.setattr(module,'load_sft',lambda *a:(gold,gold_rows,None))
    config={'review_plan':'review','generation_plan':'generation','gold_config':'gold','arms':[0,2,5]}
    assert mixed.expected(tmp_path,config)==original.expected(tmp_path,config)

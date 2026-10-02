import json
import pytest
from pilot_eval.scoring import score_gsm8k,score_gsm8k_flexible_v2

def write_pairs(path):
    pairs=[]
    def record(i,text):
        return dict(id=f'gsm8k:test:{i}',prompt=f'question {i}',gold='#### 3',generated_text=text,
            stop_reason='eos',token_count=5,score={**score_gsm8k(text,'#### 3'),
            'flexible_v2':score_gsm8k_flexible_v2(text,'#### 3')})
    for step in (0,8,16,32,64):
        for i in range(150):
            baseline=record(i,'Final answer: It takes 3 bolts.')
            adapted=baseline if step==0 else record(i,'2+1=3 bolts\n#### 3')
            pairs.append(dict(step=step,id=baseline['id'],baseline=baseline,adapted=adapted))
    path.write_text(''.join(json.dumps(p)+'\n' for p in pairs))

def test_cpu_audit_preserves_input_versions_and_repairs_format_failure(tmp_path):
    from pilot_eval.sft_score_audit import audit_sft_scores
    path=tmp_path/'paired.jsonl';write_pairs(path);before=path.read_bytes()
    result=audit_sft_scores(path,tmp_path,'audit')
    final=result['trajectory'][-1]
    assert final['strict_correct_v2_invalid']==150
    assert final['strict_correct_v3_invalid']==0
    assert final['flexible_v2_accuracy']==0
    assert final['flexible_v3_accuracy']==1
    assert final['flexible_v3_change']==0
    assert final['flexible_v3_change_95']==[0,0]
    assert result['post_hoc'] is True
    assert path.read_bytes()==before
    assert audit_sft_scores(path,tmp_path,'audit')==result
    assert (tmp_path/'reports/audit/results/paired-items.jsonl').is_file()
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    rows[-1]['adapted']['score']['strict']['correct']=False
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='score'):
        audit_sft_scores(path,tmp_path,'corrupt')
    assert not (tmp_path/'reports/corrupt').exists()

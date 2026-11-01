import json
from pilot_eval import sampling_sft_mixed_review as review


def test_review_reads_each_original_arm_and_preserves_human_requirement(tmp_path, monkeypatch):
    root=tmp_path;config=root/'mixed.json';config.write_text('{}')
    (root/'source.json').write_text(json.dumps({'run_path':'runs/source'}))
    plan={'source_plan':'source.json','problems':[{'problem_id':0,'user_message':'Q'}]}
    monkeypatch.setattr(review.generation,'load',lambda *a:(root,plan,root/'mixed'))
    monkeypatch.setattr(review.generation,'verify',lambda *a:{'state':'completed'})
    monkeypatch.setattr(review.generation,'arm_directory',lambda r,p,k:root/f'arm{k}')
    for k in (0,2,5):
        folder=root/f'arm{k}';folder.mkdir()
        row={'end_state':'finished','automatic_flexible_v3':{'status':'invalid'},
             'attempt_id':f'k{k}:p0:r0','problem_id':0,'source_id':'s',
             'retry_index':0,'text':'unclear'}
        for name in ('complete.json','dispositions.jsonl','results.json'):(folder/name).write_text('{}')
        (folder/'attempts.jsonl').write_text(json.dumps(row)+'\n')
    path=review.prepare(config,root,'mixed-review')
    _,saved,folder=review.load(path,root)
    packet=json.loads((folder/'reviewer/packet.json').read_text())
    assert [r['k'] for r in packet]==[0,2,5]
    assert saved['training_allowed'] is False
    assert all(x['status']=='pending' for x in review.current(saved,folder,'human')[0]['labels'])

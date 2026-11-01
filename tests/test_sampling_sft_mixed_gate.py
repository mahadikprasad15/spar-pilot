import json
from pilot_eval import sampling_sft_mixed_gate as gate


def test_gate_prepares_original_gold_and_mixed_review_requests(tmp_path, monkeypatch):
    root=tmp_path; review_file=root/'review.json';review_file.write_text('{}')
    review_dir=root/'review';(review_dir/'results').mkdir(parents=True)
    (review_dir/'complete.json').write_text('{}');(review_dir/'results/coverage.json').write_text('{}')
    (root/'source.json').write_text(json.dumps({'run_path':'runs/source'}))
    parent={'generation_plan':'mixed.json'}
    coverage={'arms':{str(k):{'attempts':[],'dispositions':[]} for k in (0,2,5)}}
    gen={'source_plan':'source.json','runtime_hash':'r','settings':{'runtime':{'actual_stop_ids':[3]}},
         'problems':[{'problem_id':i} for i in range(512)]}
    monkeypatch.setattr(gate.review,'load',lambda *a:(root,parent,review_dir))
    monkeypatch.setattr(gate.review,'verify',lambda *a:coverage)
    monkeypatch.setattr(gate.generation,'load',lambda *a:(root,gen,root/'gen'))
    monkeypatch.setattr(gate.source,'verify',lambda *a:{'audit':{'training':{'config_path':'gold.json'}}})
    gold=[{'prompt_tokens':1,'input_ids':[1,3],'id':str(i),'gold':'2'} for i in range(512)]
    monkeypatch.setattr(gate,'load_sft',lambda *a:(None,gold,None))
    policy=root/'policy.json';policy.write_text(json.dumps({'version':'unicode-words-v1','approved':True,'reviewer':'r','notes':'n'}))
    config=gate.prepare(review_file,root,policy,'mixed-gate')
    saved=json.loads(config.read_text())
    assert saved['protocol']=='sampling-sft-mixed-gate-v1'
    assert len(saved['units'])==512 and all(x['requests'][0]['source']=='original-loss-bearing-SFT-token-IDs' for x in saved['units'])
    assert saved['training_authorized'] is False


def test_gate_report_reads_original_arm_decisions(tmp_path, monkeypatch):
    from pilot_eval import sampling_sft_mixed_gate_report as report
    monkeypatch.setattr(report.generation,'load',lambda *a:(tmp_path,{},tmp_path/'mixed'))
    monkeypatch.setattr(report.generation,'arm_directory',lambda root,plan,k:root/f'arm{k}')
    for k in (0,2,5):
        path=tmp_path/f'arm{k}/rounds/0/blocks/000';path.mkdir(parents=True)
        (path/'scientific.jsonl').write_text(json.dumps({'decision':{'accepted':True,'step':0,'cut':0,'old_ids':[1]}})+'\n')
    coverage={'arms':{str(k):{'attempts':[],'covered_count':512,'coverage_decision':'pass'} for k in (0,2,5)}}
    views=report.collection_views(tmp_path,{'generation_plan':'mixed.json'},coverage)
    assert all(views[f'k{k}']['acceptance']['numerator']==1 for k in (0,2,5))

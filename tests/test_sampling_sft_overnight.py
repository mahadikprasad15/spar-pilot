import pytest


def test_exact_accepted_coverage_and_flags():
    from pilot_eval.sampling_sft_overnight import validate_coverage
    report={'coverage':{'blocked':False,'uncovered_ids':{'0':list(range(34)),'2':list(range(10))+list(range(34,59)),
        '5':list(range(10))+list(range(34,37))+list(range(59,86))},
        'pairwise_flags':{'0-2':{'count':49,'flagged':True},'0-5':{'count':54,'flagged':True},'2-5':{'count':49,'flagged':True}},
        't16_required':{'0':True,'2':True,'5':True}}}
    validate_coverage(report)
    report['coverage']['uncovered_ids']['2'][0]=1000
    with pytest.raises(ValueError):validate_coverage(report)
    report['coverage']['uncovered_ids']['2'][0]=0
    report['coverage']['uncovered_ids']['5'].pop()
    with pytest.raises(ValueError):validate_coverage(report)


def test_orchestration_stops_and_resumes_without_duplicate_score(tmp_path):
    from pilot_eval.sampling_sft_overnight import orchestrate
    class Ops:
        def __init__(self):self.calls=[];self.fail=True
        def preflight(self):self.calls.append('preflight')
        def score(self):self.calls.append('score')
        def report_selection(self):
            self.calls.append('selection')
            if self.fail:raise ValueError('stop')
            return 'selection.json'
        def training(self,selection):self.calls.append('training');return 'inventory.json'
    ops=Ops()
    with pytest.raises(ValueError):orchestrate(tmp_path,ops)
    ops.fail=False;orchestrate(tmp_path,ops)
    assert ops.calls==['preflight','score','selection','preflight','selection','training']


def test_preflight_failure_prevents_all_jobs(tmp_path):
    from pilot_eval.sampling_sft_overnight import orchestrate
    class Ops:
        def preflight(self):raise ValueError('provider not authenticated')
        def score(self):pytest.fail('must not run a paid job')
    with pytest.raises(ValueError,match='provider not authenticated'):orchestrate(tmp_path,Ops())


def test_checkpoint_copy_does_not_copy_metadata(tmp_path,monkeypatch):
    from pilot_eval.sampling_sft_training_backend import copy_payload
    src=tmp_path/'src';src.write_bytes(b'checkpoint');dst=tmp_path/'dst'
    monkeypatch.setattr('shutil.copystat',lambda *a,**k: (_ for _ in ()).throw(PermissionError()))
    copy_payload(src,dst)
    assert dst.read_bytes()==b'checkpoint'


def test_checkpoint_copy_detects_corruption(tmp_path,monkeypatch):
    from pilot_eval.sampling_sft_training_backend import copy_payload
    src=tmp_path/'src';src.write_bytes(b'checkpoint');dst=tmp_path/'dst'
    monkeypatch.setattr('shutil.copyfile',lambda src,dst: dst.write_bytes(b'corrupt'))
    with pytest.raises(ValueError,match='hash mismatch'):copy_payload(src,dst)


def test_report_approval_binds_real_request_and_all_flags(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_overnight as overnight
    ids={'0':list(range(34)),'2':list(range(10))+list(range(34,59)),
        '5':list(range(10))+list(range(34,37))+list(range(59,86))}
    report={'execution_kind':'real-cuda','coverage':{'blocked':False,'uncovered_ids':ids,
        'pairwise_flags':{k:{'count':n,'flagged':True} for k,n in overnight.DIFFERENCES.items()},
        't16_required':dict.fromkeys(ids,True)}}
    request={'request_path':'request.json','request_sha256':'request-hash','report_sha256':'report-hash',
        'budget_snapshot':{'sha256':'budget-hash'},'required_acknowledgments':overnight.FLAGS,'coverage_blocked':False}
    monkeypatch.setattr(overnight.pipeline,'gate_report',lambda *args:None)
    monkeypatch.setattr(overnight.gate,'verify',lambda *args:report)
    monkeypatch.setattr(overnight.gate,'request',lambda *args:request)
    def approve(config,root,path):
        import json
        value=json.loads(path.read_text())
        assert value['request_sha256']=='request-hash' and value['report_sha256']=='report-hash'
        assert value['budget_sha256']=='budget-hash' and value['acknowledged_coverage_flags']==overnight.FLAGS
        return {'selection_path':'selection.json'}
    monkeypatch.setattr(overnight.gate,'approve',approve)
    cfg={'root':str(tmp_path),'gate':'gate.json','score_job':'job.json','data':'data.json','state_dir':'state',
        'rate':'1','hours':dict.fromkeys(ids,'2'),'setup_hours':'0','reviewer':'Prasad'}
    assert overnight.Operations(cfg).report_selection()=='selection.json'

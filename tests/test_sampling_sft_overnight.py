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
        stop_requested=False
        def __init__(self):self.calls=[];self.fail=True
        def preflight(self):self.calls.append('preflight')
        def score(self):self.calls.append('score')
        def report_selection(self):
            self.calls.append('selection')
            if self.fail:raise ValueError('stop')
            return 'selection.json'
        def training(self,selection):self.calls.append('training');return 'inventory.json'
        def export(self,inventory):
            self.calls.append('export');return dict(repo='r',revision='0'*40,manifest_sha256='m')
    ops=Ops()
    with pytest.raises(ValueError):orchestrate(tmp_path,ops)
    ops.fail=False;orchestrate(tmp_path,ops)
    assert ops.calls==['preflight','score','selection','preflight','selection','training','export']
    orchestrate(tmp_path,ops)
    assert ops.calls[-1]=='preflight' # Completed stages are never repeated on a rerun.


def test_pod_stop_only_after_verified_export(tmp_path):
    import json
    from pilot_eval.sampling_sft_overnight import orchestrate
    class Ops:
        stop_requested=True
        def __init__(self):self.calls=[];self.fail=True
        def preflight(self):pass
        def score(self):pass
        def report_selection(self):return 'selection.json'
        def training(self,selection):return 'inventory.json'
        def export(self,inventory):
            if self.fail:raise ValueError('HF unavailable')
            return dict(repo='r',revision='0'*40,manifest_sha256='m')
        def stop(self):
            assert (tmp_path/'complete.json').exists();self.calls.append('stop')
    ops=Ops()
    with pytest.raises(ValueError,match='HF'):orchestrate(tmp_path,ops)
    assert ops.calls==[] and not (tmp_path/'complete.json').exists()
    assert (tmp_path/'training.complete.json').exists() # Training is preserved, not repeated.
    ops.fail=False;orchestrate(tmp_path,ops)
    assert ops.calls==['stop']
    assert json.loads((tmp_path/'complete.json').read_text())['external_copy_verified'] is True
    states=[json.loads(l)['state'] for l in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert states[-3:]==['exported-verified','pod-stop-requested','pod-stop-command-returned']


def _main_args(tmp_path):
    for n in ['gate.json','job.json','data.json']:(tmp_path/n).write_text('{}')
    return ['--root',str(tmp_path),'--gate','gate.json','--score-job','job.json','--data','data.json','--name','night',
        '--volume-id','v','--rate','1','--reviewer','r','--hours-k0','1','--hours-k2','1','--hours-k5','1','--setup-hours','0']


def test_second_wrapper_never_overwrites_live_status(tmp_path):
    import json
    from pilot_eval import sampling_sft_overnight as overnight
    from pilot_eval.training import run_lock
    args=_main_args(tmp_path);folder=tmp_path/'runs/sampling-sft/overnight/night'
    with run_lock(folder):
        (folder/'status.json').write_text(json.dumps({'state':'training'}))
        assert overnight.main(args)==2
        assert json.loads((folder/'status.json').read_text())=={'state':'training'}


def test_unexpected_crash_records_failed_stage_and_traceback(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_overnight as overnight
    def crash(folder,ops):
        overnight.record(folder,'training');raise TypeError('bug in an unexpected place')
    monkeypatch.setattr(overnight,'orchestrate',crash)
    assert overnight.main(_main_args(tmp_path))==1
    folder=tmp_path/'runs/sampling-sft/overnight/night'
    value=json.loads((folder/'status.json').read_text())
    assert value['state']=='stopped' and value['failed_stage']=='training' and value['error_type']=='TypeError'
    assert 'bug in an unexpected place' in (folder/'failure-traceback.txt').read_text()


def test_fresh_download_verification_checks_every_file(tmp_path,monkeypatch):
    import io,tarfile,json
    from pilot_eval import sampling_sft_overnight as overnight
    from pilot_eval.training import file_hash
    root=tmp_path/'root';(root/'a').mkdir(parents=True);(root/'a/x.bin').write_bytes(b'adapter')
    manifest_path=root/'m.json';manifest_path.write_text('{}')
    manifest={'files':{'a/x.bin':{'sha256':file_hash(root/'a/x.bin'),'bytes':7}}}
    def build(payload):
        archive=tmp_path/'bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for name,data in [('a/x.bin',payload),('m.json',b'{}')]:
                info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
        return archive
    def fetch(repo,name,**kwargs):
        assert kwargs['force_download'] is True and kwargs['revision']=='0'*40;return str(state['archive'])
    monkeypatch.setattr('huggingface_hub.hf_hub_download',fetch)
    state={'archive':build(b'adapter')}
    rec=dict(repo='r',archive='exports/m/bundle.tar.gz',revision='0'*40,archive_sha256=file_hash(state['archive']),archive_bytes=100)
    assert overnight.verify_download(rec,manifest,manifest_path,root)['verified_files']==2
    state['archive']=build(b'corrupt');rec['archive_sha256']=file_hash(state['archive'])
    with pytest.raises(ValueError,match='differs from the sealed manifest'):overnight.verify_download(rec,manifest,manifest_path,root)


def test_approval_notes_do_not_claim_human_metric_review():
    from pilot_eval.sampling_sft_overnight import APPROVAL_NOTES
    assert 'has not reviewed these gate metrics' in APPROVAL_NOTES and 'not human-confirmed' in APPROVAL_NOTES


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

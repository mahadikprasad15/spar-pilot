"""Shared offline attempt lifecycle, reuse rule, hashing and path helpers."""
import json
import os

import pytest

from pilot_eval.training import file_hash


class Engine:
    def __init__(self):self.full=0;self.closed=0
    def check_integrity(self,full=False):
        self.full+=1;return dict(base_unchanged=True)
    def close(self):self.closed+=1


def scope(tmp_path,**kwargs):
    from pilot_eval.offline_grpo_support import attempt_scope
    return attempt_scope(tmp_path,dict(reservation_id='r'),dict(session_id='s'),kwargs.pop('remaining',60),
                         real=False,on_failure=kwargs.pop('on_failure',lambda exc:None))


def test_ordinary_failure_runs_final_check_even_if_message_says_integrity(tmp_path):
    engine=Engine();seen=[]
    from pilot_eval.offline_grpo_support import attempt_scope
    with pytest.raises(ValueError,match='integrity'):
        with attempt_scope(tmp_path,dict(reservation_id='r'),{},60,real=False,on_failure=seen.append) as run:
            run.engine=engine
            raise ValueError('tokenizer integrity wording only')
    assert engine.full==1 and engine.closed==1 and len(seen)==1
    assert (run.path/'final-integrity.json').exists() and (run.path/'timing.complete.json').exists()


def test_integrity_failure_skips_final_check_but_records_timing(tmp_path):
    from pilot_eval.offline_grpo_support import IntegrityFailure
    engine=Engine()
    with pytest.raises(IntegrityFailure):
        with scope(tmp_path) as run:
            run.engine=engine
            raise IntegrityFailure('base changed')
    assert engine.full==0
    from pilot_eval.offline_grpo_support import attempt_start
    assert json.loads((run.path/'timing.json').read_text())['final_integrity_sha256']==file_hash(run.path/'final-integrity.json')
    with pytest.raises(ValueError,match='invalidates'):attempt_start(run.path,['base_unchanged'])


def test_no_final_check_after_allocation_is_spent(tmp_path):
    engine=Engine()
    with pytest.raises(TimeoutError):
        with scope(tmp_path,remaining=0.) as run:
            run.engine=engine
            raise TimeoutError('expired')
    assert engine.full==0


def attempt(tmp_path,*,final=None,timed=True):
    from pilot_eval.run import _write_json
    path=tmp_path/'attempt';path.mkdir();_write_json(path/'start.json',dict(reservation_id='r'))
    if final is not None:_write_json(path/'final-integrity.json',final)
    if timed:
        recorded=file_hash(path/'final-integrity.json') if final is not None else None
        _write_json(path/'timing.json',dict(seconds=1.,start_sha256=file_hash(path/'start.json'),final_integrity_sha256=recorded))
        _write_json(path/'timing.complete.json',dict(sha256=file_hash(path/'timing.json')))
    return path


def test_killed_or_expired_attempt_units_stay_reusable(tmp_path):
    from pilot_eval.offline_grpo_support import attempt_start
    killed=attempt(tmp_path,timed=False)
    assert attempt_start(killed,['base_unchanged'])==file_hash(killed/'start.json')
    (tmp_path/'x').mkdir()
    expired=attempt(tmp_path/'x')
    assert attempt_start(expired,['base_unchanged'])==file_hash(expired/'start.json')


def test_failed_or_removed_final_check_still_blocks_reuse(tmp_path):
    from pilot_eval.offline_grpo_support import attempt_start
    failed=attempt(tmp_path,final=dict(base_unchanged=False))
    with pytest.raises(ValueError,match='invalidates'):attempt_start(failed,['base_unchanged'])
    (tmp_path/'y').mkdir()
    removed=attempt(tmp_path/'y',final=dict(base_unchanged=True))
    (removed/'final-integrity.json').unlink()
    with pytest.raises(ValueError,match='changed'):attempt_start(removed,['base_unchanged'])


def test_pinned_hash_reuses_only_unchanged_files(tmp_path,monkeypatch):
    from pilot_eval import offline_grpo_support as support
    path=tmp_path/'weights.bin';path.write_bytes(b'first')
    calls=[];real=support.file_hash
    monkeypatch.setattr(support,'file_hash',lambda p:calls.append(p) or real(p))
    assert support.pinned_hash(path)==support.pinned_hash(path)==real(path)
    assert len(calls)==1
    path.write_bytes(b'second, longer')
    assert support.pinned_hash(path)==real(path) and len(calls)==2


def test_source_roots_relocate_inside_the_artifact_root(tmp_path):
    from pilot_eval.offline_grpo_support import portable_root,resolve_root
    inside=tmp_path/'run'/'sources';outside=tmp_path/'elsewhere'
    assert portable_root(tmp_path/'run',inside)=='sources'
    assert resolve_root(tmp_path/'moved','sources')==(tmp_path/'moved'/'sources').resolve()
    assert portable_root(tmp_path/'run',outside)==str(outside.resolve())
    assert resolve_root(tmp_path/'moved',str(outside))==outside.resolve()


def test_failed_integrity_result_is_not_overwritten_by_failure_record(tmp_path):
    from pilot_eval.offline_grpo_support import IntegrityFailure
    from pilot_eval.run import _write_json
    with pytest.raises(IntegrityFailure):
        with scope(tmp_path) as run:
            _write_json(run.path/'final-integrity.json',dict(base_unchanged=False,kind='full'))
            raise IntegrityFailure('base changed')
    assert json.loads((run.path/'final-integrity.json').read_text())['kind']=='full'

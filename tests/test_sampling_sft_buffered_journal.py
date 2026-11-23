import json

import pytest

from pilot_eval.sampling_sft_instrument import append
from pilot_eval.sampling_sft_buffered_journal import BufferedJournal


def test_journal_matches_current_writer_and_syncs_before_seal(tmp_path):
    rows = [{'problem_id': 14, 'old_ids': [1, 2], 'proposal_score': -0.6},
            {'text': 'μ\nanswer', 'accepted': False}]
    old, new = tmp_path / 'old', tmp_path / 'new'
    for row in rows:
        append(old, row)
    with BufferedJournal(new, max_bytes=80) as journal:
        for row in rows:
            journal.append(row)
        journal.sync()
    assert new.read_bytes() == old.read_bytes()
    assert [json.loads(line) for line in new.read_text().splitlines()] == rows


def test_failed_write_is_not_sealable(tmp_path, monkeypatch):
    journal = BufferedJournal(tmp_path / 'partial', max_bytes=1000)
    journal.append({'id': 1})
    def fail(*args):
        raise OSError('disk failed')
    monkeypatch.setattr('os.write', fail)
    with pytest.raises(OSError):
        journal.sync()
    with pytest.raises(RuntimeError):
        journal.sync()
    journal.close()


def test_short_writes_and_partial_block_replay(tmp_path, monkeypatch):
    import os
    actual_write = os.write
    monkeypatch.setattr(os, 'write', lambda fd, data: actual_write(fd, data[:3]))
    rows = [{'id': 1}, {'id': 2}]
    partial = tmp_path / 'partial'
    with BufferedJournal(partial, max_bytes=8) as journal:
        journal.append(rows[0])
        # Simulate interruption before block sync and seal.
    partial.rename(tmp_path / 'abandoned')
    with BufferedJournal(partial, max_bytes=8) as journal:
        for row in rows:
            journal.append(row)
        journal.sync()
    partial.rename(tmp_path / 'sealed')
    assert (tmp_path / 'sealed').read_bytes() == b'{"id": 1}\n{"id": 2}\n'


def test_benchmark_compares_identical_artifacts(tmp_path):
    from pilot_eval.sampling_sft_journal_benchmark import benchmark
    result = benchmark(tmp_path, rows=8, token_count=20, max_bytes=256)
    assert result['equal_bytes'] is True
    assert result['old']['fsync_count'] == 8
    assert result['buffered']['fsync_count'] == 1
    assert result['old']['bytes'] == result['buffered']['bytes']


def test_sync_error_cannot_be_sealed_and_replay_is_complete(tmp_path, monkeypatch):
    import os
    partial = tmp_path / 'partial'
    real_sync = os.fsync
    def fail(_):
        raise OSError('sync failed')
    monkeypatch.setattr(os, 'fsync', fail)
    with BufferedJournal(partial) as journal:
        journal.append({'decision': 1})
        with pytest.raises(OSError):
            journal.sync()
        with pytest.raises(RuntimeError):
            journal.append({'decision': 2})
    partial.rename(tmp_path / 'abandoned')
    monkeypatch.setattr(os, 'fsync', real_sync)
    with BufferedJournal(partial) as journal:
        journal.append({'decision': 1})
        journal.append({'decision': 2})
        journal.sync()
    partial.rename(tmp_path / 'sealed')
    assert (tmp_path / 'sealed').read_bytes() == b'{"decision": 1}\n{"decision": 2}\n'
    # A completed seal is retained instead of replayed.
    with pytest.raises(FileExistsError):
        BufferedJournal(tmp_path / 'sealed')


def test_serialization_failure_cannot_be_sealed(tmp_path):
    with BufferedJournal(tmp_path / 'partial') as journal:
        with pytest.raises(ValueError):
            journal.append({'score': float('nan')})
        with pytest.raises(RuntimeError):
            journal.sync()

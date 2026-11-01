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

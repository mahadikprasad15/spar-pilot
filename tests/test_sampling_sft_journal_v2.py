import json

import pytest

from pilot_eval.sampling_sft_instrument import append
from pilot_eval.sampling_sft_journal_v2 import BlockJournal, make_plan, verify_block
from pilot_eval.sampling_sft_sampler import sample
from test_sampling_sft_sampler import Engine, Tokenizer, generation, problem


def test_real_sampler_records_match_old_writer_and_bound_plan(tmp_path):
    plan = make_plan('fixture-v2', max_bytes=128)
    old = tmp_path / 'old.jsonl'
    with BlockJournal(tmp_path / 'new', plan, 0) as block:
        def record(value):
            value = json.loads(json.dumps(value)); value.pop('replay', None)
            append(old, value); block.record(value)
        result = sample([problem()], 0, Engine([generation([3], 'stop', 3)]),
                        Tokenizer(), [3], on_record=record)
        block.seal({'next_block': 1, 'states': result['traces']})
    assert old.read_bytes() == (tmp_path/'new/scientific.jsonl').read_bytes()
    assert verify_block(tmp_path/'new', plan)['next_block'] == 1
    with pytest.raises(ValueError):
        verify_block(tmp_path/'new', make_plan('other-v2', max_bytes=128))


def test_failure_never_seals_and_sealed_record_tamper_fails(tmp_path, monkeypatch):
    plan = make_plan('fixture-v2', max_bytes=128)
    with BlockJournal(tmp_path/'failed', plan, 0) as block:
        block.record({'decision': 1})
        monkeypatch.setattr('os.fsync', lambda fd: (_ for _ in ()).throw(OSError('fail')))
        with pytest.raises(OSError): block.seal({'next_block': 1})
    assert not (tmp_path/'failed/complete.json').exists()
    monkeypatch.undo()
    with BlockJournal(tmp_path/'good', plan, 0) as block:
        block.record({'decision': 1}); block.seal({'next_block': 1})
    (tmp_path/'good/scientific.jsonl').write_text('tampered\n')
    with pytest.raises(ValueError): verify_block(tmp_path/'good', plan)

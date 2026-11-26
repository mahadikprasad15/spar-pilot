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


@pytest.mark.parametrize('seal_first', [False, True])
def test_sigkill_preserves_only_sealed_fixture_block(tmp_path, seal_first):
    import subprocess
    import sys
    path = tmp_path / 'block'
    program = '''
import os, signal, sys
from pilot_eval.sampling_sft_journal_v2 import BlockJournal, make_plan
p=make_plan('kill-fixture', max_bytes=16)
with BlockJournal(sys.argv[1], p, 0) as b:
    b.record({'decision': 1})
    if sys.argv[2]=='yes': b.seal({'next_block': 1})
    os.kill(os.getpid(), signal.SIGKILL)
'''
    result = subprocess.run([sys.executable, '-c', program, str(path),
                             'yes' if seal_first else 'no'], check=False)
    assert result.returncode < 0
    plan = make_plan('kill-fixture', max_bytes=16)
    if seal_first:
        assert verify_block(path, plan)['next_block'] == 1
    else:
        assert not (path/'complete.json').exists()
        path.rename(tmp_path/'abandoned')
        with BlockJournal(path, plan, 0) as block:
            block.record({'decision': 1}); block.seal({'next_block': 1})
        assert verify_block(path, plan)['next_block'] == 1
        assert (path/'scientific.jsonl').read_bytes() == b'{"decision": 1}\n'

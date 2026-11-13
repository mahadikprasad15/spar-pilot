"""CPU-only versioned block-journal integration experiment, not a production plan.

Existing generation, gate and training verifiers deliberately do not accept
these fixture blocks. No cross-plan import or GPU authorization is provided.
"""
import hashlib
import json
import os
from pathlib import Path

from pilot_eval.sampling_sft_buffered_journal import BufferedJournal


def digest(value):
    return hashlib.sha256(value).hexdigest()


def make_plan(name, *, max_bytes):
    if not name or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('name and positive max_bytes required')
    here = Path(__file__).parent
    return {'protocol': 'sampling-sft-journal-fixture-v2', 'name': name,
            'writer': 'buffered-before-seal-v1', 'max_bytes': max_bytes,
            'implementation': {p: digest((here/p).read_bytes()) for p in
                ('sampling_sft_journal_v2.py', 'sampling_sft_buffered_journal.py',
                 'sampling_sft_sampler.py')}, 'production_ready': False}


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def write_once(path, data):
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


class BlockJournal:
    def __init__(self, folder, plan, index):
        if plan != make_plan(plan['name'], max_bytes=plan['max_bytes']) or index < 0:
            raise ValueError('fixture plan identity changed')
        self.folder = Path(folder); self.folder.mkdir()
        self.plan, self.index, self.sealed = plan, index, False
        self.journal = BufferedJournal(self.folder/'scientific.jsonl', max_bytes=plan['max_bytes'])

    def record(self, value):
        if self.sealed: raise RuntimeError('already sealed')
        self.journal.append(value)

    def seal(self, state):
        if self.sealed or state.get('next_block') != self.index+1:
            raise ValueError('invalid block boundary')
        self.journal.sync()
        self.journal.close()
        state_bytes = encoded(state)
        write_once(self.folder/'state.json', state_bytes)
        proof = {'plan_sha256': digest(encoded(self.plan)), 'index': self.index,
                 'state_sha256': digest(state_bytes),
                 'scientific_sha256': digest((self.folder/'scientific.jsonl').read_bytes())}
        write_once(self.folder/'complete.json', encoded(proof))
        self.sealed = True

    def __enter__(self): return self
    def __exit__(self, *args): self.journal.close()


def verify_block(folder, plan):
    folder = Path(folder)
    if plan != make_plan(plan['name'], max_bytes=plan['max_bytes']):
        raise ValueError('fixture plan implementation changed')
    proof = json.loads((folder/'complete.json').read_bytes())
    state_bytes = (folder/'state.json').read_bytes()
    if (proof['plan_sha256'] != digest(encoded(plan)) or
        proof['state_sha256'] != digest(state_bytes) or
        proof['scientific_sha256'] != digest((folder/'scientific.jsonl').read_bytes()) or
        json.loads(state_bytes)['next_block'] != proof['index']+1):
        raise ValueError('fixture block identity changed')
    return json.loads(state_bytes)

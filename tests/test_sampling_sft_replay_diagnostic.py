import json
from pathlib import Path

from pilot_eval.sampling_sft_replay_diagnostic import diagnose


def test_diagnostic_localizes_scores_decisions_and_unaligned_requests(tmp_path):
    root = tmp_path / 'pilot'
    def write(lane, block, rows, traces):
        p = root / lane / 'blocks' / f'{block:03d}'
        p.mkdir(parents=True)
        (p / 'scientific.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (root / lane / 'results.json').write_text(json.dumps({'traces': traces}))
    def score(value, ids=[7], step=-1):
        return {'kind':'score','requests':[{'problem_id':1,'block':0,'step':step,'prompt_ids':[1],'response_ids':ids}],
                'outputs':[{'token_ids':ids,'positions':[1],'logprobs':[value]}]}
    def decision(accepted):
        return {'decision':{'problem_id':1,'block':0,'step':0,'cut':0,'old_ids':[7],
                            'proposal_ids':[8],'old_score':-1.,'proposal_score':-.9,'accepted':accepted}}
    write('uninterrupted',0,[score(-.6),decision(False),score(-.5,[8],0)],[{'problem_id':1,'response_ids':[7],'text':'a'}])
    write('resumed',0,[score(-.60002),decision(True),score(-.4,[9],0)],[{'problem_id':1,'response_ids':[8],'text':'b'}])
    result = diagnose(root)
    assert result['scores']['different_values'] == 1
    assert result['scores']['max_abs_difference'] > .000019
    assert result['scores']['first_difference']['problem_id'] == 1
    assert result['blocks']['1']['phase'] == 'before_interruption'
    assert result['decisions']['accepted_differences'] == 1
    assert result['final_traces']['different_problems'] == 1
    assert result['structure']['unmatched_requests'] == 2


def test_cli_never_overwrites_and_preserves_sources(tmp_path):
    import subprocess
    import sys
    import hashlib
    pilot = tmp_path / 'pilot'
    for lane in ('uninterrupted', 'resumed'):
        block = pilot / lane / 'blocks' / '000'
        block.mkdir(parents=True)
        (block / 'scientific.jsonl').write_text('')
        (pilot / lane / 'results.json').write_text('{"traces": []}')
    source = pilot / 'uninterrupted/blocks/000/scientific.jsonl'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    command = [sys.executable, '-m', 'pilot_eval.sampling_sft_replay_diagnostic',
               '--pilot-dir', str(pilot), '--name', 'inspection']
    assert subprocess.run(command, capture_output=True).returncode == 0
    result = json.loads((pilot / 'diagnostics/inspection/summary.json').read_text())
    assert result['structure']['integrity_valid'] is False
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before

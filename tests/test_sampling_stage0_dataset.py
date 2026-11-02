import json

import numpy as np
import pytest

from pilot_eval import sampling_stage0_dataset as dataset
from pilot_eval import sampling_stage0_evaluation as evaluation

EOS = 2


def row(pid, answer):
    prompt = [5, 6]
    return {'id': pid, 'input_ids': prompt+answer+[EOS], 'labels': [-100]*2+answer+[EOS], 'prompt_tokens': 2}


def rows():
    gold = [row(f'p{i}', [10+i]) for i in range(8)]
    arms = {'k0': [row(f'p{i}', [100+i, 7]) for i in range(8) if i != 7],
            'k2': [row(f'p{i}', [200+i, 7]) for i in range(8) if i != 6],
            'k5': [row(f'p{i}', [300+i, 7]) for i in range(8) if i != 5]}
    return {'gold': gold, **arms}


def setup(tmp_path, monkeypatch, data=None):
    monkeypatch.setattr(dataset, 'COMMON_PROBLEMS', 5)
    monkeypatch.setattr(evaluation, 'check_sources', lambda config, root: [])
    behaviour = {'protocol': 'sampling-stage0-evaluation-v2', 'decoding': {'eos_token_id': EOS}, 'gpu': 'L4',
                 **{k: {} for k in ['parents', 'training_plans', 'variants', 'adapter_files', 'model_identity', 'model_snapshot']},
                 'cohort_path': 'c', 'cohort_sha256': 'h', 'weights_dir': 'w'}
    (tmp_path/'plans').mkdir(); (tmp_path/'plans/behaviour.json').write_text(json.dumps(behaviour))
    return dataset.prepare('plans/behaviour.json', tmp_path, 'night', unit_size=4, load_rows=lambda root, b: (data or rows(), {}))


class Engine:
    def __init__(self, crash_after=None): self.scored = 0; self.crash_after = crash_after
    def vocabulary(self): return {'logit_vocabulary_size': 1000}
    def score(self, answer):
        if self.crash_after is not None and self.scored >= self.crash_after: raise RuntimeError('pod stopped')
        self.scored += 1; n = len(answer['target_ids']); gold = n == 2 # gold answers: one token + EOS
        return {'surprise': np.full(n, 2.0 if gold else 1.0), 'entropy': np.full(n, 0.5), 'rank': np.ones(n, dtype=np.int64)}
    def check_integrity(self, *, full=False): return {'base_unchanged': True}


class Dependencies:
    def __init__(self, engine, session='boot-1'): self.model = engine; self.session = session
    def runtime(self): return {'execution_kind': 'real-cuda', 'device': 'NVIDIA L4', 'driver': '580', 'session_id': self.session, 'git_commit': 'c'}
    def engine(self, config, root): return self.model


def test_scores_every_unique_answer_and_reports_paired_differences(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    engine = Engine()
    result = dataset.run(path, tmp_path, dependencies=Dependencies(engine))
    assert engine.scored == 8+7*3 and result['unique_answers'] == 29
    covered = {(s['dataset'], s['scope']): s for s in result['summaries']}
    assert covered[('gold', 'covered')]['n_problems'] == 8 and covered[('k0', 'common')]['n_problems'] == 5
    assert covered[('k2', 'covered')]['nll'] == 1.0 and covered[('gold', 'common')]['excess_surprise'] == 1.5
    nll = next(p for p in result['paired_differences'] if p['arm'] == 'k5' and p['metric'] == 'nll')
    assert nll['difference'] == -1.0 and nll['ci'] == [-1.0, -1.0] and nll['n_problems'] == 5
    assert dataset.verify(path, tmp_path)['paired_differences'] == result['paired_differences']
    summary = tmp_path/'runs/stage0/night/dataset/results/summary.json'
    summary.write_text(summary.read_text().replace('"nll": 1.0', '"nll": 0.9', 1))
    with pytest.raises(ValueError, match='changed'):
        dataset.verify(path, tmp_path)


def test_answer_mask_or_end_of_turn_violation_stops_preparation(tmp_path, monkeypatch):
    bad = rows(); bad['k2'][0] = {**bad['k2'][0], 'labels': [-100]*3+bad['k2'][0]['input_ids'][3:]}
    with pytest.raises(ValueError, match='mask/end-of-turn'):
        setup(tmp_path, monkeypatch, data=bad)


def test_common_problem_count_must_match_accepted_coverage(tmp_path, monkeypatch):
    fewer = rows(); fewer['k0'] = fewer['k0'][1:] # p0 is covered by every arm
    with pytest.raises(ValueError, match='commonly covered'):
        setup(tmp_path, monkeypatch, data=fewer)


def test_resume_scores_only_unsealed_units(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='pod stopped'):
        dataset.run(path, tmp_path, dependencies=Dependencies(Engine(crash_after=10)))
    resumed = Engine()
    dataset.run(path, tmp_path, dependencies=Dependencies(resumed, session='boot-2'))
    assert resumed.scored == 29-8 # two sealed units of four answers were kept

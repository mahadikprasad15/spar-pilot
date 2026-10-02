"""CPU-only, post-hoc scoring audit of exported Pilot 2 paired responses."""

import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from pilot_eval.recovery import _name
from pilot_eval.rescoring import _save_bytes
from pilot_eval.run import _wilson_95
from pilot_eval.scoring import score_gsm8k, score_gsm8k_flexible_v2, score_gsm8k_flexible_v3
from pilot_eval.sft_reporting import _interval
from pilot_eval.workflow import _save_frozen


def _rescore(record):
    if (not isinstance(record['generated_text'], str)
            or record['stop_reason'] not in ('eos', 'cap', 'other')
            or type(record['token_count']) is not int or record['token_count'] < 0):
        raise ValueError('invalid response fields')
    capped = record['stop_reason'] == 'cap'
    expected = {**score_gsm8k(record['generated_text'], record['gold'], capped),
                'flexible_v2': score_gsm8k_flexible_v2(record['generated_text'], record['gold'], capped)}
    if record['score'] != expected:
        raise ValueError('saved source score mismatch')
    new = score_gsm8k_flexible_v3(record['generated_text'], record['gold'], capped)
    if (expected['strict']['status'] == 'valid'
            and (new['status'] != 'valid' or new['correct'] != expected['strict']['correct'])):
        raise ValueError('strict acceptance invariant failed')
    return {**record, 'score': {**expected, 'flexible_v3': new}}


def audit_sft_scores(paired_path, output_root, name):
    """Verify all legacy scores and preserve a separate v3 report and source copy.

    An exported paired file establishes response-level consistency, but cannot
    independently verify training configuration or the original run hashes.
    """
    source = Path(paired_path).resolve()
    content = source.read_bytes()
    pairs = [json.loads(line) for line in content.splitlines()]
    steps = (0, 8, 16, 32, 64)
    if len(pairs) != 750 or {p['step'] for p in pairs} != set(steps):
        raise ValueError('expected five complete 150-item checkpoint cohorts')
    groups = {step: [p for p in pairs if p['step'] == step] for step in steps}
    ids = [p['id'] for p in groups[0]]
    if len(ids) != 150 or len(set(ids)) != 150:
        raise ValueError('duplicate or incomplete item IDs')
    base = {p['id']: p['baseline'] for p in groups[0]}
    rescored, disagreements = [], []
    cohorts = {'baseline': [_rescore(base[item]) for item in ids]}
    for step, group in groups.items():
        if len(group) != 150 or set(p['id'] for p in group) != set(ids):
            raise ValueError('checkpoint item alignment mismatch')
        mapped = {p['id']: p for p in group}
        records = []
        for item in ids:
            pair = mapped[item]
            before, after = pair['baseline'], pair['adapted']
            if before != base[item] or before['id'] != item or after['id'] != item:
                raise ValueError('baseline or item identity mismatch')
            for field in ('prompt', 'gold', 'dataset_revision', 'source_index', 'source_split'):
                if before.get(field) != after.get(field):
                    raise ValueError('paired prompt or dataset mismatch')
            if step == 0 and before != after:
                raise ValueError('zero-checkpoint response mismatch')
            adapted = _rescore(after)
            records.append(adapted)
            rescored.append({**pair, 'baseline': cohorts['baseline'][ids.index(item)], 'adapted': adapted})
            if adapted['score']['strict']['correct'] and not adapted['score']['flexible_v2']['correct']:
                disagreements.append(dict(step=step, id=item, gold=after['gold'],
                    generated_text=after['generated_text'], score=adapted['score']))
        cohorts[str(step)] = records
    trajectory = []
    baseline = cohorts['baseline']
    baseline_mean = statistics.mean(r['token_count'] for r in baseline)
    for label, records in cohorts.items():
        entry = dict(label=label, total=150,
            mean_response_tokens=statistics.mean(r['token_count'] for r in records),
            median_response_tokens=statistics.median(r['token_count'] for r in records),
            cap_count=sum(r['stop_reason'] == 'cap' for r in records))
        entry['length_ratio'] = entry['mean_response_tokens'] / baseline_mean if baseline_mean else None
        for scorer in ('strict', 'flexible_v2', 'flexible_v3'):
            scores = [r['score'][scorer] for r in records]
            correct = sum(s['correct'] for s in scores)
            entry.update({scorer + '_correct': correct, scorer + '_accuracy': correct / 150,
                scorer + '_95': _wilson_95(correct, 150),
                scorer + '_invalid_count': sum(s['status'] == 'invalid' for s in scores),
                scorer + '_invalid_reasons': dict(Counter(s.get('invalid_reason') for s in scores
                    if s['status'] == 'invalid' and s.get('invalid_reason')))})
        for version in ('v2', 'v3'):
            entry['strict_correct_' + version + '_invalid'] = sum(
                r['score']['strict']['correct'] and r['score']['flexible_' + version]['status'] == 'invalid'
                for r in records)
        if label != 'baseline':
            differences = [int(r['score']['flexible_v3']['correct']) - int(b['score']['flexible_v3']['correct'])
                           for b, r in zip(baseline, records)]
            rng = random.Random(42)
            draws = [statistics.mean(rng.choices(differences, k=150)) for _ in range(2000)]
            entry.update(flexible_v3_change=statistics.mean(differences),
                         flexible_v3_change_95=_interval(draws))
        trajectory.append(entry)
    result = dict(name=_name(name), scorer='gsm8k-flexible-v3', post_hoc=True,
        source_sha256=hashlib.sha256(content).hexdigest(), trajectory=trajectory,
        paired_interval=dict(method='paired percentile bootstrap', seed=42, draws=2000),
        evidence_boundary='Exported responses and legacy scores verified; original run configs and hashes not independently verified. Intervals condition on one training seed and this cohort.')
    directory = Path(output_root).resolve() / 'reports' / _name(name)
    # All input validation completes before creating a report directory.
    _save_frozen(directory / 'config.json', dict(source=str(source), source_sha256=result['source_sha256'],
        scorer=result['scorer'], post_hoc=True, paired_interval=result['paired_interval'],
        code_sha256={filename: hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest()
                     for filename in ('scoring.py', 'sft_score_audit.py')}))
    _save_bytes(directory / 'inputs/paired-items.jsonl', content)
    _save_frozen(directory / 'results/results.json', result)
    for filename, rows in [('paired-items.jsonl', rescored), ('disagreements.jsonl', disagreements)]:
        _save_bytes(directory / 'results' / filename,
                    ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode())
    lines = [f'# {name}: flexible v3 scoring audit', '',
        'Post-hoc scorer correction applied uniformly to saved baseline and checkpoint responses. No inference or retraining.', '',
        '| Variant | Strict | Flexible v2 | Flexible v3 | V3 paired change (95%) | Mean tokens | Strict-correct / v2 invalid |',
        '|---|---:|---:|---:|---|---:|---:|']
    for row in trajectory:
        change = '—' if row['label'] == 'baseline' else (
            f"{row['flexible_v3_change']:+.3f} [{row['flexible_v3_change_95'][0]:+.3f}, {row['flexible_v3_change_95'][1]:+.3f}]")
        lines.append(f"| {row['label']} | {row['strict_accuracy']:.3f} | {row['flexible_v2_accuracy']:.3f} | {row['flexible_v3_accuracy']:.3f} | {change} | {row['mean_response_tokens']:.2f} | {row['strict_correct_v2_invalid']} |")
    lines += ['', 'V3 accepts the strict final-line contract first, independently of gold, then falls back to v2. Legacy scores and raw responses are preserved.', '', result['evidence_boundary'], '']
    _save_bytes(directory / 'results/report.md', '\n'.join(lines).encode())
    _save_frozen(directory / 'meta/status.json', dict(state='completed', total=750, source_sha256=result['source_sha256']))
    return result

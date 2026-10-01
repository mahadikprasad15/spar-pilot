"""CPU-only paired checkpoint analysis with immutable source provenance."""

import json
import random
import statistics
from pathlib import Path

from pilot_eval.rescoring import _save_bytes, rescore_gsm8k
from pilot_eval.run import _wilson_95, run_evaluation
from pilot_eval.sft import load_sft, read_artifact
from pilot_eval.sft_evaluation import evaluation_config, evaluation_directory, verify_zero
from pilot_eval.training import file_hash, training_directory
from pilot_eval.workflow import _save_frozen


def _interval(values):
    values = sorted(values)
    def quantile(p):
        index = p * (len(values) - 1)
        lower = int(index)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)
    return [quantile(.025), quantile(.975)]


def _paired_intervals(base, adapted, plan):
    rng = random.Random(plan['seed'])
    draws = {k: [] for k in ['strict_change', 'flexible_v2_change', 'length_change', 'length_ratio']}
    for _ in range(plan['draws']):
        ids = rng.choices(range(len(base)), k=len(base))
        for scorer in ['strict', 'flexible_v2']:
            draws[scorer + '_change'].append(statistics.mean(
                int(adapted[i]['score'][scorer]['correct']) - int(base[i]['score'][scorer]['correct']) for i in ids))
        before = statistics.mean(base[i]['token_count'] for i in ids)
        after = statistics.mean(adapted[i]['token_count'] for i in ids)
        draws['length_change'].append(after - before)
        draws['length_ratio'].append(after / before if before else None)
    return {key: _interval(values) if all(v is not None for v in values) else None
            for key, values in draws.items()}


def compare_sft(config_path, output_root):
    """Validate all six complete cells and persist paired report/plot artifacts."""
    root = Path(output_root).resolve()
    config, rows, items = load_sft(config_path, root)
    training = training_directory(root, config)
    if not (training / 'results/results.json').exists():
        raise ValueError('training is incomplete')
    verify_zero(root, config)
    plan = read_artifact(root, config['analysis_plan_path'])
    runtime = json.loads((training / 'meta/runtime.json').read_text())
    sources, cohorts = {}, {}

    class NoInference:
        def generate_batch(self, *args):
            raise ValueError('evaluation is incomplete; inference is not allowed in reporting')

    for checkpoint in ['baseline', '0', '8', '16', '32', '64']:
        _, expected = evaluation_config(config, root, checkpoint)
        directory = evaluation_directory(root, expected)
        if not (directory / 'results/results.json').exists():
            raise ValueError(f'checkpoint {checkpoint} evaluation is incomplete')
        saved = json.loads((directory / 'config.json').read_text())
        if saved != {**expected, 'runtime': runtime}:
            raise ValueError('comparison config/runtime mismatch')
        # Existing run verification checks every saved item, score and summary;
        # it must not infer anything during this CPU-only operation.
        run_evaluation(saved, items, NoInference(), root)
        rescore_gsm8k(directory / 'results/responses.jsonl', directory / 'config.json',
            directory / 'results/results.json', root, expected['run_id'] + '-flexible-v2')
        report = root / 'reports' / (expected['run_id'] + '-flexible-v2')
        response_path = report / 'results/responses.jsonl'
        records = [json.loads(line) for line in response_path.read_text().splitlines()]
        if [r['id'] for r in records] != [item['id'] for item in items]:
            raise ValueError('paired item alignment mismatch')
        cohorts[checkpoint] = records
        sources[checkpoint] = dict(run=str(directory.relative_to(root)),
            responses_sha256=file_hash(directory / 'results/responses.jsonl'),
            config_sha256=file_hash(directory / 'config.json'),
            summary_sha256=file_hash(directory / 'results/results.json'),
            rescored_sha256=file_hash(response_path))
    baseline = cohorts['baseline']
    baseline_length = statistics.mean(r['token_count'] for r in baseline)
    trajectory, paired = [], []
    for checkpoint, records in cohorts.items():
        lengths = [r['token_count'] for r in records]
        entry = dict(step=None if checkpoint == 'baseline' else int(checkpoint),
                     label=checkpoint, total=len(records), mean_response_tokens=statistics.mean(lengths),
                     median_response_tokens=statistics.median(lengths),
                     cap_count=sum(r['stop_reason'] == 'cap' for r in records))
        entry['cap_rate'] = entry['cap_count'] / len(records)
        for scorer in ['strict', 'flexible_v2']:
            correct = sum(r['score'][scorer]['correct'] for r in records)
            invalid = sum(r['score'][scorer]['status'] == 'invalid' for r in records)
            entry.update({scorer + '_correct': correct, scorer + '_accuracy': correct / len(records),
                          scorer + '_95': _wilson_95(correct, len(records)),
                          scorer + '_invalid_count': invalid, scorer + '_invalid_rate': invalid / len(records)})
        if checkpoint != 'baseline':
            intervals = _paired_intervals(baseline, records, plan['paired_interval'])
            for scorer in ['strict', 'flexible_v2']:
                entry[scorer + '_accuracy_change'] = statistics.mean(
                    int(a['score'][scorer]['correct']) - int(b['score'][scorer]['correct'])
                    for b, a in zip(baseline, records))
                entry[scorer + '_change_95'] = intervals[scorer + '_change']
            entry['mean_length_change'] = entry['mean_response_tokens'] - baseline_length
            entry['mean_length_change_95'] = intervals['length_change']
            entry['mean_length_ratio'] = entry['mean_response_tokens'] / baseline_length if baseline_length else None
            entry['mean_length_ratio_95'] = intervals['length_ratio']
            paired.extend(dict(step=int(checkpoint), id=a['id'], baseline=b, adapted=a)
                          for b, a in zip(baseline, records))
        trajectory.append(entry)
    summary = dict(protocol='pilot2-sft-v1', mode='exploratory', binary_collapse_threshold=None,
        trajectory=trajectory, sources=sources, runtime=runtime,
        analysis_plan_sha256=config['analysis_plan_sha256'], interval_plan=plan,
        training_config_sha256=file_hash(training / 'config.json'),
        training_result_sha256=file_hash(training / 'results/results.json'),
        gold_target_mean_tokens=statistics.mean(r['gold_tokens'] for r in rows),
        gold_target_median_tokens=statistics.median(r['gold_tokens'] for r in rows),
        historical_reference={'accuracy_before': .640, 'accuracy_after': .420,
                              'mean_tokens_before': 288, 'mean_tokens_after': 77},
        notes=['Scorers fixed before this intervention; old rescoring report post_hoc flags describe Pilot 1.',
               'Paired intervals condition on one training seed and the sampled evaluation cohort.',
               'Historical values are contextual; there is no binary collapse gate.',
               'Gold/generated length similarity does not establish a causal mechanism.'])
    destination = root / 'reports' / (config['run_id'] + '-trajectory')
    _save_frozen(destination / 'config.json', dict(training=config, sources=sources))
    _save_frozen(destination / 'results/results.json', summary)
    _save_bytes(destination / 'results/paired-items.jsonl', ''.join(
        json.dumps(r, sort_keys=True) + '\n' for r in paired).encode())
    lines = [f"# {config['run_id']}: exploratory SFT trajectory", '',
        '| Variant | Strict | Flexible v2 | Mean tokens | Median tokens | Length ratio |',
        '|---|---:|---:|---:|---:|---:|']
    for r in trajectory:
        ratio = r.get('mean_length_ratio')
        lines.append(f"| {r['label']} | {r['strict_accuracy']:.3f} | {r['flexible_v2_accuracy']:.3f} | "
                     f"{r['mean_response_tokens']:.2f} | {r['median_response_tokens']:.2f} | "
                     + (f'{ratio:.3f}' if ratio is not None else '—') + ' |')
    lines += ['', '## Paired changes against the fresh FP32 baseline', '',
        '| Step | Strict change (95%) | Flexible-v2 change (95%) | Mean-token change (95%) |',
        '|---|---|---|---|']
    for r in trajectory[1:]:
        def formatted(key, interval):
            low, high = r[interval]
            return f'{r[key]:+.3f} [{low:+.3f}, {high:+.3f}]'
        lines.append(f"| {r['step']} | {formatted('strict_accuracy_change', 'strict_change_95')} | "
                     f"{formatted('flexible_v2_accuracy_change', 'flexible_v2_change_95')} | "
                     f"{formatted('mean_length_change', 'mean_length_change_95')} |")
    lines += ['', 'GSM8K only; 150 identical held-out items, invalid/capped items included.', '',
        f"Gold-target mean/median tokens: {summary['gold_target_mean_tokens']:.2f} / "
        f"{summary['gold_target_median_tokens']:.2f} (gold text alone, no chat tokens).", '',
        'Historical reference: accuracy 0.640 → 0.420; mean response tokens 288 → 77.', '',
        *summary['notes'], '', 'See results.json for all intervals and provenance; paired-items.jsonl contains every paired response.']
    _save_bytes(destination / 'results/report.md', ('\n'.join(lines) + '\n').encode())
    plot = destination / 'results/trajectory.svg'
    if not plot.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        steps = [r['step'] for r in trajectory[1:]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for scorer, color in [('strict', '#9475ab'), ('flexible_v2', '#368a91')]:
            axes[0].plot(steps, [r[scorer + '_accuracy'] for r in trajectory[1:]],
                         marker='o', label=scorer, color=color)
            axes[0].errorbar(steps, [r[scorer + '_accuracy'] for r in trajectory[1:]],
                yerr=[[r[scorer + '_accuracy'] - r[scorer + '_95'][0] for r in trajectory[1:]],
                      [r[scorer + '_95'][1] - r[scorer + '_accuracy'] for r in trajectory[1:]]],
                fmt='none', ecolor=color, capsize=3, alpha=.7)
            axes[0].axhline(trajectory[0][scorer + '_accuracy'], color=color, linestyle='--', alpha=.6)
        axes[0].set(xlabel='Optimizer step', ylabel='Accuracy', ylim=(0, 1), title='GSM8K accuracy')
        axes[0].legend()
        axes[1].plot(steps, [r['mean_response_tokens'] for r in trajectory[1:]], marker='o', color='#368a91')
        axes[1].axhline(baseline_length, linestyle='--', color='#666666', label='Untuned baseline')
        axes[1].set(xlabel='Optimizer step', ylabel='Mean generated tokens', title='Response length')
        axes[1].legend()
        fig.suptitle('Exploratory rank-1 SFT — dashed lines: matched untuned baseline')
        fig.savefig(plot, metadata={'Date': None})
        plt.close(fig)
    _save_frozen(destination / 'meta/status.json', dict(state='completed', total=150,
                                                     checkpoints=[0, 8, 16, 32, 64]))
    return summary

"""One loader for E0.6a/E0.5 cohort sources: completed R0 (512) or matched-128 (ADR 0016)."""
import json
from pathlib import Path

from pilot_eval.stage0_sources import inside


def load_cohort(path, root):
    root = Path(root).resolve(); path = inside(root, path)
    if json.loads(path.read_text()).get('protocol') == 'stage0-matched-cohort-v1':
        from pilot_eval.stage0_matched import load
        return load(path, root)
    from pilot_eval.stage0_rollout_plan import load as load_rollouts
    from pilot_eval.stage0_rollout_results import verify as verify_rollouts
    verify_rollouts(path, root); rollout, source, training, _ = load_rollouts(path, root)
    read = lambda name: [json.loads(x) for x in (source / name).read_text().splitlines()]
    return dict(kind='r0', cohort='r0-512', plan_path=path, config=rollout, directory=source,
        complete_path=source / 'complete.json', training=training, draws=read('results/rollouts.jsonl'),
        datasets=json.loads((source / 'results/datasets.json').read_text()),
        special=json.loads((source / 'collection/inputs/special-tokens.json').read_text()),
        sampled_runtime=json.loads((path.parent / 'stage0.rollouts.execution.json').read_text())['runtime'])

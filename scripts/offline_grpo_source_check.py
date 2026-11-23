"""Report whether the offline GRPO source inputs exist and verify; changes no files.

Run from the repository: python3 scripts/offline_grpo_source_check.py --root <artifact_root>
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PREFIXES = ('plans/', 'runs/', 'inputs/')
FROZEN = 'plans/pilot4-grpo-fp32-v1-frozen-v2/grpo.frozen.json'
AUDIT = 'plans/stage0-geometry-v1/stage0.plan.json'
BASELINE = 'plans/pilot4-untuned-sampling-v1/baseline.config.json'
TRAINING = 'plans/pilot4-grpo-fp32-v1-arm-v1/grpo.training.json'
DATASET = 'plans/a100-stage0-v1/sampling-stage0.dataset.json'
CUT_REFERENCE = 'runs/stage0/a100-stage0-cuts-v1/principal-cuts/config.json'
STARTS = [FROZEN, AUDIT, BASELINE, TRAINING, 'runs/stage0/stage0-geometry-v1/results/source-audit.json']


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def scan(root, starts, limit=5_000_000):
    """Follow artifact paths named inside JSON files and split them by existence."""
    root = Path(root); present, missing, queue = set(), set(), list(starts)
    while queue:
        name = queue.pop()
        if name in present or name in missing:
            continue
        path = root/name
        if not path.exists():
            missing.add(name); continue
        present.add(name)
        if path.is_file() and path.suffix == '.json' and path.stat().st_size <= limit:
            try:
                value = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                continue
            queue.extend(s for s in _strings(value) if s.startswith(PREFIXES) and '\n' not in s)
    return present, sorted(missing)


def run_checks(checks):
    outcomes = {}
    for name, check in checks:
        try:
            print(f'OK    {name}: {check()}', flush=True); outcomes[name] = True
        except Exception as exc:
            print(f'FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}', flush=True); outcomes[name] = False
    return outcomes


def checks(root):
    def frozen():
        from pilot_eval.grpo_preflight import load_frozen
        value = load_frozen(root/FROZEN, root)
        return {k: value['settings'][k] for k in ['completion_limit', 'top_p', 'top_k', 'generation_groups', 'evaluation_batch_size', 'backward_groups']}

    def zero():
        from pilot_eval.grpo_preflight import load_frozen
        from pilot_eval.grpo_prepare import load_grpo_prepared
        from pilot_eval.training import file_hash
        plan, *_ = load_grpo_prepared(root/load_frozen(root/FROZEN, root)['source_path'], root)
        reference = plan['source_evidence']['checkpoints']['0']
        if file_hash(root/reference['path']/'adapter_model.safetensors') != reference['adapter_sha256']:
            raise ValueError('checkpoint-0 adapter hash differs')
        return reference['path']

    def audit():
        from pilot_eval.stage0_workflow import status
        value = status(root/AUDIT, root)['audit']
        if not value or not value['primary_sources_ready']:
            raise ValueError('audit is not ready')
        return {arm: sorted(source.get('checkpoints', {})) for arm, source in value['sources'].items()}

    def spectral_inputs():
        from pilot_eval.stage0_sources import factors
        from pilot_eval.stage0_workflow import status
        source = status(root/AUDIT, root)['audit']['sources']['sft']
        modules = factors(root/source['checkpoints']['64']['path'])
        weights = root/'inputs/models'/source['identity']['model_revision']
        if not weights.is_dir():
            raise FileNotFoundError(weights)
        return dict(modules=len(modules), weights=str(weights.relative_to(root)))

    def historical():
        from pilot_eval.stage0_rollout_sources import read_sources
        value = read_sources(root/AUDIT, root/BASELINE, root)
        return {k: value[k] for k in ['state', 'verified_historical_groups', 'seeds_recovered', 'policy_matches']} | dict(
            missing=len(value['missing']), runtime=value['legacy_runtime'])

    def cut_reference():
        from pilot_eval.offline_grpo_cuts import _reference
        _, ranks = _reference(root/CUT_REFERENCE, root)
        return dict(modules=len(ranks))

    def dataset():
        from pilot_eval.offline_grpo_dataset import _legacy
        config = json.loads((root/DATASET).read_text())
        answers, members, _ = _legacy(root/DATASET, root, config['model_identity'], config['scored_eos_token_id'])
        return dict(answers=len(answers), memberships={k: len(v) for k, v in members.items()})

    return [('frozen online plan', frozen), ('checkpoint-0 adapter', zero), ('source audit', audit),
            ('spectral inputs', spectral_inputs), ('historical 128 sources', historical),
            ('historical cut reference', cut_reference), ('historical dataset scores', dataset)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    present, missing = scan(root, STARTS)
    print(f'Referenced paths: {len(present)} present, {len(missing)} missing')
    for name in missing:
        print('MISSING', name)
    run_checks(checks(root))


if __name__ == '__main__':
    main()

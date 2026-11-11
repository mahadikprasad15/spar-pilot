"""Load and check the joint neuron/adapter audit's frozen run plan.

Stages (03b, 04, ...) call ``load(path, stage)``. It refuses unless the plan is
``approved`` with an approver and date, every section is present, every field listed in
``required_before[stage]`` is non-null, and the user's fixed decisions are unchanged
(zero floor unresolved, logit gate limited to logits, legacy TF32 setters forbidden).
``input_problems`` checks the sealed inventory and sequences and the background review.
"""
import hashlib
import json
from pathlib import Path

import yaml

SCHEMA = 'joint-neuron-adapter-audit/frozen-plan-v1'
SECTIONS = ['inputs', 'arms', 'partial_study', 'runtime', 'precision', 'hard_gates', 'review_flags', 'fp64_floor',
            'statistics', 'calibration_allocation', 'benchmark', 'operations', 'storage', 'preflight',
            'spec_amendments', 'required_before']
FIXED = {
    'fp64_floor.zero_floor': 'unresolved',
    'precision.legacy_allow_tf32_setters': 'forbidden',
    'hard_gates.batched_vs_singleton_logits.bound': 1.0e-3,
    'hard_gates.batched_vs_singleton_logits.applies_to': 'logits only; never a tolerance for neuron statistics',
    'hard_gates.disabled_vs_fresh_base': ('exact logits equality (torch.equal), same batch, dtype and attention; '
                                          'never add a tolerance after a failure'),
    'review_flags.signoff_record_required_before': '04',
    'precision.autocast': False,
    'precision.torch_compile': False,
    'runtime.attention': 'eager',
}


class PlanError(ValueError):
    pass


def read(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise PlanError('plan is not a mapping')
    return value


def _get(plan, dotted):
    node = plan
    for key in dotted.split('.'):
        if not isinstance(node, dict) or key not in node:
            return KeyError
        node = node[key]
    return node


def problems(plan, stage):
    found = []
    if plan.get('schema') != SCHEMA:
        found.append(f'schema is not {SCHEMA}')
    if plan.get('status') != 'approved':
        found.append(f"status is {plan.get('status')}, not approved")
    for section in SECTIONS:
        if plan.get(section) in (None, {}, []):
            found.append(f'section {section} is missing or empty')
    for dotted, expected in FIXED.items():
        value = _get(plan, dotted)
        if value is KeyError or value != expected:
            found.append(f'{dotted} must stay {expected!r} (user decision)')
    amendments = {a.get('id'): a for a in plan.get('spec_amendments') or [] if isinstance(a, dict)}
    if 'reuse-adr-0014-logit-gate' not in amendments or amendments['reuse-adr-0014-logit-gate'].get('decided_by') != 'user':
        found.append('the ADR 0014 logit-gate reuse must be documented as a user spec amendment')
    required = list((plan.get('required_before') or {}).get(stage) or [])
    if stage == '04':
        required = list((plan.get('required_before') or {}).get('03b') or []) + required
    if not required:
        found.append(f'no required_before list for stage {stage}')
    for dotted in required:
        value = _get(plan, dotted)
        if value is KeyError:
            found.append(f'{dotted} is missing')
        elif value is None:
            found.append(f'{dotted} is null')
    return found


def load(path, stage):
    plan = read(path)
    found = problems(plan, stage)
    if found:
        raise PlanError(f'plan not runnable for {stage}: ' + '; '.join(found))
    plan['_plan_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return plan


def _sealed(directory, kind):
    seal = directory / 'complete.json'
    if not seal.is_file():
        return f'{kind} run {directory} is not sealed'
    record = json.loads(seal.read_text())
    files = record.get('files') or {'inventory.json': record.get('inventory_sha256')}
    for name, digest in files.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return f'{kind} run {directory}: {name} failed its seal'
    return None


def input_problems(plan, artifact_root):
    root = Path(artifact_root)
    found = []
    inventory = root / plan['inputs']['inventory_run']
    sequences = root / plan['inputs']['sequences_run']
    for directory, kind in [(inventory, 'inventory'), (sequences, 'sequences')]:
        problem = _sealed(directory, kind)
        if problem:
            found.append(problem)
    if not found:
        review = sequences / 'review' / 'review.json'
        decision = json.loads(review.read_text())['decision'] if review.is_file() else 'pending'
        if review.is_file():
            sample = sequences / 'review' / 'background_sample.jsonl'
            if json.loads(review.read_text()).get('sample_sha256') != hashlib.sha256(sample.read_bytes()).hexdigest():
                decision = 'invalid-sample-changed'
        if decision != plan['inputs']['background_review_required']:
            found.append(f"background review is {decision}, plan requires {plan['inputs']['background_review_required']}")
    return found


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='List what still blocks a stage of the audit plan.')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', default='03b')
    parser.add_argument('--artifact_root')
    args = parser.parse_args()
    plan = read(args.plan)
    out = dict(stage=args.stage, plan_problems=problems(plan, args.stage))
    if args.artifact_root and plan.get('inputs'):
        out['input_problems'] = input_problems(plan, args.artifact_root)
    print(json.dumps(out, indent=2))

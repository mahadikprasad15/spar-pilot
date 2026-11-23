"""Sparse gold-SFT rewiring: typed launch record and frozen per-arm plan (ticket 05).

Two arms, ``sparse-gold-sft-r1`` and ``sparse-gold-sft-r0``, share one launch record and differ only in rank and
read LR (D13, D1). Every number the run uses is an explicit leaf of the record; nothing is chosen at run time.

Reused from the C-attribution runner (``c_attribution_chain``): ``typecheck`` (unknown/missing keys, bools,
'[proposed]' and null leaves, and text where a number belongs: PyYAML reads ``19.5e12`` and ``6e-4`` as text),
``record_sha256`` / ``run_sha256``, ``seal_launch_record``, ``load_launch_record`` and ``finalize_draft`` (both
take this module's ``validate``), and ``stop_commands``. New here: the schema's fields, the cross-field rules,
the D24 storage arithmetic and the plan (the full 1,024-step order of D7, checkpoints, cadences, per-step LR
multiplier and mask phase), which training, checkpoint seals and evaluation bind to by hash.

Decisions not yet signed off stay ``'[proposed] ...'`` in the draft's ``decisions`` map, so ``typecheck``
refuses to seal the record until the researcher signs each one.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

from pilot_eval import c_attribution_chain as cc
from pilot_eval.attribution_chain import LaunchRefused
from pilot_eval.workflow import _hash

LAUNCH_SCHEMA = 'sparse-rewiring/launch-record-v1'
PLAN_SCHEMA = 'sparse-rewiring/plan-v1'
ARMS = {'sparse-gold-sft-r1': 1, 'sparse-gold-sft-r0': 0}   # arm -> rank (D13; names proposed)

# Qwen2.5-1.5B-Instruct MLP shapes (pinned model).
LAYERS, HIDDEN, INTERMEDIATE = 28, 1536, 8960
FP32_BYTES, FP16_BYTES = 4, 2

NUM, INT, TEXT = cc.NUM, cc.INT, cc.TEXT
SCHEMA = {
    'schema': TEXT, 'mode': TEXT, 'label': TEXT, 'code_commit': TEXT,
    'pod_id': TEXT, 'pod_stop_commands': [TEXT], 'volume_identity': TEXT, 'export_repo': TEXT,
    'hourly_cost_usd': cc.COST,
    'timeouts_seconds': {'copy': NUM, 'export_attempt': NUM, 'stop_attempt': NUM, 'writer_join': NUM},
    'marker_stale_seconds': NUM,
    'disk': {'min_free_gb': NUM, 'container_total_gb': NUM},
    'hardware': {'gpu': TEXT, 'count': INT},
    'arms': {'*': {'run_id': TEXT, 'rank': INT, 'read_lr': NUM}},
    'model': {'layers': INT, 'hidden': INT, 'intermediate': INT, 'parts': [TEXT], 'positions': TEXT},
    'data': {'training_examples': INT, 'examples_per_step': INT, 'microbatch': INT, 'examples_per_mask': INT,
             'gold_order_steps': INT, 'later_epochs': TEXT, 'loss': TEXT, 'targets': TEXT},
    'training': {'steps': INT, 'warmup_steps': INT, 'schedule': TEXT, 'freeze_after_step': INT,
                 'freeze_rule': TEXT},
    'objective': {'l0_lambda': NUM, 'l0_term': TEXT},
    'optimizer': {'name': TEXT, 'implementation': TEXT,
                  'reads': {'betas': [NUM], 'eps': NUM, 'weight_decay': NUM, 'clip_norm': NUM},   # lr per arm
                  'masks': {'lr': NUM, 'betas': [NUM], 'eps': NUM, 'weight_decay': NUM, 'clip': TEXT}},
    'hard_concrete': {'beta': NUM, 'gamma': NUM, 'zeta': NUM, 'log_alpha_init': NUM, 'read_init': TEXT},
    'seeds': {'data_order_base': INT, 'read_init_base': INT, 'mask_noise': INT},
    'runtime': {'dtype': TEXT, 'precision': TEXT, 'attention': TEXT, 'kv_cache': TEXT,
                'gradient_checkpointing': TEXT, 'determinism': TEXT, 'cublas_workspace': TEXT},
    'checkpoints': [INT],
    'cadences': {'step_metrics_every': INT, 'validation_ce_every': INT, 'gate_snapshot_every': INT,
                 'settle': {'first': INT, 'every': INT, 'jaccard_min': NUM, 'expected_change_max': NUM}},
    'storage': {'checkpoint_contents': [TEXT], 'optimizer_state': TEXT, 'interruption': TEXT,
                'gate_snapshot_dtype': TEXT},
    'evaluation': {'validation': {'questions': INT, 'ce_masks': TEXT, 'greedy_at': TEXT},
                   'cohort150': {'steps': [INT], 'role': TEXT},
                   'cap': INT, 'eos_token_id': INT, 'repetition_penalty': NUM, 'scorer': TEXT, 'batch_size': INT},
    'decisions': {'*': TEXT},
    'inputs': {'artifact_root': TEXT, 'bundle': TEXT, 'model_dir': TEXT, 'work_dir': TEXT,
               'gold_plan': TEXT, 'gold_order': TEXT},
}
USER_FILLED = ('code_commit', 'pod_id', 'inputs.artifact_root', 'inputs.bundle', 'inputs.model_dir',
               'inputs.work_dir')
# The decisions every record states (decision-log.md). D11 is the validation set; D16/D17 are code-base choices.
REQUIRED_DECISIONS = ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14',
                      'D15', 'D18', 'D20', 'D24', 'D25')

# Values the researcher decided (2026-11-23) or the spec fixes. A production record must carry exactly these;
# a rehearsal (GPU qualification, ticket 15) may shorten the run but keeps every rule below.
PRODUCTION = {
    'hardware': {'gpu': 'A100-80GB', 'count': 1},                                     # D18
    'arms': {'sparse-gold-sft-r1': {'rank': 1, 'read_lr': 3e-3},                      # D1, D13
             'sparse-gold-sft-r0': {'rank': 0, 'read_lr': 3e-4}},
    'data.examples_per_step': 8, 'data.microbatch': 1, 'data.examples_per_mask': 4,  # gold, D14
    'data.training_examples': 512, 'data.gold_order_steps': 64,                       # gold, D7
    'training.steps': 1024, 'training.warmup_steps': 20, 'training.freeze_after_step': 960,   # D5
    'objective.l0_lambda': 6e-4,                                                      # D6
    'checkpoints': [0, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1024],                # D5
    'cadences.step_metrics_every': 1, 'cadences.validation_ce_every': 16,             # D20
    'cadences.gate_snapshot_every': 16,
    'evaluation.cohort150.steps': [64, 1024], 'evaluation.cohort150.role': 'evaluation-only',   # D12
    'evaluation.cap': 1024, 'evaluation.eos_token_id': 151645,                        # protocol
}
# Fixed in every mode: what the method and the comparability contract are, not how long the run is.
RULES = {
    'model.layers': LAYERS, 'model.hidden': HIDDEN, 'model.intermediate': INTERMEDIATE,
    'model.parts': ['gate', 'up'], 'model.positions': 'all-unpadded',
    'data.microbatch': 1, 'data.later_epochs': 'reshuffle-per-epoch',
    'data.loss': 'gold-completion-loss-step-token-denominator', 'data.targets': 'gold-frozen-training-items',
    'training.schedule': 'linear-warmup-cosine-to-zero', 'training.freeze_rule': 'z_det>0',
    'objective.l0_term': 'once-per-step-expected-active',
    'optimizer.name': 'adamw', 'optimizer.implementation': 'foreach-off',
    'optimizer.masks.clip': 'none',
    'hard_concrete.read_init': 'a~N(0,1)/sqrt(d);b=0',
    'runtime.dtype': 'float32', 'runtime.precision': 'ieee', 'runtime.attention': 'eager',
    'runtime.kv_cache': 'off-in-training', 'runtime.gradient_checkpointing': 'non-reentrant',
    'runtime.determinism': 'full', 'runtime.cublas_workspace': ':4096:8',
    'storage.checkpoint_contents': ['read_factors', 'log_alpha', 'seal'],             # D24
    'storage.optimizer_state': 'none', 'storage.interruption': 'restart-from-step-0',
    'storage.gate_snapshot_dtype': 'float16',
    'evaluation.validation.ce_masks': 'deterministic', 'evaluation.validation.greedy_at': 'checkpoints',
    'evaluation.repetition_penalty': 1.0, 'evaluation.scorer': 'flexible-v3',
}


def _get(record, dotted):
    value = record
    for key in dotted.split('.'):
        value = value[key]
    return value


def _same(a, b):
    """Equality that keeps ints and floats apart only where it matters: 6e-4 == 6.0e-4, but 1 != True."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is not bool and type(b) is not bool and a == b


# --- storage arithmetic (D24) -----------------------------------------------------------------------

def read_parameters(rank, *, layers=LAYERS, hidden=HIDDEN, intermediate=INTERMEDIATE):
    """Trainable read-factor elements: rank 1 ``a[d] + b[n]``, rank k ``A[k,d] + B[n,k]``, rank 0 ``W[n,d]``,
    per layer and per part (gate, up)."""
    per_part = intermediate * hidden if rank == 0 else rank * (hidden + intermediate)
    return layers * 2 * per_part


def checkpoint_bytes(rank, **shape):
    """One D24 checkpoint: FP32 read factors and FP32 ``logα`` (one gate per neuron); the seal is negligible."""
    layers = shape.get('layers', LAYERS)
    intermediate = shape.get('intermediate', INTERMEDIATE)
    return (read_parameters(rank, **shape) + layers * intermediate) * FP32_BYTES


def planned_disk_bytes(record):
    """Bytes the run writes for its checkpoints and gate snapshots, summed over both arms."""
    model = record['model']
    shape = dict(layers=model['layers'], hidden=model['hidden'], intermediate=model['intermediate'])
    snapshots = len(gate_snapshot_steps(record)) * model['layers'] * model['intermediate'] * FP16_BYTES
    return sum(len(record['checkpoints']) * checkpoint_bytes(arm['rank'], **shape) + snapshots
               for arm in record['arms'].values())


# --- derived step lists -----------------------------------------------------------------------------

def _every(record, n, *, start=None, stop=None):
    steps = record['training']['steps']
    start = n if start is None else start
    stop = steps if stop is None else stop
    return list(range(start, stop + 1, n))


def validation_steps(record):
    """Validation CE with deterministic masks every ``validation_ce_every`` steps (D20: 16..1024, 64 checks).
    Step 0 is the untuned reference, measured once from checkpoint 0."""
    return _every(record, record['cadences']['validation_ce_every'])


def gate_snapshot_steps(record):
    return _every(record, record['cadences']['gate_snapshot_every'])


def settle_steps(record):
    """D5 settle diagnostics: every 64 steps from 256 up to the freeze (logged only)."""
    settle = record['cadences']['settle']
    return _every(record, settle['every'], start=settle['first'], stop=record['training']['freeze_after_step'])


def lr_multiplier(step, *, warmup, total):
    """Shared ``LambdaLR`` multiplier of both groups for optimizer step ``step`` (1-based): linear warmup to 1
    at ``warmup``, then cosine to exactly 0 at ``total``. ``LambdaLR`` calls its lambda with the 0-based count,
    so the training loop passes ``lambda i: lr_multiplier(i + 1, ...)``."""
    if not 1 <= step <= total:
        raise ValueError(f'step {step} outside 1..{total}')
    if step <= warmup:
        return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


# --- validation --------------------------------------------------------------------------------------

def validate(record, *, code_commit):
    problems = cc.typecheck(record, SCHEMA)
    if problems:
        raise LaunchRefused(f'launch record problems: {problems[:10]}')
    if record['schema'] != LAUNCH_SCHEMA:
        raise LaunchRefused('not a sparse-rewiring launch record')
    if record['code_commit'] != code_commit:
        raise LaunchRefused(f"code commit {code_commit} differs from the record's {record['code_commit']}")
    mode = record['mode']
    if mode not in ('rehearsal', 'production'):
        raise LaunchRefused('mode must be production or rehearsal')
    if mode == 'rehearsal' and not record['label'].startswith('rehearsal'):
        raise LaunchRefused('a rehearsal needs a label starting with "rehearsal"')
    if mode == 'production' and record['label'] != 'production':
        raise LaunchRefused('a production record is labelled "production"')
    for dotted, expected in RULES.items():
        if not _same(_get(record, dotted), expected):
            raise LaunchRefused(f'{dotted} must be {expected!r} (spec; fixed in every mode)')
    if mode == 'production':
        for dotted, expected in PRODUCTION.items():
            actual = _get(record, dotted)
            if dotted == 'arms':
                actual = {k: {'rank': v['rank'], 'read_lr': v['read_lr']} for k, v in actual.items()}
            if not _same(actual, expected):
                raise LaunchRefused(f'production {dotted} must be {expected!r} (decided 2026-11-23), '
                                    f'got {actual!r}')
    missing = [d for d in REQUIRED_DECISIONS if d not in record['decisions']]
    if missing:
        raise LaunchRefused(f'decisions missing: {missing}')
    _check_arms(record)
    _check_shape(record)
    planned = planned_disk_bytes(record) / 1e9
    if record['disk']['container_total_gb'] < planned + record['disk']['min_free_gb']:
        raise LaunchRefused(f'disk.container_total_gb {record["disk"]["container_total_gb"]} is below the D24 '
                            f'planned {planned:.1f} GB plus min_free_gb {record["disk"]["min_free_gb"]}')
    cc.stop_commands(record)
    return record


def _check_arms(record):
    arms = record['arms']
    if sorted(arms) != sorted(ARMS):
        raise LaunchRefused(f'arms must be {sorted(ARMS)}')
    for name, arm in arms.items():
        if arm['rank'] != ARMS[name]:
            raise LaunchRefused(f'{name} is rank {ARMS[name]}, not {arm["rank"]}')
        if not arm['read_lr'] > 0:
            raise LaunchRefused(f'{name}.read_lr must be positive')
        if not str(arm['run_id']).replace('-', '').isalnum():
            raise LaunchRefused(f"run_id {arm['run_id']!r} must be a simple name")
    if len({arm['run_id'] for arm in arms.values()}) != len(arms):
        raise LaunchRefused('each arm needs its own run_id')


def _check_shape(record):
    data, training, cad = record['data'], record['training'], record['cadences']
    steps, warmup, freeze = training['steps'], training['warmup_steps'], training['freeze_after_step']
    if not 0 < warmup < freeze < steps:
        raise LaunchRefused('need 0 < warmup_steps < freeze_after_step < steps')
    if data['examples_per_step'] % data['examples_per_mask']:
        raise LaunchRefused('examples_per_step must be a multiple of examples_per_mask (D14 groups)')
    if data['training_examples'] % data['examples_per_step']:
        raise LaunchRefused('an epoch must be a whole number of steps')
    epoch_steps = data['training_examples'] // data['examples_per_step']
    if data['gold_order_steps'] != min(epoch_steps, steps):
        raise LaunchRefused('gold_order_steps must be one epoch (gold saved exactly one epoch of order)')
    points = record['checkpoints']
    if points != sorted(set(points)) or points[0] != 0 or points[-1] != steps or freeze not in points:
        raise LaunchRefused('checkpoints must be increasing, start at 0, end at steps and include the freeze')
    if record['mode'] == 'production' and 64 not in points:
        raise LaunchRefused("checkpoints must include gold's step 64")
    for step in record['evaluation']['cohort150']['steps']:
        if step not in points:
            raise LaunchRefused(f'cohort150 step {step} is not a checkpoint')
    for key in ('step_metrics_every', 'validation_ce_every', 'gate_snapshot_every'):
        if not 0 < cad[key] <= steps:
            raise LaunchRefused(f'cadences.{key} must be in 1..steps')
    settle = cad['settle']
    if not (0 < settle['every'] and 0 < settle['first'] <= freeze and 0 < settle['jaccard_min'] <= 1
            and settle['expected_change_max'] > 0):
        raise LaunchRefused('cadences.settle is inconsistent with the freeze or out of range')
    groups = record['optimizer']
    for part in ('reads', 'masks'):
        if len(groups[part]['betas']) != 2 or not all(0 <= b < 1 for b in groups[part]['betas']):
            raise LaunchRefused(f'optimizer.{part}.betas must be two numbers in [0, 1)')
    if groups['reads']['clip_norm'] <= 0:
        raise LaunchRefused('optimizer.reads.clip_norm must be a positive number (D4)')
    hc = record['hard_concrete']
    if not (hc['gamma'] < 0 < 1 < hc['zeta'] and hc['beta'] > 0):
        raise LaunchRefused('hard concrete needs gamma < 0, zeta > 1, beta > 0')


def load_launch_record(path, *, code_commit):
    return cc.load_launch_record(path, code_commit=code_commit, validator=validate)


def finalize_draft(draft_path, out_path, *, code_commit, pod_id=None, fill=None):
    return cc.finalize_draft(draft_path, out_path, code_commit=code_commit, pod_id=pod_id, fill=fill,
                             validator=validate, fillable=USER_FILLED)


def draft_problems(draft_path):
    """Every unresolved or mistyped leaf of a YAML draft, as ``typecheck`` reports it (fill-ins excluded)."""
    import yaml
    record = yaml.safe_load(Path(draft_path).read_text())
    filled = set(USER_FILLED)
    return [p for p in cc.typecheck(record, SCHEMA)
            if not (p.endswith('(unresolved: null)') and p.split(' ')[0] in filled)]


# --- plan --------------------------------------------------------------------------------------------

def full_order(training_ids, gold_order, *, steps, examples_per_step, base_seed):
    """The step -> example-ID list (D7). Epoch 1 is gold's saved order exactly; epoch e >= 2 is the training IDs,
    in ``training.items.json`` order, shuffled with ``random.Random(base_seed + e - 1)`` (gold's own epoch 1 is
    the same rule with e = 1, applied by ``sft_backend`` to row indices). Truncated to ``steps``."""
    training_ids = list(training_ids)
    if len(set(training_ids)) != len(training_ids):
        raise ValueError('training IDs are not unique')
    if sorted(gold_order) != sorted(training_ids):
        raise ValueError("gold's saved order is not a permutation of the training IDs")
    if len(training_ids) % examples_per_step:
        raise ValueError('an epoch must be a whole number of steps')
    epoch_steps = len(training_ids) // examples_per_step
    epochs = -(-steps // epoch_steps)
    flat = list(gold_order)
    for epoch in range(2, epochs + 1):
        ids = list(training_ids)
        random.Random(base_seed + epoch - 1).shuffle(ids)
        flat += ids
    return [flat[i * examples_per_step:(i + 1) * examples_per_step] for i in range(steps)]


def gold_reconstruction(training_ids, gold_order, base_seed):
    """'matches' if gold's saved order equals the code's seed-``base_seed`` shuffle of row indices, else
    'differs'. Informational: the saved file governs (design fog; ticket 01 reads the real file)."""
    order = list(range(len(training_ids)))
    random.Random(base_seed).shuffle(order)
    return 'matches' if [training_ids[i] for i in order] == list(gold_order) else 'differs'


def _groups(n, size):
    return [[start, min(start + size, n)] for start in range(0, n, size)]


def build_plan(record, arm, training_ids, gold_order, *, inputs_sha256):
    """The frozen plan for one arm: settings from the record, the full step order with D14 mask groups, the
    per-step LR multiplier and mask phase, checkpoints and cadences. ``inputs_sha256`` names the hashed gold
    files the IDs came from. The plan hash binds checkpoint seals, evaluation and analysis."""
    if arm not in record['arms']:
        raise ValueError(f'unknown arm {arm}')
    data, training = record['data'], record['training']
    if len(training_ids) != data['training_examples']:
        raise ValueError(f'{len(training_ids)} training IDs; the record says {data["training_examples"]}')
    total, warmup, freeze = training['steps'], training['warmup_steps'], training['freeze_after_step']
    order = full_order(training_ids, gold_order, steps=total, examples_per_step=data['examples_per_step'],
                       base_seed=record['seeds']['data_order_base'])
    epoch_steps = data['training_examples'] // data['examples_per_step']
    steps = [dict(step=s, epoch=(s - 1) // epoch_steps + 1, ids=ids,
                  mask_groups=_groups(len(ids), data['examples_per_mask']),
                  lr_multiplier=lr_multiplier(s, warmup=warmup, total=total),
                  masks='trainable' if s <= freeze else 'frozen-binary')
             for s, ids in enumerate(order, 1)]
    arm_record = record['arms'][arm]
    plan = dict(
        schema=PLAN_SCHEMA, arm=arm, run_id=arm_record['run_id'], rank=arm_record['rank'],
        launch_run_sha256=cc.run_sha256(record),
        settings={k: record[k] for k in ('model', 'data', 'training', 'objective', 'hard_concrete', 'seeds',
                                         'runtime', 'storage', 'evaluation', 'decisions')},
        optimizer=dict(record['optimizer'], reads=dict(record['optimizer']['reads'], lr=arm_record['read_lr'])),
        read_init_seeds=[record['seeds']['read_init_base'] + layer for layer in range(record['model']['layers'])],
        trainable_parameters=dict(reads=read_parameters(arm_record['rank'], layers=record['model']['layers'],
                                                        hidden=record['model']['hidden'],
                                                        intermediate=record['model']['intermediate']),
                                  log_alpha=record['model']['layers'] * record['model']['intermediate']),
        checkpoints=list(record['checkpoints']),
        checkpoint_bytes=checkpoint_bytes(arm_record['rank'], layers=record['model']['layers'],
                                          hidden=record['model']['hidden'],
                                          intermediate=record['model']['intermediate']),
        validation_steps=validation_steps(record), gate_snapshot_steps=gate_snapshot_steps(record),
        settle_steps=settle_steps(record), cohort150_steps=list(record['evaluation']['cohort150']['steps']),
        inputs=dict(inputs_sha256, training_ids=list(training_ids), gold_order=list(gold_order),
                    gold_order_reconstruction=gold_reconstruction(list(training_ids), gold_order,
                                                                  record['seeds']['data_order_base'])),
        steps=steps)
    plan['plan_sha256'] = plan_sha256(plan)
    return plan


def plan_sha256(plan):
    return _hash({k: v for k, v in plan.items() if k != 'plan_sha256'})


def plan_from_files(record, arm, training_items_path, gold_order_path):
    """Plan from gold's frozen ``training.items.json`` and saved ``inputs/order.json``, hashed as ``sft`` hashes
    them (``workflow._hash`` of the parsed JSON)."""
    rows = json.loads(Path(training_items_path).read_text())
    gold_order = json.loads(Path(gold_order_path).read_text())
    return build_plan(record, arm, [r['id'] for r in rows], gold_order,
                      inputs_sha256=dict(training_items_sha256=_hash(rows), gold_order_sha256=_hash(gold_order)))


def write_plan(plan, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
    return path


def load_plan(path, record, arm):
    """A saved plan, refused unless its hash verifies and it is exactly what this record builds for ``arm``
    from the plan's own recorded inputs."""
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != PLAN_SCHEMA or plan.get('plan_sha256') != plan_sha256(plan):
        raise LaunchRefused(f'{path}: plan does not match its hash')
    if plan['arm'] != arm or plan['launch_run_sha256'] != cc.run_sha256(record):
        raise LaunchRefused(f'{path}: plan was built for another arm or launch record')
    rebuilt = _rebuild(plan, record)
    if rebuilt != plan['plan_sha256']:
        raise LaunchRefused(f'{path}: plan differs from what the launch record builds')
    return plan


def _rebuild(plan, record):
    inputs = {k: v for k, v in plan['inputs'].items()
              if k not in ('training_ids', 'gold_order', 'gold_order_reconstruction')}
    return build_plan(record, plan['arm'], plan['inputs']['training_ids'], plan['inputs']['gold_order'],
                      inputs_sha256=inputs)['plan_sha256']


def step_ids(plan, step):
    """The example IDs the loop consumes at ``step`` (1-based), straight from the frozen plan."""
    entry = plan['steps'][step - 1]
    if entry['step'] != step:
        raise ValueError(f'plan step {step} is out of place')
    return entry['ids']


# --- command line ------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pilot_eval.sparse_rewiring_plan')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='list unresolved or mistyped leaves of a YAML draft')
    check.add_argument('draft')
    seal = sub.add_parser('seal', help='approved YAML draft -> sealed, typed JSON record')
    seal.add_argument('--draft', required=True)
    seal.add_argument('--out', required=True)
    seal.add_argument('--code-commit', required=True)
    seal.add_argument('--pod-id')
    for name in ('artifact-root', 'bundle', 'model-dir', 'work-dir'):
        seal.add_argument(f'--{name}')
    plan = sub.add_parser('plan', help='write the frozen plan of one arm')
    plan.add_argument('--record', required=True)
    plan.add_argument('--code-commit', required=True)
    plan.add_argument('--arm', required=True, choices=sorted(ARMS))
    plan.add_argument('--training-items', required=True)
    plan.add_argument('--gold-order', required=True)
    plan.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'check':
        problems = draft_problems(args.draft)
        print('\n'.join(problems) if problems else 'no problems')
        return 1 if problems else 0
    if args.command == 'seal':
        fill = {f'inputs.{k}': getattr(args, k) for k in ('artifact_root', 'bundle', 'model_dir', 'work_dir')}
        print(finalize_draft(args.draft, args.out, code_commit=args.code_commit, pod_id=args.pod_id, fill=fill))
        return 0
    record = load_launch_record(args.record, code_commit=args.code_commit)
    print(write_plan(plan_from_files(record, args.arm, args.training_items, args.gold_order), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())

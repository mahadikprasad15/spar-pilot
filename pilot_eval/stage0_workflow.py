"""CPU Stage 0 preparation, resumable source auditing and global budget ledger."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_sources import audit_source, initialization, inside
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen

CEILING = 20


def _now():
    return datetime.now(timezone.utc).isoformat()


def _implementation():
    directory = Path(__file__).parent
    return {name: file_hash(directory/name) for name in
            ['stage0_workflow.py', 'stage0_sources.py', 'trace_readout_assets.py']}


def _software():
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(git_commit=commit, python=sys.version, platform=platform.platform(),
                scope='CPU files/tensor audit; no model-runtime import')


@contextmanager
def _writer(directory):
    """Exclusive product ownership; stale cross-session ownership is not auto-cleared."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'writer.json'
    with run_lock(directory):
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError('Stage 0 writer ownership exists; inspect before recovery.') from exc
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(dict(token=uuid.uuid4().hex, pid=os.getpid(), host=platform.node(), created_at=_now()), stream)
                stream.flush()
                os.fsync(stream.fileno())
            yield
        finally:
            marker.unlink()


def _state(directory, state, completed, total, **extra):
    value = dict(state=state, completed=completed, total=total, updated_at=_now(), **extra)
    _write_json(directory/'meta/status.json', value)
    _write_json(directory/'checkpoints/progress.json', value)
    path = directory/'logs/run.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    return value


def prepare(selection_path, output_root, name):
    root = Path(output_root).resolve()
    safe_name(name)
    if name == 'budget':
        raise ValueError('The name budget is reserved for the cumulative ledger.')
    selection = json.loads(Path(selection_path).read_text())
    sources = selection.get('sources', {})
    products = selection.get('products', {})
    if (not {'sft', 'grpo'} <= set(sources) or set(sources)-{'sft', 'grpo', 'random'}
            or not isinstance(products, dict)):
        raise ValueError('Select explicit sft/grpo prepared sources and optional named products.')
    normalized = {}
    parents = {}
    for label, value in sources.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Each selected source must have exactly one path.')
        path = inside(root, value)
        normalized[label] = str(path.relative_to(root))
        if path.is_file():
            parents[normalized[label]] = file_hash(path)
    for value in products.values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Each optional product must have one completion-receipt path.')
        inside(root, value)
    config = dict(schema_version=1, protocol='stage0-source-audit-v1', name=name,
                  run_path=f'runs/stage0/{name}', sources=normalized,
                  products={k: str(inside(root, v).relative_to(root)) for k, v in products.items()},
                  parents=parents, implementation=_implementation(), budget_ceiling_units=CEILING,
                  selection_sha256=_hash(selection), execution_kind='cpu-source-audit-only')
    path = root/'plans'/name/'stage0.plan.json'
    with _writer(path.parent):
        _save_frozen(path, config)
        directory = root/config['run_path']
        _save_frozen(directory/'meta/run_manifest.json', dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config)))
        if not (directory/'meta/software.json').exists():
            _save_frozen(directory/'meta/software.json', _software())
        if not (directory/'meta/status.json').exists():
            _state(directory, 'prepared', 0, len(normalized))
    return path


def _load(path, root):
    path = inside(root, path)
    config = json.loads(path.read_text())
    if (config.get('protocol') != 'stage0-source-audit-v1'
            or config.get('budget_ceiling_units') != CEILING
            or config.get('implementation') != _implementation()):
        raise ValueError('Stage 0 plan or implementation changed; preserve it and use a new name.')
    directory = inside(root, config['run_path'])
    expected = dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config))
    if json.loads((directory/'meta/run_manifest.json').read_text()) != expected:
        raise ValueError('Stage 0 frozen plan manifest mismatch.')
    _verify_parents(root, config['parents'])
    return config, directory


def _verify_parents(root, parents):
    for value, digest in parents.items():
        path = inside(root, value)
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError('Stage 0 source payload hash mismatch.')


def _labels(config):
    return [label for label in ['sft', 'grpo', 'random'] if label in config['sources']]


def _unit(directory, root, label, config):
    folder = directory/'units'/label
    if not (folder/'complete.json').exists():
        return None
    seal = json.loads((folder/'complete.json').read_text())
    path = folder/'result.json'
    if seal != dict(files={'result.json': file_hash(path)},
                    identity=dict(config_sha256=_hash(config), source=label)):
        raise ValueError('Stage 0 audit unit seal mismatch.')
    result = json.loads(path.read_text())
    _verify_parents(root, result['parents'])
    return result


def audit(path, output_root, *, max_new_units=None):
    if max_new_units is not None and max_new_units < 0:
        raise ValueError('max-new-units must be nonnegative.')
    root = Path(output_root).resolve()
    config, directory = _load(path, root)
    with _writer(directory):
        if (directory/'complete.json').exists():
            return status(path, root)['audit']
        completed, new, sources = 0, 0, {}
        try:
            for label in _labels(config):
                value = config['sources'][label]
                result = _unit(directory, root, label, config)
                if result is None:
                    if max_new_units is not None and new >= max_new_units:
                        _state(directory, 'incomplete', completed, len(config['sources']))
                        return dict(state='incomplete', completed=completed)
                    result = audit_source(root, label, value)
                    folder = directory/'units'/label
                    _write_json(folder/'result.json', result)
                    _write_json(folder/'complete.json', dict(files={'result.json': file_hash(folder/'result.json')},
                                identity=dict(config_sha256=_hash(config), source=label)))
                    new += 1
                sources[label] = result
                completed += 1
                _state(directory, 'running', completed, len(config['sources']))
                print(f'Stage 0 source audit: {completed}/{len(config["sources"])} sealed; {label}.', flush=True)
            identities = [row['identity'] for row in sources.values() if row['state']=='audited']
            if identities and any(row != identities[0] for row in identities):
                raise ValueError('Stage 0 source model/tokenizer/base identities differ.')
            products = {}
            product_parents = {}
            for label, value in config['products'].items():
                marker = inside(root, value)
                if not marker.is_file():
                    products[label] = dict(state='missing', path=value)
                    continue
                saved = json.loads(marker.read_text())
                files = saved.get('files', {})
                if not files:
                    raise ValueError('Optional product receipt has no payload hashes.')
                product_parents[value] = file_hash(marker)
                for relative, expected in files.items():
                    payload = inside(root, marker.parent/relative)
                    if not payload.is_file() or file_hash(payload) != expected:
                        raise ValueError('Optional product payload hash mismatch.')
                    product_parents[str(payload.relative_to(root))] = expected
                products[label] = dict(state='verified-files-only', path=value,
                    scope='payload integrity; scientific/schema compatibility not established')
            ready = all(sources[label].get('checkpoints', {}).get('64', {}).get('state') == 'verified-adapter-payload'
                        for label in ['sft', 'grpo'])
            result = dict(schema_version=1, state='audited', primary_sources_ready=ready,
                sources=sources, products=products, initialization=initialization(root, sources),
                gpu_qualified=False, measurement_complete=False, product_parents=product_parents,
                execution_kind='cpu-source-audit-only')
            output = directory/'results/source-audit.json'
            _write_json(output, result)
            _write_json(directory/'complete.json', dict(config_sha256=_hash(config),
                files={'results/source-audit.json': file_hash(output)}, source_units=list(sources)))
            _state(directory, 'audited', completed, completed, primary_sources_ready=ready)
            return result
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            _state(directory, 'failed', completed, len(config['sources']), error=str(exc))
            errors = directory/'logs/errors.jsonl'
            with errors.open('a') as stream:
                stream.write(json.dumps(dict(error=str(exc), type=type(exc).__name__, updated_at=_now()))+'\n')
            raise


def budget_status(root):
    ledger = root/'runs/stage0/budget/ledger.jsonl'
    head_path = ledger.parent/'head.json'
    previous, total, ids = None, None, set()
    prefixes = [dict(events=0, last_sha256=None, spent_units=None)]
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            digest = row.pop('sha256')
            amount = row['total_units']
            if (row.get('previous_sha256') != previous or _hash(row) != digest
                    or row['event_id'] in ids or type(amount) not in [int, float]
                    or not 0 <= amount < float('inf') or (total is not None and amount < total)):
                raise ValueError('Stage 0 budget ledger integrity mismatch.')
            previous, total = digest, amount
            ids.add(row['event_id'])
            prefixes.append(dict(events=len(ids), last_sha256=previous, spent_units=total))
    published = json.loads(head_path.read_text()) if head_path.exists() else prefixes[0]
    count = published.get('events')
    if type(count) is not int or not 0 <= count < len(prefixes) or published != prefixes[count]:
        raise ValueError('Stage 0 budget publication mismatch; ledger was truncated or changed.')
    pending = count != len(ids)
    return dict(ceiling_units=CEILING, spent_units=total,
                remaining_units=None if total is None else max(0, CEILING-total),
                state=('unpublished-observation' if pending else
                       'unknown' if total is None else 'ceiling-reached' if total >= CEILING else 'observed'),
                events=len(ids), last_sha256=previous,
                gpu_authorized=False, scope='operator-recorded cumulative Stage 0 units; not an account-wide meter')


def _publish_budget(directory, current):
    _write_json(directory/'head.json', {key:current[key] for key in ['events', 'last_sha256', 'spent_units']})


def record_budget(path, output_root, observation):
    root = Path(output_root).resolve()
    _load(path, root)
    value = json.loads(Path(observation).read_text())
    required = {'event_id', 'session_id', 'total_units', 'observed_at', 'note'}
    if set(value) != required or not all(isinstance(value[k], str) and value[k].strip() for k in required-{'total_units'}):
        raise ValueError('Budget observation requires IDs, timestamp, note and cumulative units.')
    amount = value['total_units']
    if type(amount) not in [int, float] or not 0 <= amount < float('inf'):
        raise ValueError('Invalid cumulative budget amount.')
    datetime.fromisoformat(value['observed_at'])
    directory = root/'runs/stage0/budget'
    with _writer(directory):
        current = budget_status(root)
        ledger = directory/'ledger.jsonl'
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                row = json.loads(line)
                if row['event_id'] == value['event_id']:
                    if any(row[key] != value[key] for key in required):
                        raise ValueError('Budget event identity changed.')
                    _publish_budget(directory, current)
                    return budget_status(root)
        if current['spent_units'] is not None and amount < current['spent_units']:
            raise ValueError('Cumulative Stage 0 consumption cannot decrease.')
        row = dict(**value, previous_sha256=current['last_sha256'])
        row['sha256'] = _hash(row)
        with ledger.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        _publish_budget(directory, budget_status(root))
        return budget_status(root)


def status(path, output_root):
    root = Path(output_root).resolve()
    config, directory = _load(path, root)
    units = {label: _unit(directory, root, label, config) for label in _labels(config)}
    result = None
    marker = directory/'complete.json'
    if marker.exists():
        seal = json.loads(marker.read_text())
        output = directory/'results/source-audit.json'
        if (seal != dict(config_sha256=_hash(config), files={'results/source-audit.json': file_hash(output)},
                         source_units=_labels(config)) or any(row is None for row in units.values())):
            raise ValueError('Stage 0 audit completion mismatch.')
        result = json.loads(output.read_text())
        if result['sources'] != units:
            raise ValueError('Audit summary disagrees with sealed units.')
        _verify_parents(root, result['product_parents'])
    ownership = {}
    for label, folder in [('audit', directory), ('budget', root/'runs/stage0/budget')]:
        owner = folder/'writer.json'
        ownership[label] = json.loads(owner.read_text()) if owner.exists() else None
    return dict(name=config['name'], execution_kind='cpu-source-audit-only',
                completed_units=sum(row is not None for row in units.values()), total_units=len(units),
                audit=result, budget=budget_status(root), ownership=ownership,
                next_step='review source audit; no GPU qualification implemented in Ticket 01')


def recover_writer(path, output_root, owner_token, review_notes, *, product='audit'):
    root = Path(output_root).resolve()
    _, directory = _load(path, root)
    if product == 'budget':
        directory = root/'runs/stage0/budget'
    if not owner_token or not review_notes.strip():
        raise ValueError('Recovery requires the recorded owner token and review notes.')
    with run_lock(directory):
        marker = directory/'writer.json'
        owner = json.loads(marker.read_text())
        if owner.get('token') != owner_token:
            raise ValueError('Writer ownership token mismatch.')
        if owner.get('host') == platform.node():
            try:
                os.kill(owner['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('The recorded writer process is still active.')
        log = directory/'logs/ownership-recovery.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            stream.write(json.dumps(dict(owner=owner, review_notes=review_notes, updated_at=_now()))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        marker.unlink()
    return dict(state='ownership-released', product=product,
                scope='explicit operator recovery; no source or completion payload changed')


def console_summary(value):
    """Keep notebook outputs concise; full source evidence lives in the saved JSON."""
    if 'audit' in value:
        return {**value, 'audit': console_summary(value['audit']) if value['audit'] else None}
    if 'sources' not in value:
        return value
    result = {k: v for k, v in value.items() if k not in ['sources', 'product_parents']}
    if 'initialization' in result:
        result['initialization'] = {k: v for k, v in result['initialization'].items() if k != 'different_modules'}
        result['initialization']['different_module_count'] = len(value['initialization'].get('different_modules', []))
    result['sources'] = {label: dict(state=row['state'], path=row['path'],
        checkpoints={step: {k: v for k, v in cp.items() if k != 'update_norms'}
                     for step, cp in row.get('checkpoints', {}).items()})
        for label, row in value['sources'].items()}
    return result

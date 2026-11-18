"""Artifact catalog: canonical arms, cohorts, storage locations and response sets.

The catalog (``catalog/`` at the repository root, in Git) points at artifacts wherever they live; it never
moves or rewrites them. ``verify`` checks every locally reachable entry against the actual bytes (file
hashes, selected row counts, draws per question, cohort membership, per-row adapter hashes) and reports
entries stored elsewhere as ``not-local`` rather than as failures.

    python -m pilot_eval.catalog verify --catalog catalog --root local=/path/to/artifacts
    python -m pilot_eval.catalog find --catalog catalog --arm online-grpo --mode sampled
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

RESPONSE_KINDS = ('evaluation', 'training-data', 'training-rollout', 'replay')
C_SUITABILITY = ('on-policy', 'off-policy', 'fixed-text', 'greedy-only', 'insufficient')
EVIDENCE = ('verified', 'inferred', 'unknown')
LOCATION_KINDS = ('local-artifacts', 'local-path', 'hf-dataset', 'runpod-volume', 'drive')
REQUIRED = ('id', 'kind', 'arm', 'cohort', 'policy', 'draws_per_question', 'question_field', 'scorer',
            'selection', 'c_suitability', 'files', 'evidence')
LOCAL_KINDS = ('local-artifacts', 'local-path')


class CatalogError(ValueError):
    pass


def _yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise CatalogError('PyYAML is required: pip install "spar-pilot-eval[catalog]"') from exc
    return yaml


def _read(path):
    value = _yaml().safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise CatalogError(f'{path}: expected a mapping')
    return value


def ids_digest(ids):
    """SHA-256 of the sorted, newline-joined question IDs (order-independent cohort identity)."""
    return hashlib.sha256('\n'.join(sorted(str(i) for i in ids)).encode()).hexdigest()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


# --- loading and validation ------------------------------------------------------------------------

def load(directory):
    directory = Path(directory)
    locations = _read(directory / 'locations.yaml').get('locations') or {}
    arms = _read(directory / 'arms.yaml').get('arms') or {}
    cohorts = _read(directory / 'cohorts.yaml').get('cohorts') or {}
    responses = {}
    for path in sorted((directory / 'responses').glob('*.yaml')):
        entry = _read(path)
        entry['_file'] = str(path)
        if entry.get('id') in responses:
            raise CatalogError(f"duplicate response-set id {entry.get('id')!r} ({path})")
        responses[entry.get('id')] = entry
    catalog = dict(directory=str(directory), locations=locations, arms=arms, cohorts=cohorts, responses=responses)
    _validate(catalog)
    return catalog


def _validate(catalog):
    locations, arms, cohorts = catalog['locations'], catalog['arms'], catalog['cohorts']
    for name, loc in locations.items():
        if loc.get('kind') not in LOCATION_KINDS:
            raise CatalogError(f'location {name}: kind must be one of {LOCATION_KINDS}')
    aliases = {}
    for name, arm in arms.items():
        for alias in [name] + list(arm.get('aliases') or []):
            if alias in aliases and aliases[alias] != name:
                raise CatalogError(f'alias {alias!r} names two arms')
            aliases[alias] = name
        for step, ckpt in (arm.get('checkpoints') or {}).items():
            _check_file_ref(ckpt, locations, f'arm {name} checkpoint {step}')
    catalog['aliases'] = aliases
    for name, cohort in cohorts.items():
        for key in ('size', 'ids_sha256', 'source'):
            if key not in cohort:
                raise CatalogError(f'cohort {name}: missing {key}')
        _check_file_ref(cohort['source'], locations, f'cohort {name}', need_sha=False)
    for rid, entry in catalog['responses'].items():
        where = f"response set {rid!r} ({entry['_file']})"
        for key in REQUIRED:
            if key not in entry:
                raise CatalogError(f'{where}: missing {key}')
        if entry['kind'] not in RESPONSE_KINDS:
            raise CatalogError(f'{where}: kind must be one of {RESPONSE_KINDS}')
        if entry['arm'] not in arms:
            raise CatalogError(f"{where}: unknown arm {entry['arm']!r} (use the canonical name)")
        if entry['cohort'] not in cohorts:
            raise CatalogError(f"{where}: unknown cohort {entry['cohort']!r}")
        if entry['c_suitability'] not in C_SUITABILITY:
            raise CatalogError(f'{where}: c_suitability must be one of {C_SUITABILITY}')
        if entry.get('replay_of') is not None and entry['replay_of'] not in catalog['responses']:
            raise CatalogError(f"{where}: unknown replay_of {entry['replay_of']!r}")
        step = entry.get('checkpoint_step')
        if step is not None and step not in (arms[entry['arm']].get('checkpoints') or {}):
            raise CatalogError(f'{where}: arm {entry["arm"]} has no checkpoint {step}')
        evidence = entry['evidence']
        if not isinstance(evidence, dict) or evidence.get('status') not in EVIDENCE:
            raise CatalogError(f'{where}: evidence.status must be one of {EVIDENCE}')
        if evidence['status'] != 'verified' and not evidence.get('note'):
            raise CatalogError(f'{where}: evidence that is not verified needs a note saying what is missing')
        if not entry['files']:
            raise CatalogError(f'{where}: at least one file is required')
        if entry.get('cohort_coverage', 'full') not in ('full', 'subset'):
            raise CatalogError(f'{where}: cohort_coverage must be full or subset')
        for f in entry['files']:
            _check_file_ref(f, locations, where)
            if f['sha256'] == 'unknown':
                if evidence['status'] == 'verified':
                    raise CatalogError(f'{where}: a verified entry cannot have an unknown sha256')
                if locations[f['location']]['kind'] in LOCAL_KINDS:
                    raise CatalogError(f'{where}: local files must record their sha256, not unknown')


def _check_file_ref(ref, locations, where, need_sha=True):
    if not isinstance(ref, dict) or 'location' not in ref or 'path' not in ref:
        raise CatalogError(f'{where}: file references need location and path')
    if ref['location'] not in locations:
        raise CatalogError(f"{where}: unknown location {ref['location']!r}")
    if need_sha and not ref.get('sha256'):
        raise CatalogError(f'{where}: file reference needs sha256')


def resolve_arm(catalog, name):
    try:
        return catalog['aliases'][name]
    except KeyError:
        raise CatalogError(f'unknown arm or alias {name!r}') from None


def find(catalog, *, arm=None, mode=None, kind=None, cohort=None, c_suitability=None):
    arm = resolve_arm(catalog, arm) if arm else None
    out = []
    for entry in catalog['responses'].values():
        if (arm and entry['arm'] != arm) or (kind and entry['kind'] != kind) or \
                (cohort and entry['cohort'] != cohort) or (c_suitability and entry['c_suitability'] != c_suitability) or \
                (mode and (entry['policy'] or {}).get('mode') != mode):
            continue
        out.append(entry)
    return sorted(out, key=lambda e: e['id'])


# --- verification against bytes -------------------------------------------------------------------

def _resolve(ref, catalog, roots):
    loc = catalog['locations'][ref['location']]
    if loc['kind'] not in LOCAL_KINDS:
        return None
    if ref['location'] in roots:
        base = Path(roots[ref['location']])
    elif loc['kind'] == 'local-path' and loc.get('path'):
        base = Path(loc['path']).expanduser()
    else:
        return None
    return base / ref['path']


def _jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _cohort_ids(cohort, catalog, roots):
    path = _resolve(cohort['source'], catalog, roots)
    if path is None or not path.exists():
        return None
    field = cohort['source'].get('id_field', 'id')
    if path.suffix == '.jsonl':
        return [r[field] for r in _jsonl(path)]
    value = json.loads(path.read_text())
    for key in cohort['source'].get('json_path', []):
        value = value[key]
    return [v[field] if isinstance(v, dict) else v for v in value]


def verify(catalog, *, roots):
    checks = []

    def add(entry, status, detail):
        checks.append(dict(entry=entry, status=status, detail=detail))

    arm_sha = {}
    for name, arm in catalog['arms'].items():
        for step, ckpt in (arm.get('checkpoints') or {}).items():
            arm_sha[(name, step)] = ckpt['sha256']
            path = _resolve(ckpt, catalog, roots)
            if path is None:
                add(f'arm:{name}', 'not-local', f'checkpoint {step} at {ckpt["location"]}')
            elif not path.exists():
                add(f'arm:{name}', 'fail', f'checkpoint {step} missing: {path}')
            elif file_sha256(path) != ckpt['sha256']:
                add(f'arm:{name}', 'fail', f'checkpoint {step} sha256 differs: {path}')
            else:
                add(f'arm:{name}', 'pass', f'checkpoint {step} sha256 ok')

    cohort_ids = {}
    for name, cohort in catalog['cohorts'].items():
        ids = _cohort_ids(cohort, catalog, roots)
        if ids is None:
            add(f'cohort:{name}', 'not-local', f'source at {cohort["source"]["location"]}')
            continue
        cohort_ids[name] = set(ids)
        if len(ids) != cohort['size'] or len(set(ids)) != len(ids):
            add(f'cohort:{name}', 'fail', f'size {len(ids)} (unique {len(set(ids))}) != {cohort["size"]}')
        elif ids_digest(ids) != cohort['ids_sha256']:
            add(f'cohort:{name}', 'fail', 'ids_sha256 differs from the source IDs')
        else:
            add(f'cohort:{name}', 'pass', f'{len(ids)} IDs ok')

    for rid, entry in catalog['responses'].items():
        key = f'responses:{rid}'
        for ref in entry['files']:
            path = _resolve(ref, catalog, roots)
            if path is None:
                add(key, 'not-local', f'{ref["path"]} at {ref["location"]}')
                continue
            if not path.exists():
                add(key, 'fail', f'missing: {path}')
                continue
            if file_sha256(path) != ref['sha256']:
                add(key, 'fail', f'sha256 differs: {path}')
                continue
            if path.suffix != '.jsonl':
                add(key, 'pass', 'sha256 ok')
                continue
            problems = _check_rows(entry, ref, path, arm_sha, cohort_ids)
            add(key, 'fail' if problems else 'pass', '; '.join(problems) if problems else 'sha256 and rows ok')
    return dict(ok=not any(c['status'] == 'fail' for c in checks), checks=checks)


def _check_rows(entry, ref, path, arm_sha, cohort_ids):
    select = ref.get('select') or {}
    total, rows = 0, []
    for row in _jsonl(path):
        total += 1
        if all(row.get(k) == v for k, v in select.items()):
            rows.append(row)
    problems = []
    if 'rows_total' in ref and total != ref['rows_total']:
        problems.append(f'rows_total {total} != {ref["rows_total"]}')
    if 'rows_selected' in ref and len(rows) != ref['rows_selected']:
        problems.append(f'rows_selected {len(rows)} != {ref["rows_selected"]}')
    field = entry['question_field']
    per_q = collections.Counter(str(r.get(field)) for r in rows)
    want = entry['draws_per_question']
    if want is not None and set(per_q.values()) != {want}:
        problems.append(f'draws per question {sorted(set(per_q.values()))} != {want}')
    if entry['cohort'] in cohort_ids:
        cohort = cohort_ids[entry['cohort']]
        extra, missing = set(per_q) - cohort, cohort - set(per_q)
        if extra:
            problems.append(f'{len(extra)} question IDs not in cohort {entry["cohort"]}')
        if missing and entry.get('cohort_coverage', 'full') == 'full':
            problems.append(f'{len(missing)} cohort {entry["cohort"]} questions missing (declare '
                            f'cohort_coverage: subset if intended)')
    expected = entry.get('row_adapter_sha256') or arm_sha.get((entry['arm'], entry.get('checkpoint_step')))
    adapter_field = entry.get('adapter_field', 'adapter_sha256')
    if expected and rows and adapter_field in rows[0]:
        bad = sum(r.get(adapter_field) != expected for r in rows)
        if bad:
            problems.append(f'{bad} rows have {adapter_field} != {expected[:12]}')
    return problems


# --- command line ------------------------------------------------------------------------------------

def _roots(values):
    out = {}
    for value in values or []:
        name, sep, path = value.partition('=')
        if not sep:
            raise CatalogError(f'--root expects NAME=PATH, not {value!r}')
        out[name] = path
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pilot_eval.catalog')
    sub = parser.add_subparsers(dest='command', required=True)
    v = sub.add_parser('verify', help='check catalog entries against local bytes')
    f = sub.add_parser('find', help='list response sets matching filters')
    for p in (v, f):
        p.add_argument('--catalog', default='catalog')
        p.add_argument('--root', action='append', help='NAME=PATH override for a local location')
    for opt in ('arm', 'mode', 'kind', 'cohort', 'c-suitability'):
        f.add_argument(f'--{opt}')
    args = parser.parse_args(argv)
    catalog = load(args.catalog)
    if args.command == 'find':
        for e in find(catalog, arm=args.arm, mode=args.mode, kind=args.kind, cohort=args.cohort,
                      c_suitability=args.c_suitability):
            pol = e['policy'] or {}
            print(f"{e['id']}\t{e['arm']}\tstep={e.get('checkpoint_step')}\t{pol.get('mode')}\t"
                  f"x{e['draws_per_question']}\t{e['c_suitability']}\t{e['evidence']['status']}\t"
                  f"{e['files'][0]['location']}:{e['files'][0]['path']}")
        return 0
    report = verify(catalog, roots=_roots(args.root))
    for c in report['checks']:
        print(f"{c['status'].upper():9s} {c['entry']}: {c['detail']}")
    counts = collections.Counter(c['status'] for c in report['checks'])
    print(f"\n{dict(counts)} -> {'OK' if report['ok'] else 'FAILED'}")
    return 0 if report['ok'] else 1


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())

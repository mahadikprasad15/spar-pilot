"""Four singleton reference answers using the production cached-generation engine.

Performance evidence only; never approves production or renews a reservation.
Requires a completed same-session diagnostic, no active reservation, and explicit
rate/time/usage bounds. Does not reread unrelated SVD/random-basis artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid


def reserve_units(rate, seconds):
    import math
    if not all(math.isfinite(v) and v > 0 for v in [rate, seconds]):
        raise ValueError('Positive finite rate and time bounds are required.')
    return rate * seconds / 3600


def make_work(rows, selected_ids):
    chosen = selected_ids[:2]
    if len(chosen) != 2 or len(set(chosen)) != 2:
        raise ValueError('Two distinct fixed questions are required.')
    by_id = {row['id']: (i, row) for i, row in enumerate(rows)}
    if any(key not in by_id for key in chosen):
        raise ValueError('Diagnostic question missing from original cohort.')
    return [dict(variant=v, row=by_id[key][1], seed=42 + by_id[key][0])
            for v in ['V0a', 'V0b'] for key in chosen]


def summarize(rows, load_seconds):
    seconds = sum(row['seconds'] for row in rows)
    tokens = sum(len(row['generated_token_ids']) for row in rows)
    return dict(answers=len(rows), generated_tokens=tokens,
                generation_seconds=seconds, model_load_seconds=load_seconds,
                aggregate_tokens_per_second=tokens / seconds if seconds else None,
                production_authorized=False, all_variants_eta_seconds=None,
                limitation='Two questions and two references only; edited variants and length tails unmeasured.')


def load_sources(root, path):
    from pilot_eval.stage0_qualification import _verify, _implementation
    from pilot_eval.stage0_sources import inside
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash

    path = inside(root, path)
    q = json.loads(path.read_text())
    if q.get('fixture') or q.get('qualification_kind') == 'conditional-intruders':
        raise ValueError('This helper is only for the real primary qualification.')
    if q['implementation'] != _implementation():
        raise ValueError('Use the same pinned implementation as qualification.')
    qdir = inside(root, q['run_path'])
    report = _verify(q, qdir)  # Small sealed diagnostic results; no spectral scans.
    if not report['eligible_for_review']:
        raise ValueError('Numerical diagnostic is not eligible for review.')
    for key, field in [('cohort_path', 'cohort_sha256'), ('edits_path', 'edits_sha256')]:
        if file_hash(inside(root, q[key])) != q[field]:
            raise ValueError(f'Changed source: {key}')
    for name, digest in q['tokenizer_files'].items():
        if file_hash(inside(root, q['weights_dir']) / name) != digest:
            raise ValueError('Changed tokenizer.')
    edit = json.loads(inside(root, q['edits_path']).read_text())
    edir = inside(root, edit['run_path'])
    complete = json.loads((edir / 'complete.json').read_text())
    if complete['config_sha256'] != _hash(edit):
        raise ValueError('Edit catalogue identity changed.')
    # Only original reference adapters are used in this profile.
    for variant in ['V0a', 'V0b']:
        folder = inside(root, q['variants'][variant]['adapter_path'])
        expected = complete['files'][f'adapters/{variant}/complete.json']
        if file_hash(folder / 'complete.json') != expected:
            raise ValueError('Reference adapter receipt changed.')
        seal = json.loads((folder / 'complete.json').read_text())
        if seal['identity'] != dict(config_sha256=_hash(edit), variant=variant):
            raise ValueError('Reference adapter identity mismatch.')
        for name, digest in seal['files'].items():
            if file_hash(inside(folder, name)) != digest:
                raise ValueError('Reference adapter payload changed.')
    cohort = json.loads(inside(root, q['cohort_path']).read_text())
    return q, report, cohort['items'], file_hash(path), file_hash(qdir / 'complete.json')


def run(root, qualification, name, rate, seconds, observed_units, *, dependencies=None):
    from pilot_eval.run import _write_json
    from pilot_eval.sft import safe_name
    from pilot_eval.stage0_budget import reserve, reservations
    from pilot_eval.stage0_workflow import _writer, _state, record_budget
    from pilot_eval.stage0_qualification_backend import deadline_guard
    from pilot_eval.stage0_evaluation_backend import EvaluationDependencies
    from pilot_eval.workflow import _hash, _save_frozen
    from pilot_eval.training import file_hash

    root = Path(root).resolve()
    safe_name(name)
    reserve_units(rate, seconds)
    directory = root / 'runs/stage0' / name / 'cached-profile'
    if directory.exists():
        raise ValueError('Profile already exists. Inspect saved outputs; no automatic rerun or renewed spend.')
    if any(row['state'] == 'active' for row in reservations(root).values()):
        raise ValueError('Settle the previous reservation first.')
    print('Checking saved diagnostic receipts and reference adapters; no SVD/random-basis scan.', flush=True)
    q, report, cohort, source_hash, diagnostic_hash = load_sources(root, qualification)
    deps = dependencies or EvaluationDependencies()
    runtime = deps.runtime()
    if runtime != report['runtime']:
        raise ValueError('Profile must use the exact diagnostic runtime/session.')
    work = make_work(cohort, q['sequence_ids'])
    config = dict(protocol='stage0-cached-reference-profile-v1', name=name,
                  run_path=str(directory.relative_to(root)), qualification_sha256=source_hash,
                  diagnostic_sha256=diagnostic_hash, runtime=runtime,
                  script_sha256=file_hash(Path(__file__)),
                  variant_order=['V0a', 'V0b'],
                  variants={key: q['variants'][key] for key in ['V0a', 'V0b']},
                  weights_dir=q['weights_dir'], model_identity=q['model_identity'],
                  prompt_contract=q.get('evaluation_config', {}).get('prompt_template', {}),
                  decoding={**q['decoding'], 'use_cache': True}, seed=42,
                  items=[dict(variant=w['variant'], id=w['row']['id'], seed=w['seed']) for w in work],
                  max_units_per_hour=rate, max_seconds=seconds,
                  production_authorized=False, execution_kind=runtime['execution_kind'])
    with _writer(directory):
        _save_frozen(directory / 'config.json', config)
        observation = directory / 'usage-before.json'
        _write_json(observation, dict(event_id='cached-profile-' + uuid.uuid4().hex,
                    session_id=runtime['session_id'], total_units=observed_units,
                    observed_at=datetime.now(timezone.utc).isoformat(),
                    note='Operator-reported cumulative Stage0 usage before bounded cached profile.'))
        edit = json.loads((root / q['edits_path']).read_text())
        spectral = json.loads((root / edit['spectral_path']).read_text())
        audit_plan = root / spectral['audit_path']
        record_budget(audit_plan, root, observation)
        approval = dict(reviewer='Prasad', notes='Approved four-answer performance-only cached benchmark.',
                        session_id=runtime['session_id'], max_units_per_hour=rate,
                        max_seconds=seconds, source_reviewed=True)
        _save_frozen(directory / 'approval.json', approval)
        reservation = reserve(root, _hash(config), approval)
        _save_frozen(directory / 'reservation.json', reservation)
        print(f'Reserved {reservation["reserved_units"]:.4f} units; {len(work)} answers; {seconds}s workload bound.', flush=True)
        start = time.monotonic()
        engine = None
        results = []
        try:
            with deadline_guard(seconds, runtime['execution_kind'] == 'real-cuda'):
                load_start = time.monotonic()
                engine = deps.engine(config, root)
                load_seconds = time.monotonic() - load_start
                if runtime['execution_kind'] == 'real-cuda':
                    _write_json(directory / 'cache-policy.json', dict(
                        explicit_generate_use_cache=True,
                        model_config_use_cache=engine.model.config.use_cache,
                        generation_config_use_cache=engine.model.generation_config.use_cache))
                inputs = engine.prepare_inputs([w['row'] for w in work[:2]])
                _save_frozen(directory / 'tokenized-inputs.json', inputs)
                print(f'Model loaded: {load_seconds:.1f}s; explicit generation use_cache=True.', flush=True)
                for i, item in enumerate(work):
                    if time.monotonic() - start >= seconds:
                        raise TimeoutError('Benchmark workload bound reached.')
                    print(f'{i+1}/4 starting: {item["variant"]}, {item["row"]["id"]}', flush=True)
                    begin = time.monotonic()
                    answer = engine.generate([item['row']], item['variant'], item['seed'])[0]
                    duration = time.monotonic() - begin
                    integrity = engine.check_integrity(full=False)
                    if not integrity['base_unchanged'] or not integrity['adapters_unchanged']:
                        raise ValueError('Model/adapter integrity failed.')
                    value = dict(**answer, variant=item['variant'], id=item['row']['id'],
                                 prompt=item['row']['prompt'], gold=item['row']['gold'],
                                 seed=item['seed'], seconds=duration, integrity=integrity)
                    _write_json(directory / 'answers' / f'{i:02d}.json', value)
                    with (directory / 'responses.jsonl').open('a') as stream:
                        stream.write(json.dumps(value) + '\n')
                        stream.flush()
                        __import__('os').fsync(stream.fileno())
                    results.append(value)
                    _state(directory, 'running', len(results), 4,
                           reservation_id=reservation['reservation_id'])
                    print(f'{i+1}/4 saved: {len(answer["generated_token_ids"])} tokens, {duration:.2f}s.', flush=True)
                integrity = engine.check_integrity(full=True)
                if not integrity['base_unchanged'] or not integrity['adapters_unchanged']:
                    raise ValueError('Final full integrity check failed.')
                summary = {**summarize(results, load_seconds), 'runtime': runtime,
                           'reservation_id': reservation['reservation_id'], 'integrity': integrity}
                _write_json(directory / 'summary.json', summary)
                _write_json(directory / 'complete.json', dict(config_sha256=_hash(config),
                            files={str(p.relative_to(directory)): file_hash(p)
                                   for p in [directory / 'summary.json', directory / 'responses.jsonl',
                                             *sorted((directory / 'answers').glob('*.json'))]}))
                _state(directory, 'complete', 4, 4, reservation_id=reservation['reservation_id'])
                print(json.dumps(summary, indent=2), flush=True)
        except BaseException as exc:
            _state(directory, 'failed', len(results), 4, error=str(exc),
                   reservation_id=reservation['reservation_id'])
            raise
        finally:
            _write_json(directory / 'timing.json', {'workload_seconds': time.monotonic() - start})
            if engine is not None:
                engine.close()
            print('Record actual cumulative usage and settle this profile reservation. Production remains unapproved.', flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['repo-dir', 'output-root', 'qualification', 'name']:
        parser.add_argument('--' + key, required=True)
    for key in ['rate', 'seconds', 'observed-units']:
        parser.add_argument('--' + key, required=True, type=float)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    run(args.output_root, args.qualification, args.name, args.rate, args.seconds, args.observed_units)

"""Bounded batch memory/speed probe. Never authorizes production batching.

Needs the previously uploaded stage0_cached_profile.py and its completed output.
Short timings repeat the SAME TWO questions to fill larger batches; this is
throughput evidence, not an accuracy sample. Memory stress uses the longest
original prompt plus the 1024-token generation allowance and one cached step.
That full-prefix prefill is more conservative than incremental generation.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import runpy
import sys
import time
import uuid


def profile_cases(engine, rows, baseline, sizes, record, memory_probe):
    if len(rows) != 2 or len({r['id'] for r in rows}) != 2:
        raise ValueError('Exactly two fixed distinct timing questions are required.')
    reference = {(r['variant'], r['id']): r for r in baseline}
    expected = {(v, r['id']) for v in ['V0a', 'V0b'] for r in rows}
    if set(reference) != expected:
        raise ValueError('Singleton reference coverage differs.')
    if sizes != sorted(set(sizes)) or any(type(b) is not int or b < 1 for b in sizes):
        raise ValueError('Batch sizes must be positive, distinct and increasing.')
    results = []
    for size in sizes:
        print(f'Batch {size}: maximum-context memory stress.', flush=True)
        memory = memory_probe(size)
        record(f'b{size}-memory.json', memory)
        summary = dict(batch_size=size, memory=memory, production_authorized=False,
                       batch_qualified=False, repeated_two_question_probe=True)
        if not memory['fits']:
            summary['stop_reason'] = 'conservative-memory-stress-oom'
            results.append(summary)
            record('summary.json', results)
            break
        if size == 1:
            seconds = sum(r['seconds'] for r in baseline)
            tokens = sum(len(r['generated_token_ids']) for r in baseline)
            summary.update(generation_seconds=seconds, generated_tokens=tokens,
                           answers=4, exact_singleton_token_agreement=True,
                           timing_reused=True, tokens_per_second=tokens / seconds)
        else:
            batch = [rows[i % 2] for i in range(size)]
            seconds = 0.
            tokens = 0
            exact = True
            peaks = []
            for variant in ['V0a', 'V0b']:
                print(f'Batch {size}: {variant}, {size} answers; use_cache=True.', flush=True)
                torch = getattr(engine, 'torch', None)
                if torch is not None:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                answers = engine.generate(batch, variant, 42)
                elapsed = time.monotonic() - start
                if len(answers) != size:
                    raise ValueError('Engine returned the wrong number of answers.')
                if torch is not None:
                    peaks.append(dict(variant=variant, allocated=torch.cuda.max_memory_allocated(),
                                      reserved=torch.cuda.max_memory_reserved()))
                values = []
                for index, (row, answer) in enumerate(zip(batch, answers)):
                    same = answer['generated_token_ids'] == reference[(variant, row['id'])]['generated_token_ids']
                    exact = exact and same
                    tokens += len(answer['generated_token_ids'])
                    values.append(dict(**answer, id=row['id'], variant=variant,
                                       prompt=row['prompt'], gold=row['gold'], seed=42,
                                       duplicate_index=index, matches_saved_singleton=same,
                                       batch_seconds=elapsed, performance_only=True))
                record(f'b{size}-{variant}-answers.json', values)
                seconds += elapsed
            summary.update(generation_seconds=seconds, generated_tokens=tokens,
                           answers=2 * size, exact_singleton_token_agreement=exact,
                           timing_reused=False, tokens_per_second=tokens / seconds,
                           generation_memory_peaks=peaks)
        results.append(summary)
        record('summary.json', results)
        print(json.dumps(summary), flush=True)
    return results


def memory_stress(engine, longest_tokens, size):
    """Conservative full-prefix + cached-step peak, not an exact generation peak.

    Qwen2 logits_to_keep=1 avoids a full sequence-by-vocabulary allocation:
    https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2
    """
    torch = engine.torch
    outputs = later = tokens = mask = next_token = next_mask = None
    length = len(longest_tokens) + engine.config['decoding']['max_new_tokens'] - 1
    filler = engine.tokenizer.encode('0', add_special_tokens=False)[0]
    sequence = longest_tokens + [filler] * (length - len(longest_tokens))
    engine.model.set_adapter('V0a')
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        tokens = torch.tensor([sequence] * size, device='cuda:0', dtype=torch.long)
        mask = torch.ones_like(tokens)
        with torch.inference_mode():
            outputs = engine.model(input_ids=tokens, attention_mask=mask, use_cache=True,
                                   logits_to_keep=1, return_dict=True)
            if outputs.past_key_values is None:
                raise ValueError('Memory probe did not return a KV cache.')
            next_token = torch.full((size, 1), filler, device='cuda:0', dtype=torch.long)
            next_mask = torch.ones((size, length + 1), device='cuda:0', dtype=torch.long)
            later = engine.model(input_ids=next_token, attention_mask=next_mask,
                                 past_key_values=outputs.past_key_values, use_cache=True,
                                 logits_to_keep=1, return_dict=True)
        torch.cuda.synchronize()
        return dict(fits=True, context_tokens=length + 1,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
                    seconds=time.monotonic() - start,
                    scope='conservative full-prefix prefill plus one cached decode step')
    except torch.cuda.OutOfMemoryError as exc:
        return dict(fits=False, context_tokens=length + 1, error=str(exc),
                    seconds=time.monotonic() - start,
                    scope='stress OOM does not establish incremental-generation OOM')
    finally:
        del outputs, later, tokens, mask, next_token, next_mask
        gc.collect()
        torch.cuda.empty_cache()


def main(args, *, memory_probe=None):
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    from pilot_eval.run import _write_json
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash, _save_frozen
    from pilot_eval.stage0_sources import inside
    from pilot_eval.sft import safe_name
    from pilot_eval.stage0_workflow import _writer, _state, record_budget
    from pilot_eval.stage0_budget import reservations, settle, reserve
    from pilot_eval.stage0_evaluation_backend import EvaluationDependencies
    from pilot_eval.stage0_qualification_backend import deadline_guard

    shared = runpy.run_path(args.cached_helper)
    root = Path(args.output_root).resolve()
    safe_name(args.name)
    shared['reserve_units'](args.rate, args.seconds)
    if (args.batch_sizes != sorted(set(args.batch_sizes))
            or not args.batch_sizes or any(b < 1 for b in args.batch_sizes)):
        raise ValueError('Select positive distinct increasing batch sizes before launch.')
    directory = root / 'runs/stage0' / args.name / 'batch-profile'
    if directory.exists():
        raise ValueError('Profile exists. Inspect partial results; no automatic restart or extra spend.')
    baseline_dir = inside(root, args.baseline)
    base_config = json.loads((baseline_dir / 'config.json').read_text())
    seal = json.loads((baseline_dir / 'complete.json').read_text())
    if seal['config_sha256'] != _hash(base_config):
        raise ValueError('Baseline config changed.')
    for name, digest in seal['files'].items():
        if file_hash(inside(baseline_dir, name)) != digest:
            raise ValueError('Baseline payload changed.')
    baseline = [json.loads(p.read_text()) for p in sorted((baseline_dir / 'answers').glob('*.json'))]
    if len(baseline) != 4 or base_config['decoding']['use_cache'] is not True:
        raise ValueError('Expected the completed four-answer cached singleton baseline.')
    print('Checking small diagnostic/reference receipts; no spectral scan.', flush=True)
    q, diagnostic, cohort, qhash, dhash = shared['load_sources'](root, args.qualification)
    deps = EvaluationDependencies()
    runtime = deps.runtime()
    if runtime != diagnostic['runtime'] or runtime != base_config['runtime']:
        raise ValueError('Use the same GPU/runtime/session as both previous measurements.')
    if qhash != base_config['qualification_sha256']:
        raise ValueError('Baseline and batch probe qualification identities differ.')
    if base_config['decoding'] != {**q['decoding'], 'use_cache': True}:
        raise ValueError('Generation policies differ.')
    verifier_file = getattr(args, 'verifier_file', None)
    if verifier_file:
        import pilot_eval.stage0_evaluation_backend as backend
        exact = runpy.run_path(verifier_file)['verify_loaded_adapter']
        backend.verify_loaded_adapter = exact
        print('Using recorded exact-adapter-name verifier; tensor checks remain exact.', flush=True)
    ids = q['sequence_ids'][:2]
    by_id = {row['id']: row for row in cohort}
    rows = [by_id[key] for key in ids]
    if {(r['variant'], r['id']) for r in baseline} != {(v, key) for v in ['V0a', 'V0b'] for key in ids}:
        raise ValueError('Baseline question/variant selection differs.')
    edit = json.loads(inside(root, q['edits_path']).read_text())
    spectral = json.loads(inside(root, edit['spectral_path']).read_text())
    audit_plan = inside(root, spectral['audit_path'])
    allocations = reservations(root)
    previous = json.loads((baseline_dir / 'reservation.json').read_text())
    active = [r for r in allocations.values() if r['state'] == 'active']
    settlement = None
    if active:
        if len(active) != 1:
            raise ValueError('More than one active allocation; inspect the ledger.')
        failed_profile = getattr(args, 'settle_previous_profile', None)
        if failed_profile:
            previous_dir = inside(root, failed_profile)
            previous_config = json.loads((previous_dir / 'config.json').read_text())
            previous_state = json.loads((previous_dir / 'meta/status.json').read_text())
            previous_receipt = json.loads((previous_dir / 'reservation.json').read_text())
            if (previous_config.get('protocol') != 'stage0-batch-performance-profile-v1'
                    or previous_config.get('runtime') != runtime
                    or previous_state.get('state') != 'failed'
                    or previous_state.get('completed') != 0
                    or (previous_dir / 'writer.json').exists()
                    or (previous_dir / 'complete.json').exists()
                    or active[0]['reservation_id'] != previous_receipt['reservation_id']
                    or active[0]['job_sha256'] != _hash(previous_config)):
                raise ValueError('Previous profile is not a stopped, failed-before-measurement attempt.')
            settlement = previous_receipt['reservation_id']
        elif (args.settle_baseline and active[0]['reservation_id'] == previous['reservation_id']
              and active[0]['job_sha256'] == _hash(base_config)):
            settlement = previous['reservation_id']
        else:
            raise ValueError('Settle the exact previous workload before launching another profile.')
    config = dict(**base_config)
    config.update(name=args.name, run_path=str(directory.relative_to(root)),
                  protocol='stage0-batch-performance-profile-v1', batch_sizes=args.batch_sizes,
                  baseline_complete_sha256=file_hash(baseline_dir / 'complete.json'),
                  script_sha256=file_hash(Path(__file__)), cached_helper_sha256=file_hash(Path(args.cached_helper)),
                  verifier_sha256=file_hash(Path(verifier_file)) if verifier_file else None,
                  settles_failed_profile=getattr(args, 'settle_previous_profile', None),
                  max_units_per_hour=args.rate, max_seconds=args.seconds,
                  memory_scope='longest prompt in original150 + 1024 allowance; conservative full-prefix prefill',
                  variants=q['variants'], variant_order=['V0a', 'V0b'] + sorted(set(q['variants']) - {'V0a', 'V0b'}),
                  timing_scope='two repeated prompts, two references; not a diverse evaluation cohort',
                  batch_qualified=False, production_authorized=False)
    with _writer(directory):
        _save_frozen(directory / 'config.json', config)
        usage = directory / 'usage-before.json'
        _write_json(usage, dict(event_id='batch-profile-' + uuid.uuid4().hex,
                    session_id=runtime['session_id'], total_units=args.observed_units,
                    observed_at=datetime.now(timezone.utc).isoformat(),
                    note='Operator-reported cumulative Stage0 usage before bounded batch probe.'))
        record_budget(audit_plan, root, usage)
        if settlement:
            settle(root, settlement)
            print('Reconciled previous profile reservation against operator usage.', flush=True)
        approval = dict(reviewer='Prasad', notes='Bounded batch capacity/speed probe; not production approval.',
                        session_id=runtime['session_id'], max_units_per_hour=args.rate,
                        max_seconds=args.seconds, source_reviewed=True)
        _save_frozen(directory / 'approval.json', approval)
        allocation = reserve(root, _hash(config), approval)
        _save_frozen(directory / 'reservation.json', allocation)
        print(f'Reserved {allocation["reserved_units"]:.4f} units; {args.seconds}s model-work bound.', flush=True)
        engine = None
        try:
            with deadline_guard(args.seconds, True):
                start = time.monotonic()
                engine = deps.engine(config, root)  # Load all10 adapters, as in production.
                print(f'Loaded all ten adapters: {time.monotonic()-start:.1f}s.', flush=True)
                inputs = engine.prepare_inputs(cohort)
                _save_frozen(directory / 'tokenized-inputs.json', inputs)
                longest = max(engine.inputs.values(), key=len)
                def record(name, value):
                    _write_json(directory / name, value)
                    if name.endswith('-answers.json'):
                        with (directory / 'responses.jsonl').open('a') as stream:
                            for answer in value:
                                stream.write(json.dumps({**answer, 'batch_size': len(value)}) + '\n')
                            stream.flush()
                            __import__('os').fsync(stream.fileno())
                    if name == 'summary.json':
                        _state(directory, 'running', len(value), len(args.batch_sizes),
                               reservation_id=allocation['reservation_id'])
                results = profile_cases(engine, rows, baseline, args.batch_sizes, record,
                                        lambda size: (memory_probe or memory_stress)(engine, longest, size))
                integrity = engine.check_integrity(full=True)
                if not integrity['base_unchanged'] or not integrity['adapters_unchanged']:
                    raise ValueError('Final integrity check failed.')
                _write_json(directory / 'integrity.json', integrity)
                _write_json(directory / 'complete.json', dict(config_sha256=_hash(config),
                            files={str(p.relative_to(directory)): file_hash(p)
                                   for p in [*directory.glob('*.json'), *directory.glob('*.jsonl')]
                                   if p.name != 'complete.json'}))
                _state(directory, 'complete', len(results), len(args.batch_sizes),
                       stop_reason=results[-1].get('stop_reason'), reservation_id=allocation['reservation_id'])
        except BaseException as exc:
            _state(directory, 'failed', len(list(directory.glob('b*-memory.json'))),
                   len(args.batch_sizes), error=str(exc), reservation_id=allocation['reservation_id'])
            raise
        finally:
            if engine is not None:
                engine.close()
            print('Profile allocation remains active: record actual usage and settle it before another job.', flush=True)
    print('Results:', directory / 'summary.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['repo-dir', 'output-root', 'qualification', 'baseline', 'cached-helper', 'name']:
        parser.add_argument('--' + key, required=True)
    for key in ['rate', 'seconds', 'observed-units']:
        parser.add_argument('--' + key, required=True, type=float)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--settle-baseline', action='store_true')
    parser.add_argument('--settle-previous-profile', help='Exact failed-before-measurement profile directory to reconcile.')
    parser.add_argument('--verifier-file', help='Recorded exact-name verifier for the older pinned backend.')
    main(parser.parse_args())

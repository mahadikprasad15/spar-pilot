"""Versioned fixed-pool training plan and sealed-boundary recovery."""
import json
from pathlib import Path

from pilot_eval.grpo_prepare import _within, load_grpo_prepared
from pilot_eval.grpo_preflight import _finite, load_frozen
from pilot_eval.grpo_training import CHECKPOINTS, _history, _monitor_events, _seal, _verified_checkpoints
from pilot_eval.grpo_baseline import _percentile
from pilot_eval.offline_grpo_engine import OfflineGRPOTrainingEngine, verify_initial_adapter
from pilot_eval.offline_grpo_pool import load_pool
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock, verified_checkpoints
from pilot_eval.workflow import _hash, _save_frozen


def _identity():
    from pilot_eval import offline_grpo_engine, offline_grpo_diagnostics, grpo_algorithm, grpo_train_engine, scoring
    return {k: file_hash(Path(v.__file__)) for k, v in {
        'workflow': __import__(__name__, fromlist=['']), 'offline_engine': offline_grpo_engine,
        'diagnostics': offline_grpo_diagnostics, 'online_engine': grpo_train_engine,
        'window': grpo_algorithm, 'scorer': scoring}.items()}


def prepare(frozen_path, pool_path, checkpoint_zero, output_root, name, *, runtime,
            weights_sha256, config_sha256):
    root = Path(output_root).resolve(); safe_name(name)
    frozen_path = _within(root, frozen_path); pool_path = _within(root, pool_path)
    checkpoint_zero = _within(root, checkpoint_zero)
    frozen = load_frozen(frozen_path, root)
    plan, _, _, _ = load_grpo_prepared(root / frozen['source_path'], root)
    windows = load_pool(pool_path, root)
    pool_plan = json.loads(pool_path.read_text())
    if pool_plan['source_sha256'] != file_hash(frozen_path):
        raise ValueError('pool does not bind the selected frozen online plan')
    groups = [[r['id'] for r in window] for window in windows]
    if groups != frozen['optimizer_groups']:
        raise ValueError('pool windows differ from frozen optimizer order')
    if checkpoint_zero.name != 'checkpoint-0' or verified_checkpoints(checkpoint_zero.parent.parent).get(0) != checkpoint_zero:
        raise ValueError('verified checkpoint-0 required')
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
           for v in (weights_sha256, config_sha256)):
        raise ValueError('independent checkpoint-0 SHA-256 identities required')
    verify_initial_adapter(checkpoint_zero, weights_sha256, config_sha256)
    reference = plan['source_evidence']['checkpoints']['0']
    reference_path = _within(root, reference['path'])
    if file_hash(reference_path/'complete.json') != reference['complete_sha256']:
        raise ValueError('reference checkpoint-0 seal changed')
    reference_seal = json.loads((reference_path/'complete.json').read_text())
    if (weights_sha256 != reference['adapter_sha256']
            or config_sha256 != reference_seal['files']['adapter_config.json']):
        raise ValueError('checkpoint-0 differs from verified reference identity')
    if any(policy['action'] != 'flag' for policy in frozen['review']['monitors'].values()):
        raise ValueError('offline monitors must flag and continue')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('explicit reviewed runtime identity required')
    directory = root / 'runs' / 'offline-grpo' / name / 'training'
    config = dict(protocol='offline-grpo-training-v1', method='fixed-unweighted',
        frozen_path=str(frozen_path.relative_to(root)), frozen_sha256=file_hash(frozen_path),
        pool_path=str(pool_path.relative_to(root)), pool_sha256=file_hash(pool_path),
        checkpoint_zero=str(checkpoint_zero.relative_to(root)),
        weights_sha256=weights_sha256,
        adapter_config_sha256=config_sha256,
        checkpoint_seal_sha256=file_hash(checkpoint_zero/'complete.json'),
        model=plan['model'], model_revision=plan['model_revision'], tokenizer_revision=plan['tokenizer_revision'],
        adapter=plan['adapter'], dtype=plan['dtype'], prompt_template=plan['source_prompt_contract'],
        scorer='gsm8k-flexible-v3', seed=plan['seed'],
        settings=frozen['settings'], optimizer=frozen['optimizer'], optimizer_groups=groups,
        monitor_policy=frozen['review']['monitors'], runtime=runtime,
        length_change_definition='step-lag-8', run_path=str(directory.relative_to(root)), implementation=_identity())
    target = root / 'plans' / name / 'offline.training.json'
    with run_lock(directory):
        _save_frozen(directory/'config.json', config); _save_frozen(target, config)
        _save_frozen(directory/'meta/run_manifest.json', dict(
            config_path=str(target.relative_to(root)), run_path=config['run_path'],
            checkpoint_steps=CHECKPOINTS, pool_path=config['pool_path']))
        if not (directory/'meta/status.json').exists():
            _write_state(directory, 'prepared', 0, 64)
    return target


def _load(path, root):
    path = _within(root, path); config = json.loads(path.read_text())
    if config.get('protocol') != 'offline-grpo-training-v1' or config.get('method') != 'fixed-unweighted':
        raise ValueError('unsupported offline training plan')
    if config['implementation'] != _identity(): raise ValueError('offline implementation changed')
    directory = _within(root, config['run_path'])
    if json.loads((directory/'config.json').read_text()) != config: raise ValueError('training plan changed')
    frozen_path = _within(root, config['frozen_path']); pool_path = _within(root, config['pool_path'])
    if file_hash(frozen_path) != config['frozen_sha256'] or file_hash(pool_path) != config['pool_sha256']:
        raise ValueError('training source identity changed')
    frozen = load_frozen(frozen_path, root)
    plan, _, _, _ = load_grpo_prepared(root/frozen['source_path'], root)
    if json.loads(pool_path.read_text())['source_sha256'] != file_hash(frozen_path):
        raise ValueError('pool frozen-plan identity changed')
    if (config['settings'] != frozen['settings'] or config['optimizer'] != frozen['optimizer']
            or config['optimizer_groups'] != frozen['optimizer_groups']
            or config['monitor_policy'] != frozen['review']['monitors']):
        raise ValueError('training settings differ from frozen online plan')
    reference = plan['source_evidence']['checkpoints']['0']
    reference_path = _within(root, reference['path'])
    if file_hash(reference_path/'complete.json') != reference['complete_sha256']:
        raise ValueError('reference checkpoint-0 seal changed')
    reference_seal = json.loads((reference_path/'complete.json').read_text())
    if (config['weights_sha256'] != reference['adapter_sha256']
            or config['adapter_config_sha256'] != reference_seal['files']['adapter_config.json']):
        raise ValueError('checkpoint-0 reference identity changed')
    windows = load_pool(pool_path, root)
    if [[r['id'] for r in w] for w in windows] != config['optimizer_groups']:
        raise ValueError('pool window order changed')
    zero = _within(root, config['checkpoint_zero'])
    if file_hash(zero/'complete.json') != config['checkpoint_seal_sha256'] or verified_checkpoints(zero.parent.parent).get(0) != zero:
        raise ValueError('checkpoint-0 seal changed')
    verify_initial_adapter(zero, config['weights_sha256'], config['adapter_config_sha256'])
    return config, directory, frozen, plan, windows, zero


def _verify_complete(directory, root, config, groups, marker=None):
    if marker is None: marker = json.loads((directory/'complete.json').read_text())
    if marker['config_sha256'] != _hash(config): raise ValueError('completed config changed')
    required = {'config.json', 'results/results.json', 'results/responses.jsonl', 'results/steps.jsonl'}
    if not required.issubset(marker['files']): raise ValueError('training result seal incomplete')
    for name, digest in marker['files'].items():
        if file_hash(_within(directory, directory/name)) != digest:
            raise ValueError('training result changed')
    checkpoints = _verified_checkpoints(directory)
    if sorted(checkpoints) != CHECKPOINTS: raise ValueError('scientific checkpoints incomplete')
    history, records, logs = _accepted_history(root, checkpoints[64], groups)
    if len(records) != 4096 or len(logs) != 64 or len({r['draw_id'] for r in records}) != 4096:
        raise ValueError('training history incomplete')
    for filename, expected in [('responses.jsonl', records), ('steps.jsonl', logs)]:
        actual = [json.loads(line) for line in (directory/'results'/filename).read_text().splitlines()]
        if actual != expected: raise ValueError('published results differ from sealed history')
    for step, checkpoint in checkpoints.items():
        if json.loads((checkpoint/'history.json').read_text()) != history[:step]:
            raise ValueError('checkpoint history mismatch')
    result = json.loads((directory/'results/results.json').read_text())
    if not result['base_integrity']['base_unchanged']: raise ValueError('base integrity failure')
    return result


def _accepted_history(root, checkpoint, groups):
    history, records, logs = _history(root, checkpoint, groups)
    for step, log in enumerate(logs, 1):
        evidence = log.get('diagnostic_evidence', {})
        required = {'logratio', 'window1_push'} if step == 1 else {'logratio'}
        if set(evidence) != required: raise ValueError('missing accepted diagnostic evidence')
        for reference in evidence.values():
            path = _within(root, reference['path'])
            if file_hash(path) != reference['sha256']:
                raise ValueError('accepted diagnostic evidence changed')
        if step == 1 and not json.loads(_within(root, evidence['window1_push']['path']).read_text()).get('passed'):
            raise ValueError('window-1 push gate not passed')
    return history, records, logs


def run(config_path, output_root, *, dependencies=None):
    root = Path(output_root).resolve()
    config, directory, frozen, plan, windows, zero = _load(config_path, root)
    groups = config['optimizer_groups']; engine = None
    with run_lock(directory):
        if (directory/'complete.json').exists():
            return _verify_complete(directory, root, config, groups)
        checkpoints = _verified_checkpoints(directory)
        if any(step not in CHECKPOINTS for step in checkpoints): raise ValueError('unexpected checkpoint')
        last = max(checkpoints, default=0)
        sealed_step = last
        history, records, logs = _accepted_history(root, checkpoints[last], groups) if checkpoints else ([], [], [])
        attempt_dir = directory/'attempts'; attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt = attempt_dir/f'attempt-{len(list(attempt_dir.glob("attempt-*"))):04d}'
        attempt.mkdir(); _write_json(attempt/'meta.json', {'resumed_from': last})
        try:
            _write_state(directory, 'running', last, 64)
            if dependencies is None:
                _verify_preflight(directory, config)
                pool_config = json.loads((root/config['pool_path']).read_text())
                receipt = json.loads((_within(root, pool_config['run_path'])/'handoff.json').read_text())
                if receipt['metadata']['execution_kind'] != 'real-cuda':
                    raise ValueError('real training requires a verified real-cuda pool')
                engine = OfflineGRPOTrainingEngine(plan, config['settings'], directory,
                    checkpoint_zero=zero, weights_sha256=config['weights_sha256'],
                    config_sha256=config['adapter_config_sha256'])
            else:
                engine = dependencies.load_training(plan, config, directory, zero)
            if engine.runtime() != config['runtime']: raise ValueError('runtime differs from reviewed plan')
            if dependencies is None and engine.initial_base != plan['source_evidence']['base_sha256']:
                raise ValueError('base weights differ from verified reference')
            if checkpoints: engine.restore_checkpoint(checkpoints[last])
            else:
                if not engine.check_integrity()['base_unchanged']: raise ValueError('base integrity failure')
                _seal(directory, 0, engine, [], groups)
            for step in range(last+1, 65):
                def diagnostic_sink(kind, diagnostic_step, value):
                    if diagnostic_step != step: raise ValueError('diagnostic step mismatch')
                    path = attempt/'diagnostics'/f'{step:03d}-{kind}.json'
                    _save_frozen(path, value)
                    return dict(path=str(path.relative_to(root)), sha256=file_hash(path))
                engine.diagnostic_sink = diagnostic_sink
                draws, details = engine.train_window(windows[step-1], step); _finite([draws, details])
                required = {'logratio', 'window1_push'} if step == 1 else {'logratio'}
                if set(details.get('diagnostic_evidence', {})) != required:
                    raise ValueError('missing pre-update diagnostic evidence')
                expected = [f'{q}:draw:{d}' for q in groups[step-1] for d in range(8)]
                if [r['draw_id'] for r in draws] != expected or any(r['step'] != step for r in draws):
                    raise ValueError('consumed draw identity mismatch')
                lengths = [r['token_count'] for r in draws]
                log = dict(step=step, **details, reward=sum(r['reward'] for r in draws)/64,
                    strict_accuracy=sum(r['strict']['correct'] for r in draws)/64,
                    flexible_accuracy=sum(r['flexible']['correct'] for r in draws)/64,
                    cap_fraction=sum(r['capped'] for r in draws)/64,
                    mean_tokens=sum(lengths)/64, p90_tokens=_percentile(lengths, 90),
                    dead_group_fraction=sum(len({r['reward'] for r in draws[i:i+8]}) == 1 for i in range(0,64,8))/8)
                log['monitor_events'] = _monitor_events(logs+[log], config['monitor_policy'], config['length_change_definition'])
                unit = attempt/f'step-{step:03d}.json'
                _write_json(unit, dict(step=step, prompt_ids=groups[step-1], responses=draws, log=log))
                history.append(dict(path=str(unit.relative_to(root)), sha256=file_hash(unit))); records.extend(draws); logs.append(log)
                if step in CHECKPOINTS:
                    if not engine.check_integrity()['base_unchanged']: raise ValueError('base integrity failure')
                    _seal(directory, step, engine, history, groups)
                    sealed_step = step
                _write_state(directory, 'running', step, 64)
                print(f'Offline GRPO step {step}/64; last sealed {sealed_step}; '
                      f'status {directory / "meta/status.json"}', flush=True)
            integrity = engine.check_integrity()
            if not integrity['base_unchanged']: raise ValueError('base integrity failure')
            _, records, logs = _accepted_history(root, directory/'checkpoints/checkpoint-64', groups)
            if len(records) != 4096 or len({r['draw_id'] for r in records}) != 4096:
                raise ValueError('training history incomplete')
            result = dict(optimizer_steps=64, accepted_completions=len(records), base_integrity=integrity)
            _write_json(directory/'results/results.json', result)
            for filename, values in [('responses.jsonl', records), ('steps.jsonl', logs)]:
                target=directory/'results'/filename; target.parent.mkdir(parents=True, exist_ok=True)
                temporary=target.with_suffix('.tmp'); temporary.write_text(''.join(json.dumps(v, sort_keys=True)+'\n' for v in values)); temporary.replace(target)
            files = {str(p.relative_to(directory)): file_hash(p) for p in (directory/'results').glob('*') if p.is_file()}
            files['config.json'] = file_hash(directory/'config.json')
            marker = dict(config_sha256=_hash(config), files=files)
            _verify_complete(directory, root, config, groups, marker)
            _write_json(directory/'complete.json', marker)
            _write_state(directory, 'completed', 64, 64)
            return result
        except BaseException as exc:
            _write_state(directory, 'failed', max(_verified_checkpoints(directory), default=0), 64, str(exc))
            raise
        finally:
            if engine is not None: engine.close()


def _verify_preflight(directory, config, *, marker=None):
    path = directory/'diagnostics/preflight.json'
    if marker is None:
        marker = json.loads((directory/'diagnostics/preflight.complete.json').read_text())
    if marker != {'config_sha256': _hash(config), 'sha256': file_hash(path)}:
        raise ValueError('diagnostic preflight seal changed')
    receipt = json.loads(path.read_text())
    _finite(receipt)
    if receipt.get('protocol') != 'offline-grpo-diagnostic-preflight-v1' or receipt.get('config_sha256') != _hash(config):
        raise ValueError('diagnostic preflight is not bound to the training plan')
    if receipt.get('runtime') != config['runtime'] or receipt.get('execution_kind') != 'real-cuda':
        raise ValueError('diagnostic preflight runtime mismatch')
    fidelity = receipt['fidelity']; observations = fidelity['observations']
    if (type(fidelity.get('long_min_tokens')) is not int or type(fidelity.get('max_new_tokens')) is not int
            or not 1 <= fidelity['long_min_tokens'] <= fidelity['max_new_tokens']
            or type(receipt.get('prompt_count')) is not int or len(observations) != receipt['prompt_count']
            or not 1 <= receipt['prompt_count'] <= 8):
        raise ValueError('invalid bounded fidelity observation criteria')
    if not observations or not any(x['eos'] for x in observations) or not any(
            x['capped'] or len(x['generated_token_ids']) >= fidelity['long_min_tokens'] for x in observations):
        raise ValueError('fidelity lacks EOS and long or capped observations')
    for item in observations:
        a=item['generation_log_probs']; b=item['teacher_forced_log_probs']; d=item['differences']
        if not len(a)==len(b)==len(d)==len(item['generated_token_ids']) or not a:
            raise ValueError('fidelity token alignment invalid')
        if any(not __import__('math').isfinite(x) for x in a+b+d) or any(x-y!=z for x,y,z in zip(a,b,d)):
            raise ValueError('fidelity differences invalid')
        if item['eos'] != (item['generated_token_ids'][-1] == fidelity['eos_token_id']):
            raise ValueError('fidelity EOS identity invalid')
        if (len(item['generated_token_ids']) > fidelity['max_new_tokens']
                or fidelity['eos_token_id'] in item['generated_token_ids'][:-1]
                or item['capped'] != (not item['eos'] and len(item['generated_token_ids']) == fidelity['max_new_tokens'])
                or (not item['eos'] and not item['capped'])):
            raise ValueError('fidelity cap identity invalid')
    if (receipt['smoke'].get('finite_gradients') is not True
            or receipt['smoke'].get('same_input_gradient_match') is not True
            or receipt['smoke'].get('optimizer_steps') != 0
            or receipt['smoke'].get('adapter_unchanged') is not True
            or receipt['smoke'].get('base_unchanged') is not True):
        raise ValueError('same-input smoke has not passed')
    return receipt


def check_diagnostics(config_path, output_root, *, long_min_tokens, prompt_count):
    """Explicit bounded real-GPU preflight; never steps the optimizer."""
    import time
    from pilot_eval.offline_grpo_engine import pool_window
    from pilot_eval.offline_grpo_diagnostics import measure_generation_scores, independent_push
    root=Path(output_root).resolve()
    config,directory,_,plan,windows,zero=_load(config_path,root)
    if type(prompt_count) is not int or not 1 <= prompt_count <= 8:
        raise ValueError('explicit prompt count between 1 and 8 required')
    with run_lock(directory):
        if (directory/'diagnostics/preflight.complete.json').exists():
            existing = _verify_preflight(directory,config)
            if existing['prompt_count'] != prompt_count or existing['fidelity']['long_min_tokens'] != long_min_tokens:
                raise ValueError('existing diagnostic preflight uses different bounds')
            return existing
        if _verified_checkpoints(directory):
            raise ValueError('diagnostic preflight must run before training starts')
        pool_config=json.loads((root/config['pool_path']).read_text())
        receipt=json.loads((_within(root,pool_config['run_path'])/'handoff.json').read_text())
        if receipt['metadata']['execution_kind'] != 'real-cuda':
            raise ValueError('real diagnostic preflight requires a verified real-cuda pool')
        engine=None
        try:
            engine=OfflineGRPOTrainingEngine(plan,config['settings'],directory,checkpoint_zero=zero,
                weights_sha256=config['weights_sha256'],config_sha256=config['adapter_config_sha256'])
            if engine.model.device.type != 'cuda' or engine.runtime()!=config['runtime']:
                raise ValueError('real GPU runtime differs from reviewed plan')
            if engine.initial_base != plan['source_evidence']['base_sha256']:
                raise ValueError('base weights differ from verified reference')
            t=engine.torch; t.cuda.reset_peak_memory_stats(); started=time.perf_counter()
            adapter_before = {n:p.detach().clone() for n,p in engine.model.named_parameters() if p.requires_grad}
            fidelity=measure_generation_scores(engine.model,[r['input_ids'] for r in windows[0][:prompt_count]],
                eos_token_id=engine.tokenizer.eos_token_id, pad_token_id=engine.tokenizer.pad_token_id,
                max_new_tokens=config['settings']['completion_limit'],long_min_tokens=long_min_tokens)
            prompts,outputs=pool_window(windows[0]); rewards=[d['reward_v3'] for r in windows[0] for d in r['draws']]
            capped=[d['capped'] for r in windows[0] for d in r['draws']]
            engine.optimizer.zero_grad(set_to_none=True)
            loss=engine.window.backward(prompts,[x['generated_token_ids'] for x in outputs],rewards,capped=capped)
            gradient=engine.window.flat_gradient()
            push=independent_push(engine.model,prompts,[x['generated_token_ids'] for x in outputs],
                                  engine.window.advantages,engine.window.parameter_names)
            t.testing.assert_close(gradient,-push,atol=1e-6,rtol=1e-4)
            engine.optimizer.zero_grad(set_to_none=True); engine._sync()
            if not engine.check_integrity()['base_unchanged']:raise ValueError('base changed during diagnostic preflight')
            if any(not t.equal(p,adapter_before[n]) for n,p in engine.model.named_parameters() if p.requires_grad):
                raise ValueError('adapter changed during diagnostic preflight')
            proof=dict(protocol='offline-grpo-diagnostic-preflight-v1',config_sha256=_hash(config),
                execution_kind='real-cuda',runtime=config['runtime'],prompt_count=prompt_count,fidelity=fidelity,
                smoke=dict(finite_gradients=True,same_input_gradient_match=True,optimizer_steps=0,loss=loss,
                    adapter_unchanged=True,base_unchanged=True,
                    gradient_norm=float(gradient.norm()),seconds=time.perf_counter()-started,
                    peak_memory_bytes=t.cuda.max_memory_allocated(),criterion=dict(atol=1e-6,rtol=1e-4)))
            _finite(proof); _save_frozen(directory/'diagnostics/preflight.json',proof)
            marker=dict(config_sha256=_hash(config),sha256=file_hash(directory/'diagnostics/preflight.json'))
            _verify_preflight(directory,config,marker=marker)
            _save_frozen(directory/'diagnostics/preflight.complete.json',marker)
            return proof
        finally:
            if engine is not None:engine.close()

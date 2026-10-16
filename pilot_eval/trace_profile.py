"""Bounded endpoint cost evidence. This workflow cannot freeze production."""
import json
import time
from pathlib import Path
import numpy as np
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json, _write_state
from pilot_eval.trace_corpus import _inside
from pilot_eval.trace_capture_workflow import _load as load_diagnostic, SITES
from pilot_eval.activation_workflow import _save_arrays
from pilot_eval.trace_capture import summarize_pairs, retention_evidence
from pilot_eval.sft import safe_name

CONDITIONS = ['base', 'sft-64', 'grpo-64', 'random-64']
BATCHES = [1, 4]


def implementation():
    folder = Path(__file__).parent
    return {name: file_hash(folder / name) for name in
            ['trace_profile.py', 'trace_profile_backend.py', 'trace_capture.py']}


def endpoint(path, root, arm):
    """Read sealed preparation and only the used endpoint, without trainer replay."""
    path = _inside(root, path)
    config = json.loads(path.read_text())
    kind = config.get('source_contract', {}).get('kind')
    if ((arm == 'sft' and config.get('protocol_version') != 'pilot3-write-v1')
            or (arm == 'grpo' and kind != 'grpo')
            or (arm == 'random' and kind != 'random-rank1-control')):
        raise ValueError('Wrong endpoint source kind.')
    marker_path = path.parent / 'prepare-complete.json'
    marker = json.loads(marker_path.read_text())
    if marker['config_sha256'] != _hash(config) or str(path.relative_to(root)) not in marker['files']:
        raise ValueError('Prepared endpoint seal mismatch.')
    parents = {str(marker_path.relative_to(root)): file_hash(marker_path)}
    for relative, expected in marker['files'].items():
        if file_hash(_inside(root, relative)) != expected:
            raise ValueError('Prepared endpoint payload hash mismatch.')
        parents[relative] = expected
    cp = config['source_evidence']['checkpoints']['64']
    directory = _inside(root, cp['path'])
    complete = directory / 'complete.json'
    if file_hash(complete) != cp['complete_sha256']:
        raise ValueError('Endpoint checkpoint seal hash mismatch.')
    seal = json.loads(complete.read_text())
    if arm != 'random' and seal.get('step') != 64:
        raise ValueError('Wrong endpoint checkpoint step.')
    required = {'adapter_config.json', 'adapter_model.safetensors'}
    if arm == 'random': required.add('construction.json')
    if not required <= set(seal['files']):
        raise ValueError('Endpoint seal omits required files.')
    for name in required:
        p = _inside(directory, name)
        if file_hash(p) != seal['files'][name]:
            raise ValueError('Endpoint payload hash mismatch.')
        parents[str(p.relative_to(root))] = file_hash(p)
    if file_hash(directory / 'adapter_model.safetensors') != cp['adapter_sha256']:
        raise ValueError('Endpoint source adapter hash mismatch.')
    parents[str(complete.relative_to(root))] = file_hash(complete)
    adapter = json.loads((directory / 'adapter_config.json').read_text())
    if (adapter.get('r') != 1 or adapter.get('lora_alpha') != 1 or adapter.get('lora_dropout') != 0
            or adapter.get('use_dora') or adapter.get('use_rslora') or adapter.get('bias', 'none') != 'none'
            or adapter.get('rank_pattern') or adapter.get('alpha_pattern')):
        raise ValueError('Profile requires plain rank-one alpha-one adapters.')
    return config, parents


def prepare_profile(diagnostic_path, random_path, output_root, name, *, model_config=None):
    """Freeze the approved 4-document/4-condition/2-batch diagnostic workload."""
    root = Path(output_root).resolve(); safe_name(name)
    path = root / 'plans' / name / 'traces.profile.json'
    with run_lock(path.parent):
        if path.exists():
            config = load_profile(path, root)
            if config['diagnostic_path'] != str(_inside(root, diagnostic_path).relative_to(root)) or config['random_path'] != str(_inside(root, random_path).relative_to(root)):
                raise ValueError('Profile selection changed; use a new name.')
            return path
        diagnostic = load_diagnostic(diagnostic_path, root)
        parents = dict(diagnostic['parents'])
        parents[str(_inside(root, diagnostic_path).relative_to(root))] = file_hash(diagnostic_path)
        source_paths = dict(sft=diagnostic['source_paths'][1], grpo=diagnostic['source_paths'][2], random=str(_inside(root, random_path).relative_to(root)))
        sources = {}; adapters = {}
        for arm, relative in source_paths.items():
            source, evidence = endpoint(relative, root, arm)
            parents.update(evidence); sources[arm] = source
            if any(source[k] != diagnostic[k] for k in ['model', 'model_revision', 'tokenizer_revision']):
                raise ValueError('Endpoint model/tokenizer identity differs from diagnostic.')
            if source['source_evidence']['base_sha256'] != diagnostic['expected_base_hash']:
                raise ValueError('Endpoint frozen base differs.')
            adapters[arm + '-64'] = source['source_evidence']['checkpoints']['64']['path']
        contract = sources['random']['source_contract']
        if _inside(root, contract['parent_prepared_path']) != _inside(root, source_paths['grpo']):
            raise ValueError('Random source has a different GRPO parent.')
        control = _inside(root, adapters['random-64'])
        if _inside(root, contract['control_path']) != control:
            raise ValueError('Random source path mismatch.')
        # Existing completion is mandatory. This call verifies; it cannot create a control.
        from pilot_eval.grpo_random_control import construct_control
        construct_control(_inside(root, adapters['grpo-64']), control)
        construction = json.loads((control / 'construction.json').read_text())
        if construction.get('seed') != 42 or construction.get('realizations') != 1:
            raise ValueError('Unexpected random construction seed or realization count.')
        if model_config is None:
            from huggingface_hub import hf_hub_download
            model_config = hf_hub_download(diagnostic['model'], 'config.json', revision=diagnostic['model_revision'])
        metadata = json.loads(Path(model_config).read_text())
        if metadata.get('num_hidden_layers') != 28 or type(metadata.get('hidden_size')) is not int or metadata['hidden_size'] < 1:
            raise ValueError('Pinned model metadata does not match the 29-site contract.')
        model_path = root / 'runs/pilot-5' / name / 'profile/inputs/model-config.json'
        _save_frozen(model_path, metadata)
        parents[str(model_path.relative_to(root))] = file_hash(model_path)
        width = metadata['hidden_size']
        config = dict(schema_version=1, mode='bounded-endpoint-profile', name=name,
            run_path=f'runs/pilot-5/{name}/profile', diagnostic_path=str(_inside(root, diagnostic_path).relative_to(root)),
            random_path=source_paths['random'], parents=parents, implementation=implementation(),
            examples=diagnostic['examples'], conditions=CONDITIONS, batch_sizes=BATCHES, sites=SITES,
            model=diagnostic['model'], model_revision=diagnostic['model_revision'], tokenizer_revision=diagnostic['tokenizer_revision'],
            expected_base_hash=diagnostic['expected_base_hash'], adapters=adapters, width=width, seed=42,
            capture_precision='float32', accumulation='float64', random_construction=construction,
            approval='Prasad: same four documents; base/SFT64/GRPO64/existing-random64; batches 1 and 4',
            workload=dict(sequence_forwards=32, model_calls=20, units=8),
            uncompressed_profile_capture_bytes=32*29*128*width*4,
            storage_note='Raw FP32 profile arrays retained for CPU diagnostics; distinct from production retention. Compression is measured, not assumed.',
            production_ready=False, acceptance='No production freeze, numerical gate, decoded-token reveal or endpoint-only scope approval.')
        _save_frozen(path, config)
        directory = root / config['run_path']
        _save_frozen(directory / 'meta/run_manifest.json', dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config)))
        _write_state(directory, 'prepared', 0, 8)
        return path


def load_profile(path, root):
    path = _inside(root, path); config = json.loads(path.read_text())
    if config['mode'] != 'bounded-endpoint-profile' or config['implementation'] != implementation():
        raise ValueError('Profile implementation changed; preserve this run and use a new name.')
    manifest = json.loads((_inside(root, config['run_path']) / 'meta/run_manifest.json').read_text())
    if manifest != dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config)):
        raise ValueError('Profile manifest identity mismatch.')
    for relative, expected in config['parents'].items():
        if file_hash(_inside(root, relative)) != expected:
            raise ValueError('Profile parent hash mismatch.')
    return config


def units():
    return [(batch, condition, f'b{batch}-{condition}') for batch in BATCHES for condition in CONDITIONS]


def read_unit(directory, key, identity):
    folder = directory / 'units' / key
    if not (folder / 'complete.json').exists(): return None
    marker = json.loads((folder / 'complete.json').read_text())
    if marker['identity'] != identity or set(marker['files']) != {'arrays.npz', 'evidence.json'}:
        raise ValueError('Profile unit seal identity mismatch.')
    for name, expected in marker['files'].items():
        if file_hash(folder / name) != expected: raise ValueError('Profile unit payload hash mismatch.')
    return json.loads((folder / 'evidence.json').read_text())


def verify_profile(path, output_root):
    root = Path(output_root).resolve(); config = load_profile(path, root); directory = root / config['run_path']
    marker = json.loads((directory / 'complete.json').read_text())
    if marker['config_sha256'] != _hash(config) or set(marker['files']) != {'results/results.json', 'meta/runtime.json', 'meta/run_manifest.json'}:
        raise ValueError('Profile completion identity mismatch.')
    for relative, expected in marker['files'].items():
        if file_hash(directory / relative) != expected: raise ValueError('Profile completion hash mismatch.')
    runtime = json.loads((directory / 'meta/runtime.json').read_text())
    for _, _, key in units():
        if read_unit(directory, key, dict(config_sha256=_hash(config), runtime=runtime, unit=key)) is None:
            raise ValueError('Missing profile unit.')
    return json.loads((directory / 'results/results.json').read_text())


def _arrays(directory, key):
    with np.load(directory / 'units' / key / 'arrays.npz', allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def run_profile(path, output_root, *, dependencies=None):
    root = Path(output_root).resolve(); config = load_profile(path, root); directory = root / config['run_path']
    with run_lock(directory):
        if (directory / 'complete.json').exists(): return verify_profile(path, root)
        from pilot_eval.trace_profile_backend import ProfileDependencies
        deps = dependencies or ProfileDependencies(); engine = None; completed = 0
        try:
            runtime = deps.runtime(); _save_frozen(directory / 'meta/runtime.json', runtime)
            evidence = {}
            for _, _, key in units():
                existing = read_unit(directory, key, dict(config_sha256=_hash(config), runtime=runtime, unit=key))
                if existing: evidence[key] = existing
            completed = len(evidence); reused = completed
            print(f'Profile: {completed}/8 saved units; approved total 32 sequence forwards, 20 model calls.', flush=True)
            loading = 0.
            if completed < 8:
                started = time.monotonic(); engine = deps.engine(config, root); loading = time.monotonic() - started
            for batch, condition, key in units():
                if key in evidence: continue
                _write_state(directory, 'running', completed, 8)
                started = time.monotonic(); before = engine.integrity(); hash_seconds = time.monotonic() - started
                if before != config['expected_base_hash']: raise ValueError('Frozen base hash mismatch before profile unit.')
                arrays = {}; forward = time.monotonic()
                for offset in range(0, 4, batch):
                    rows = config['examples'][offset:offset+batch]
                    captured = engine.capture(rows, condition)
                    if set(captured) != set(SITES): raise ValueError('Missing profile capture site.')
                    for site in SITES:
                        values = np.asarray(captured[site])
                        if values.dtype != np.float32 or values.shape != (len(rows), 128, config['width']) or not np.isfinite(values).all():
                            raise ValueError(f'Invalid profile activation: {key}/{site}')
                        for index, value in enumerate(values): arrays[f'{offset+index}.{site}'] = value
                    print(f'Profile {key}: {offset+len(rows)}/4 documents captured.', flush=True)
                forward_seconds = time.monotonic() - forward
                started = time.monotonic(); after = engine.integrity(); hash_seconds += time.monotonic() - started
                if after != before: raise ValueError('Frozen base changed during profile.')
                base = arrays if condition == 'base' else _arrays(directory, f'b{batch}-base')
                started = time.monotonic()
                pairs = [dict(example_ids=[r['id']], content_positions=[r['content_positions']],
                    base={s:base[f'{i}.{s}'][None] for s in SITES},
                    adapted={s:arrays[f'{i}.{s}'][None] for s in SITES}) for i,r in enumerate(config['examples'])]
                summary = summarize_pairs(pairs, sites=SITES)
                reduction_seconds = time.monotonic() - started
                conversion = {site:retention_evidence(v,storage_dtype='float16') for site,v in summary['retained_delta'].items()}
                folder = directory / 'units' / key; started = time.monotonic()
                _save_arrays(folder / 'arrays.npz', arrays); save_seconds = time.monotonic() - started
                evidence[key] = dict(condition=condition,batch_size=batch,sequence_forwards=4,model_calls=4//batch,
                    forward_capture_transfer_seconds=forward_seconds,hash_seconds=hash_seconds,
                    reduction_moments_seconds=reduction_seconds,array_save_seconds=save_seconds,
                    durable_bytes=(folder/'arrays.npz').stat().st_size,memory=engine.memory(),
                    fp16_retention=conversion,qualified=False,
                    timing_note='Includes CPU hook transfer; no separate transfer timing claim; no production ETA extrapolation.')
                _write_json(folder/'evidence.json', evidence[key])
                _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),runtime=runtime,unit=key),
                    files={n:file_hash(folder/n) for n in ['arrays.npz','evidence.json']}))
                completed += 1
                print(f'Profile: {completed}/8 units sealed ({key}).',flush=True)
            if engine: engine.close(); engine = None
            comparisons = {}
            for condition in CONDITIONS:
                one = _arrays(directory, f'b1-{condition}'); four = _arrays(directory,f'b4-{condition}')
                worst = max(one, key=lambda key:float(np.max(np.abs(one[key].astype(np.float64)-four[key]))))
                delta = np.abs(one[worst].astype(np.float64)-four[worst]); coord = np.unravel_index(int(delta.argmax()), delta.shape)
                comparisons[condition] = dict(max_absolute_difference=float(delta.max()),example_site=worst,
                    coordinate=[int(x) for x in coord],singleton=float(one[worst][coord]),batched=float(four[worst][coord]),allowance=None,qualified=False)
            result = dict(profile_complete=True,production_ready=False,qualified=False,completed_units=8,
                sequence_forwards=32,model_calls=20,newly_sealed_units=8-reused,reused_units=reused,
                this_attempt_loading_seconds=loading,measurements=list(evidence.values()),batching_diagnostics=comparisons,
                durable_capture_bytes=sum(r['durable_bytes'] for r in evidence.values()),
                pending=['capture numerical gates','nonzero storage/ranking/bootstrap qualification','community lens disposition',
                         'production endpoint versus trajectory scope','reviewed budget and predictions'],
                note='Four fixed documents only; no task-token readout or production capture. Timings are diagnostic observations.')
            _write_json(directory/'results/results.json',result)
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in
                ['results/results.json','meta/runtime.json','meta/run_manifest.json']}))
            _write_state(directory,'awaiting-review',8,8)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',completed,8,str(exc)); raise
        finally:
            if engine: engine.close()

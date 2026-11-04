"""Bounded, resumable fresh R0 collection for an offline GRPO pool.

This additive protocol reuses the existing local R0 sampler and budget ledger.
No model is loaded during preparation. Real collection requires a session-bound
budget approval and the exact frozen runtime.
"""
import json
import time
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.offline_grpo_support import IntegrityFailure, attempt_scope, attempt_start, pinned_hash
from pilot_eval.stage0_evaluation import _jsonl, _reservation
from pilot_eval.stage0_rollout_math import normalize
from pilot_eval.stage0_rollout_plan import implementation as rollout_implementation
from pilot_eval.stage0_rollout_sources import POLICY
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _state, _writer, status
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval import offline_grpo_pool


def prepare(pool_path, audit_path, weights_dir, runtime_path, output_root, *, dependencies=None):
    from pilot_eval.stage0_rollout_backend import RolloutDependencies
    root = Path(output_root).resolve()
    pool_path = inside(root, pool_path)
    pool, directory, training = offline_grpo_pool._plan(pool_path, root)
    audit_path = inside(root, audit_path)
    audit = status(audit_path, root)['audit']
    if not audit or not audit['primary_sources_ready']:
        raise ValueError('Verified common model source audit required')
    source = audit['sources']['sft']
    identity = source['identity']
    if any(identity[k] != pool['source_identity'][k] for k in pool['source_identity']):
        raise ValueError('Audited model identity differs from frozen training source')
    runtime_path = inside(root, runtime_path)
    approved_runtime = json.loads(runtime_path.read_text())
    if not isinstance(approved_runtime, dict) or not approved_runtime.get('session_id'):
        raise ValueError('Explicit session-bound runtime required')
    weights = inside(root, weights_dir)
    model_config_path = weights/'config.json'
    expected_eos = json.loads(model_config_path.read_text()).get('eos_token_id')
    if type(expected_eos) is not int or expected_eos < 0:
        raise ValueError('Verified model config requires a scalar EOS token identity')
    deps = dependencies or RolloutDependencies()
    inventory = deps.inventory(weights, source)
    if inventory['base_sha256'] != identity['base_sha256']:
        raise ValueError('Weight identity differs from audited base')
    parents = {str(inside(root, p).relative_to(root)): digest for p, digest in inventory['parents'].items()}
    for filename in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        path = weights / filename
        parents[str(path.relative_to(root))] = file_hash(path)
    fresh = dict(protocol='offline-grpo-fresh-r0-v1', pool_path=str(pool_path.relative_to(root)),
                 pool_sha256=file_hash(pool_path), audit_path=str(audit_path.relative_to(root)), audit_sha256=file_hash(audit_path),
                 weights_dir=str(weights.relative_to(root)), model_identity=identity, parents=parents,
                 runtime_path=str(runtime_path.relative_to(root)), runtime_sha256=file_hash(runtime_path), runtime=approved_runtime,
                 implementation_sha256=file_hash(Path(__file__)), support_sha256=file_hash(Path(attempt_scope.__code__.co_filename)), sampler_implementation=rollout_implementation(),
                 expected_eos_token_id=expected_eos,
                 execution_kind='real-cuda' if inventory['execution_kind'] != 'synthetic-cpu-fixture' else 'synthetic-cpu-fixture',
                 run_path=str((directory/'fresh').relative_to(root)), generation_settings={**POLICY,'seed':42},
                 units=[dict(key=f'group-{n:04d}',id=identifier,seed=42+n,origin='new-uniform') for n,identifier in enumerate(pool['optimizer_order'])])
    path = directory/'fresh.plan.json'
    with _writer(directory):
        _save_frozen(path, fresh)
    return path


def _load(path, root):
    path=inside(root,path)
    config=json.loads(path.read_text())
    if config.get('protocol')!='offline-grpo-fresh-r0-v1' or config['implementation_sha256']!=file_hash(Path(__file__)) or config['support_sha256']!=file_hash(Path(attempt_scope.__code__.co_filename)):
        raise ValueError('Fresh collection implementation or protocol changed')
    if config['sampler_implementation']!=rollout_implementation():
        raise ValueError('Pinned R0 sampler or scorer changed')
    pool_path=inside(root,config['pool_path'])
    if file_hash(pool_path)!=config['pool_sha256'] or file_hash(inside(root,config['audit_path']))!=config['audit_sha256']:
        raise ValueError('Fresh source plan or audit changed')
    pool,_,training=offline_grpo_pool._plan(pool_path,root)
    audit=status(inside(root,config['audit_path']),root)['audit']
    if not audit or not audit['primary_sources_ready'] or audit['sources']['sft']['identity']!=config['model_identity']:
        raise ValueError('Fresh model source audit changed')
    if file_hash(inside(root,config['runtime_path']))!=config['runtime_sha256'] or json.loads(inside(root,config['runtime_path']).read_text())!=config['runtime']:
        raise ValueError('Approved runtime changed')
    for path_,digest in config['parents'].items():
        if pinned_hash(inside(root,path_))!=digest:raise ValueError('Pinned weight/tokenizer payload changed')
    expected=[dict(key=f'group-{n:04d}',id=i,seed=42+n,origin='new-uniform') for n,i in enumerate(pool['optimizer_order'])]
    if config['units']!=expected or config['generation_settings']!={**POLICY,'seed':42}:
        raise ValueError('Fresh group order or sampling settings changed')
    return config,inside(root,config['run_path']),{r['id']:r for r in training}


def _saved(config, directory, runtime, specials, rows):
    result=[]
    verified_attempts={}
    for unit in config['units']:
        folder=directory/'units'/unit['key']; marker=folder/'complete.json'
        if not marker.exists():
            result.append(None);continue
        seal=json.loads(marker.read_text())
        if seal['identity']!=dict(config_sha256=_hash(config),unit=unit,runtime_sha256=_hash(runtime),specials=specials):
            raise ValueError('Fresh group identity changed')
        for name,digest in seal['files'].items():
            if file_hash(inside(folder,name))!=digest:raise ValueError('Fresh group payload changed')
        if set(seal['files'])!={'responses.jsonl','metadata.json'}:raise ValueError('Incomplete group seal')
        meta=json.loads((folder/'metadata.json').read_text())
        if meta['integrity'].get('base_unchanged') is not True:raise ValueError('Missing base integrity proof')
        attempt=inside(directory/'attempts',meta['attempt_id'])
        if meta['attempt_id'] not in verified_attempts:
            verified_attempts[meta['attempt_id']]=attempt_start(attempt,['base_unchanged'])
        if verified_attempts[meta['attempt_id']]!=meta['attempt_start_sha256']:
            raise ValueError('Attempt identity changed')
        records=[json.loads(line) for line in (folder/'responses.jsonl').read_text().splitlines()]
        expected=[normalize(rows[unit['id']],r,n,origin=unit['origin'],seed=unit['seed'],limit=1024,eos=specials['eos_token_id']) for n,r in enumerate(records)]
        if len(records)!=8 or records!=expected:raise ValueError('Fresh group failed reconstruction')
        result.append(records)
    return result


def run(path, output_root, *, approval_path=None, max_new_units=None, dependencies=None, reservation_provider=None):
    from pilot_eval.stage0_rollout_backend import RolloutDependencies
    root=Path(output_root).resolve(); config,directory,rows=_load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid group bound')
    deps=dependencies or RolloutDependencies()
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        runtime=deps.runtime()
        if runtime['execution_kind']!=config['execution_kind'] or runtime!=config['runtime']:
            raise ValueError('Actual runtime differs from the explicitly approved runtime')
        runtime_path=directory/'meta/runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text())!=runtime:raise ValueError('Resume runtime changed')
        # The existing ledger supplies the explicit session and wall-time bound.
        execution={'reservation':{'reservation_id':None,'job_sha256':_hash(config)}}
        reservation,remaining=(reservation_provider or _reservation)(config,execution,directory,root,runtime,approval_path)
        saved=[]
        def failed(exc):_state(directory,'failed',sum(x is not None for x in saved),512,error=str(exc))
        with attempt_scope(directory,reservation,runtime,remaining,real=config['execution_kind']=='real-cuda',on_failure=failed) as run:
            attempt=run.path
            if hasattr(deps,'preflight'):deps.preflight(config,root)
            _save_frozen(runtime_path,runtime)
            engine=run.engine=deps.engine(config,root)
            specials=engine.special_tokens()
            if any(type(specials.get(k)) is not int for k in ['eos_token_id','pad_token_id']):raise ValueError('Missing tokenizer special-token identities')
            if specials['eos_token_id'] != config['expected_eos_token_id']:
                raise ValueError('Tokenizer EOS differs from the pinned model config')
            _save_frozen(directory/'inputs/special-tokens.json',specials)
            prompts=engine.prepare_inputs([rows[u['id']] for u in config['units']])
            expected_prompts=[dict(id=u['id'],input_ids=rows[u['id']]['input_ids'][:rows[u['id']]['prompt_tokens']]) for u in config['units']]
            if prompts!=expected_prompts:raise ValueError('Sampler prompt token identity changed')
            _save_frozen(directory/'inputs/tokenized-prompts.json',prompts)
            saved=_saved(config,directory,runtime,specials,rows)
            for index,unit in enumerate(config['units']):
                if saved[index] is not None:continue
                if max_new_units is not None and run.new>=max_new_units:break
                if time.monotonic()-run.started>=remaining:raise TimeoutError('Reserved sampling time exhausted')
                if engine.check_integrity().get('base_unchanged') is not True:raise IntegrityFailure('Base integrity failed before group')
                values=engine.sample([rows[unit['id']]],config['generation_settings'],unit['seed'])
                if len(values)!=8:raise ValueError('Expected eight draws')
                records=[normalize(rows[unit['id']],v,n,origin=unit['origin'],seed=unit['seed'],limit=1024,eos=specials['eos_token_id']) for n,v in enumerate(values)]
                integrity=engine.check_integrity()
                if integrity.get('base_unchanged') is not True:raise IntegrityFailure('Base integrity failed after group')
                folder=directory/'units'/unit['key']
                _jsonl(folder/'responses.jsonl',records)
                _write_json(folder/'metadata.json',dict(attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),integrity=integrity))
                _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),unit=unit,runtime_sha256=_hash(runtime),specials=specials),files={n:file_hash(folder/n) for n in ['responses.jsonl','metadata.json']}))
                saved[index]=records;run.new+=1
                _state(directory,'running',sum(x is not None for x in saved),512,reservation_id=reservation['reservation_id'])
                print(f"Fresh R0: {sum(x is not None for x in saved)}/512 groups; {unit['id']}",flush=True)
            integrity=engine.check_integrity(full=True)
            _write_json(attempt/'final-integrity.json',integrity)
            if integrity.get('base_unchanged') is not True:raise IntegrityFailure('Full base integrity failed')
            if any(group is None for group in saved):return _state(directory,'incomplete',sum(x is not None for x in saved),512)
            _load(path,root)
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),runtime_sha256=_hash(runtime),specials_sha256=_hash(specials),final_integrity_sha256=file_hash(attempt/'final-integrity.json'),final_attempt=attempt.name))
            _state(directory,'complete',512,512)
            return dict(complete=True,execution_kind=config['execution_kind'],groups=512)


def _verified(path,root):
    config,directory,rows=_load(path,root)
    marker=json.loads((directory/'complete.json').read_text())
    runtime=json.loads((directory/'meta/runtime.json').read_text());specials=json.loads((directory/'inputs/special-tokens.json').read_text())
    if runtime != config['runtime'] or specials['eos_token_id'] != config['expected_eos_token_id']:
        raise ValueError('Saved runtime or EOS differs from the approved identity')
    final=inside(directory/'attempts',marker['final_attempt'])/'final-integrity.json'
    if marker['config_sha256']!=_hash(config) or marker['runtime_sha256']!=_hash(runtime) or marker['specials_sha256']!=_hash(specials) or file_hash(final)!=marker['final_integrity_sha256'] or json.loads(final.read_text()).get('base_unchanged') is not True:
        raise ValueError('Fresh completion or final base integrity changed')
    expected_prompts=[dict(id=u['id'],input_ids=rows[u['id']]['input_ids'][:rows[u['id']]['prompt_tokens']]) for u in config['units']]
    if json.loads((directory/'inputs/tokenized-prompts.json').read_text())!=expected_prompts:
        raise ValueError('Saved prompt token identity changed')
    saved=_saved(config,directory,runtime,specials,rows)
    if any(group is None for group in saved):raise ValueError('Fresh collection incomplete')
    return config,saved,specials


def verify(path,output_root):
    config,_,_=_verified(path,Path(output_root).resolve())
    return dict(complete=True,execution_kind=config['execution_kind'],groups=512)


def verified_records(path,root):
    config,saved,specials=_verified(path,root)
    records=[r for group in saved for r in group]
    identity=config['model_identity']
    return records,dict(eos_token_id=specials['eos_token_id'],route='uniform-fresh-512',execution_kind=config['execution_kind'],model_identity={k:identity[k] for k in ['model','model_revision','tokenizer_revision','dtype']})

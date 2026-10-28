"""Public CPU source/plan workflow for Sampling SFT. No paid execution exists here."""
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from pilot_eval.run import _write_state
from pilot_eval.sampling_sft_sources import (gold_source, grpo_source, inside, observe,
                                             read, verify_parents)
from pilot_eval.sft import safe_name
from pilot_eval.stage0_workflow import _writer, _software
from pilot_eval.trace_session import _save
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash


PROPOSER_TEMPLATE = """You are given a grade-school math word problem, a reference solution, and the beginning
of a response. Study the reference solution, then continue the response in your own words,
showing your reasoning. Keep it consistent with the reference solution, and make sure the
final answer equals the reference solution's final answer.

Task:
{TRAINING_USER_MESSAGE}

Reference solution:
{CLEANED_GOLD}

Beginning of the response:
{PARTIAL_RESPONSE_TEXT}

Continue the response from exactly where it stops."""


def recipe():
    """Approved scientific contract; unresolved engine controls are never defaulted."""
    return dict(version='sampling-sft-v1', model='Qwen/Qwen2.5-1.5B-Instruct',
        k_values=[0,2,5], train_arms=[0,5], data_seed=0, training_seed=42,
        sampling_gpu='L40S', training_gpu='L4', dtype='float32', tf32=False,
        jump_tokens=64, cap_tokens=1024, max_blocks=32, max_model_retries=3,
        temperature=0.6, top_p=1.0, repetition_penalty=1.1,
        accept_rule='strict-mean-response-logprob-improvement', cut_range='whole-trace',
        prompts=dict(version='sampling-sft-proposer-v1', scorer_template='gsm8k-v1',
            proposer_template=PROPOSER_TEMPLATE, template_sha256=_hash(PROPOSER_TEMPLATE),
            prefix_policy='decoded-in-user-and-token-ids-as-assistant-prefill',
            gold_cleaning='strip-calculator-markup-and-rewrite-final-marker-as-prose-v1'),
        stop_token_strings=['<|im_end|>','<|endoftext|>'],
        baseline='deferred', offset='not-measured', evaluation='deferred', internals='deferred',
        checkpoint_steps=[0,8,16,32,64], budget_usd=dict(total=25,sampling=10,training=6,storage_shared=9),
        required_approvals=['first-paid-job-after-preflight','training-after-gate'],
        unresolved_engine_fields=['vllm_version','top_k','min_p','presence_penalty','frequency_penalty',
            'worker_numeric_flags','max_num_seqs','max_num_batched_tokens','gpu_memory_utilization',
            'prefix_caching','max_model_len','driver','cuda','torch_version','actual_stop_ids'],
        engine_policy='resolve explicitly during bounded engine qualification; no implicit defaults')


def implementation():
    folder = Path(__file__).parent
    return {name:file_hash(folder/name) for name in ['sampling_sft.py','sampling_sft_sources.py',
        'sft.py','protocol.py','stage0_sources.py','trace_readout_assets.py']}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(path, value)


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Sampling SFT frozen artifact mismatch: {path}; use a new plan name.')
    else:
        save(path, value)


def prepare(selection_path, output_root, name):
    safe_name(name)
    root = Path(output_root).resolve()
    selection = json.loads(Path(selection_path).read_text())
    if not isinstance(selection, dict) or set(selection) != {'sft','grpo'}:
        raise ValueError('Select explicit sft training/prepared config and grpo prepared config paths.')
    parents, sources = {}, {}
    for label, value in selection.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Each selected source must be one nonempty path.')
        path = observe(root, value, parents)
        sources[label] = str(path.relative_to(root))
    config = dict(schema_version=1, protocol='sampling-sft-source-plan-v1', name=name,
        run_path=f'runs/sampling-sft/{name}', sources=sources, parents=parents,
        recipe=recipe(), algorithm_hash=_hash(recipe()), implementation=implementation(),
        selection_sha256=_hash(sources), execution_kind='cpu-source-audit-only')
    path = root/'plans'/name/'sampling-sft.plan.json'
    directory = root/config['run_path']
    with run_lock(directory):
        frozen(path, config)
        frozen(directory/'meta/run_manifest.json', dict(config_path=str(path.relative_to(root)),
            config_sha256=_hash(config), config_file_sha256=file_hash(path), execution_kind='cpu-source-audit-only'))
        frozen(directory/'inputs/selection.json',sources)
        if not (directory/'meta/software.json').exists():
            frozen(directory/'meta/software.json',_software())
        if not (directory/'meta/status.json').exists():
            _write_state(directory,'prepared',0,2)
    return path


def load(path, output_root):
    root = Path(output_root).resolve()
    path = inside(root, path)
    config = json.loads(path.read_text())
    if (config.get('protocol') != 'sampling-sft-source-plan-v1'
            or config.get('recipe') != recipe() or config.get('algorithm_hash') != _hash(recipe())
            or config.get('implementation') != implementation()):
        raise ValueError('Sampling SFT plan/implementation changed; preserve it and use a new name.')
    safe_name(config['name'])
    if config.get('run_path') != f'runs/sampling-sft/{config["name"]}':
        raise ValueError('Sampling SFT run identity changed.')
    directory = inside(root, config['run_path'])
    manifest = json.loads((directory/'meta/run_manifest.json').read_text())
    if manifest != dict(config_path=str(path.relative_to(root)),config_sha256=_hash(config),config_file_sha256=file_hash(path),execution_kind='cpu-source-audit-only'):
        raise ValueError('Sampling SFT frozen plan/manifest mismatch.')
    verify_parents(root,config['parents'])
    if json.loads((directory/'inputs/selection.json').read_text()) != config['sources']:
        raise ValueError('Sampling SFT source selection changed.')
    return root,config,directory


def audit(path, output_root):
    root, config, directory = load(path, output_root)
    with _writer(directory):
        if (directory/'complete.json').exists():
            return status(path,output_root)['audit']
        parents = dict(config['parents'])
        blockers, selected = [], {}
        _write_state(directory,'running',0,2)
        for i,(label,reader) in enumerate([('sft',gold_source),('grpo',grpo_source)],1):
            try:
                selected[label] = reader(root,config['sources'][label],parents)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                blockers.append(dict(source=label,type=type(exc).__name__,reason=str(exc)))
            print(f'Sampling SFT audit: {i}/2 sources checked; {label}.',flush=True)
            _write_state(directory,'running',i,2)
        gold,grpo = selected.get('sft'),selected.get('grpo')
        if gold and grpo:
            expected={**gold['identity'],'base_sha256':gold['base_sha256']}
            if grpo['identity'] != expected:
                blockers.append(dict(source='paired-identities',reason='Gold/GRPO model, tokenizer or declared base hash differs.'))
        verify_parents(root,parents)
        result=dict(schema_version=1,state='blocked' if blockers else 'source-audited',
            source_ready=not blockers, blockers=blockers, recipe=config['recipe'],
            training=gold, grpo_reference=grpo, parents=parents,
            algorithm_hash=config['algorithm_hash'], runtime_hash=None,
            gpu_qualified=False, executed=False, training_approved=False,
            execution_kind='cpu-source-audit-only',
            evidence_scope='saved cohort/masks/config/order and adapter payloads; no model/tokenizer forwards, no base rehash or trainer replay',
            pending=['engine controls and stop IDs','prompt retokenization qualification',
                     'L40S worker flags, score/replay/throughput qualification','budget reservation and first-spend approval'])
        save(directory/'results/source-audit.json',result)
        _write_state(directory,result['state'],2,2)
        save(directory/'complete.json',dict(config_sha256=_hash(config),
            files={'results/source-audit.json':file_hash(directory/'results/source-audit.json')}))
        return result


def status(path, output_root):
    root, config, directory = load(path, output_root)
    value=dict(plan=str(inside(root,path).relative_to(root)),recipe=config['recipe'],
               state=json.loads((directory/'meta/status.json').read_text()),audit=None)
    marker=directory/'complete.json'
    if marker.exists():
        expected=dict(config_sha256=_hash(config),files={'results/source-audit.json':file_hash(directory/'results/source-audit.json')})
        if json.loads(marker.read_text()) != expected:
            raise ValueError('Sampling SFT audit completion seal mismatch.')
        result=json.loads((directory/'results/source-audit.json').read_text())
        verify_parents(root,result['parents'])
        value['audit']=result
    return value


def verify(path, output_root):
    result=status(path,output_root)
    if result['audit'] is None:
        raise ValueError('Sampling SFT audit not completed.')
    return result


def recover_writer(path, output_root, owner_token, review_notes):
    """Explicit reviewed ownership recovery; never automatically clear a writer."""
    _, _, directory = load(path, output_root)
    if not owner_token or not review_notes.strip():
        raise ValueError('Recovery requires the recorded owner token and review notes.')
    with run_lock(directory):
        marker = directory/'writer.json'
        owner = json.loads(marker.read_text())
        if owner.get('token') != owner_token:
            raise ValueError('Sampling SFT writer token mismatch.')
        if owner.get('host') == platform.node():
            pid = owner.get('pid')
            if not isinstance(pid, int) or pid <= 0:
                raise ValueError('Invalid writer process identity.')
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('The recorded writer process is still active.')
        event = dict(owner=owner, review_notes=review_notes,
                     updated_at=datetime.now(timezone.utc).isoformat())
        log = directory/'logs/ownership-recovery.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            stream.write(json.dumps(event)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        marker.unlink()
    return dict(state='ownership-released', scope='No source, plan or scientific payload changed.')


def console_summary(value):
    audit=value.get('audit',value)
    if not isinstance(audit,dict) or 'source_ready' not in audit:
        return value
    return dict(state=audit['state'],source_ready=audit['source_ready'],blockers=audit['blockers'],
                gpu_qualified=audit['gpu_qualified'],executed=audit['executed'],
                pending=audit['pending'], detail='Full source inventory and resolved plan saved in canonical artifacts.')

"""Verified GRPO source contract for the existing fixed-token instrument."""
import json
from pathlib import Path

from pilot_eval.activation_prepare import load_prepared, STEPS, PROJECTIONS, _preparation_status
from pilot_eval.grpo_evaluation import _source
from pilot_eval.grpo_prepare import _within
from pilot_eval.grpo_training import verify_training
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen


def source_evidence(contract, root):
    """Reverify the real GRPO history and checkpoint inventory without a model."""
    path = _within(root, contract['training_path'])
    training, frozen, plan, items, checkpoints = _source(path, root)
    original_path = _within(root, contract['measurement_path'])
    original, rows = load_prepared(original_path, root)
    if (original.get('protocol_version') != 'pilot3-write-v1'
            or plan['measurement_items_sha256'] != original['items_sha256']
            or plan['sources']['measurement']['sha256'] != file_hash(original_path)
            or _within(root, plan['sources']['measurement']['path']) != original_path
            or any(training[k] != original[k] for k in ['model','model_revision','tokenizer_revision','dtype','seed'])
            or training['prompt_template']['chat_template_sha256'] != original['chat_template_sha256']
            or [(r['id'],r['gold']) for r in rows if r['corpus']=='gsm8k'] != [(r['id'],r['gold']) for r in items]):
        raise ValueError('GRPO and fixed measurement source identities differ')
    integrity = verify_training(path, root)['base_integrity']
    base = original['source_evidence']['base_sha256']
    if integrity != dict(base_before=base,base_after=base,base_unchanged=True):
        raise ValueError('GRPO frozen-base hash differs from fixed measurement source')
    evidence = dict(base_sha256=base, source_config_sha256=_hash(training), checkpoints={},
                    files={str(path.relative_to(root)):file_hash(path),
                           training['run_path']+'/complete.json':file_hash(root/training['run_path']/'complete.json')})
    for step, checkpoint in sorted(checkpoints.items()):
        adapter = json.loads((checkpoint/'adapter_config.json').read_text())
        full = {f"model.layers.{i}.{'self_attn' if p in PROJECTIONS[:4] else 'mlp'}.{p}"
                for i in range(28) for p in PROJECTIONS}
        targets = adapter.get('target_modules')
        if (adapter.get('r') != 1 or adapter.get('lora_alpha') != 1 or adapter.get('lora_dropout') != 0
                or (targets != 'all-linear' and set(targets or []) not in (set(PROJECTIONS),full))):
            raise ValueError('GRPO adapter is not rank-1 on the required projections')
        evidence['checkpoints'][str(step)] = dict(path=str(checkpoint.relative_to(root)),
            complete_sha256=file_hash(checkpoint/'complete.json'),
            adapter_sha256=file_hash(checkpoint/'adapter_model.safetensors'))
    return evidence, original, rows


def prepare_grpo_writes(training_path, measurement_path, output_root, name):
    root = Path(output_root).resolve()
    name = safe_name(name)
    contract = dict(kind='grpo',training_path=str(_within(root,training_path).relative_to(root)),
                    measurement_path=str(_within(root,measurement_path).relative_to(root)))
    evidence, original, rows = source_evidence(contract, root)
    plan = root/'plans'/name
    path = plan/'activation.prepared.json'
    with run_lock(plan), _preparation_status(plan):
        if (plan/'prepare-complete.json').exists():
            config,_ = load_prepared(path,root)
            if config['source_contract'] != contract: raise ValueError('GRPO measurement plan differs; use a new name')
            return path
        run = root/'runs/pilot-4'/original['model'].replace('/','--')/'fixed-writes'/name
        audit = run/'results/input-audit.md'
        config = {**original, 'protocol_version':'pilot4-writes-v1','run_id':name,
                  'source_contract':contract,'source_evidence':evidence,
                  'source_config_path':contract['training_path'], 'audit_path':str(audit.relative_to(root)),
                  'run_path':str(run.relative_to(root)), 'source_input_config_sha256':_hash(original)}
        # Keep the original items_path: no copy, re-tokenization or reselection.
        _save_frozen(path,config)
        _save_frozen(run/'meta/run_manifest.json',config)
        audit.parent.mkdir(parents=True,exist_ok=True)
        audit.write_text('# GRPO fixed-input audit\n\nExact Pilot 3 token IDs, masks and complete gold solutions reused.\n'
                         f"Input hash: {config['items_sha256']}\n\nSource: {contract['measurement_path']}\n"
                         'Fresh GRPO instrument validation, calibration and profiling are required.\n')
        files=[path,audit,run/'meta/run_manifest.json',root/config['items_path']]
        _write_json(plan/'prepare-complete.json',dict(config_sha256=_hash(config),
            files={str(p.relative_to(root)):file_hash(p) for p in files}))
        print(f'GRPO fixed inputs prepared: {len(rows)}; {path}',flush=True)
        return path

"""Read-only mixed-arm provenance verification; not a gate or training approval."""
from pathlib import Path

from pilot_eval import sampling_sft_generation as old
from pilot_eval import sampling_sft_generation_v2 as new
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


CONTRACT = ('source_plan','pilot_plan','pilot_report','settings','problems','arms',
            'data_seed','retry_indices','retry_data_seeds','algorithm_hash',
            'runtime_hash','baseline','eligibility')


def check_contract(legacy, buffered):
    if any(key not in legacy or key not in buffered or legacy[key] != buffered[key]
           for key in CONTRACT):
        raise ValueError('Mixed arms have different frozen scientific contracts.')
    if buffered.get('writer',{}).get('mode') != 'buffered-before-seal-v1':
        raise ValueError('Buffered writer identity missing.')


def verify(legacy_config, buffered_config, output_root):
    root = Path(output_root).resolve()
    _, a, af = old.load(legacy_config,root)
    _, b, bf = new.load(buffered_config,root)
    check_contract(a,b)
    new.check_writer_replay(root,b)
    arms = {}
    for k,module,plan,folder in ((0,old,a,af),(2,old,a,af),(5,new,b,bf)):
        result = module.verify_arm(root,plan,k,folder/f'k{k}')
        if result['state'] != 'completed' or result['execution_kind'] != 'real-cuda':
            raise ValueError('Each mixed arm must be independently complete and real-CUDA.')
        arms[str(k)] = {'plan_sha256':_hash(plan),
            'arm_complete_sha256':file_hash(folder/f'k{k}'/'complete.json'),
            'plan_path':str(inside(root,legacy_config if k != 5 else buffered_config).relative_to(root))}
    return {'protocol':'sampling-sft-mixed-arms-provenance-v1', 'arms':arms,
            'runtime_hash':a['runtime_hash'], 'algorithm_hash':a['algorithm_hash'],
            'scope':'read-only evidence only; not accepted by coverage gate or training',
            'training_allowed':False}

"""Versioned view of independently verified legacy and buffered arms."""
import json
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_generation as old
from pilot_eval import sampling_sft_generation_v2 as new
from pilot_eval import sampling_sft_generation_bridge as bridge
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

ARMS=old.ARMS
read_lines=old.read_lines
jsonl=old.jsonl
snapshot=old.snapshot


def implementation():
    parent=Path(__file__).parent
    return {name:file_hash(parent/name) for name in (
        'sampling_sft_mixed_generation.py','sampling_sft_generation_bridge.py')}


def prepare(legacy_config, buffered_config, output_root, name):
    root=Path(output_root).resolve(); safe_name(name)
    legacy_config=inside(root,legacy_config);buffered_config=inside(root,buffered_config)
    evidence=bridge.verify(legacy_config,buffered_config,root)
    _,legacy,_=old.load(legacy_config,root)
    source_plan=json.loads((root/legacy['source_plan']).read_text())
    folder=root/source_plan['run_path']/'mixed-generation'/name
    plan={**legacy,'protocol':'sampling-sft-mixed-generation-v1','name':name,
          'legacy_plan':str(legacy_config.relative_to(root)),
          'buffered_plan':str(buffered_config.relative_to(root)),
          'provenance':evidence,'implementation':implementation(),
          'run_path':str(folder.relative_to(root)),'training_allowed':False}
    target=root/'plans'/name/'sampling-sft.mixed-generation.json'
    source.frozen(target,plan);source.frozen(folder/'meta/run_manifest.json',{'config_sha256':_hash(plan)})
    return target


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path);plan=json.loads(path.read_text())
    folder=inside(root,plan['run_path'])
    if (plan.get('protocol')!='sampling-sft-mixed-generation-v1'
        or plan.get('implementation')!=implementation()
        or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(plan)}):
        raise ValueError('Mixed plan identity changed.')
    if bridge.verify(root/plan['legacy_plan'],root/plan['buffered_plan'],root)!=plan['provenance']:
        raise ValueError('Mixed arm provenance changed.')
    return root,plan,folder


def arm_directory(root,plan,k):
    if k not in ARMS:raise ValueError('Unknown arm.')
    module,path=(old,plan['legacy_plan']) if k in (0,2) else (new,plan['buffered_plan'])
    _,_,folder=module.load(root/path,root)
    return folder/f'k{k}'


def verify(path,output_root):
    _,plan,_=load(path,output_root)
    return {'state':'completed','execution_kind':'real-cuda',
            'runtime_hash':plan['runtime_hash'],'training_allowed':False}


def validate_for_scoring(root,plan):
    """Verify both original generation approvals and the v2 replay before scoring."""
    for module,key in ((old,'legacy_plan'),(new,'buffered_plan')):
        _,original,folder=module.load(root/plan[key],root)
        receipt=json.loads((folder/'generation.review.json').read_text())
        module.plan_approval(original,receipt['review'],file_hash(root/original['pilot_report']))
        _,assessment=module.qualification(root,original,receipt['review'])
        if (receipt.get('config_sha256')!=_hash(original)
            or receipt.get('alarm_assessment')!=assessment
            or receipt.get('pilot_report_sha256')!=file_hash(root/original['pilot_report'])):
            raise ValueError('Original generation approval changed.')
        if module is new:new.check_writer_replay(root,original)

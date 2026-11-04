"""Record the live runtimes and write the inputs for offline-grpo-overnight-template.

Run this on the same GPU session and git commit as the Stage 0 qualification. It
briefly loads the model to record the training runtime; it does not train.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FROZEN = 'plans/pilot4-grpo-fp32-v1-frozen-v2/grpo.frozen.json'
AUDIT = 'plans/stage0-geometry-v1/stage0.plan.json'
BASELINE = 'plans/pilot4-untuned-sampling-v1/baseline.config.json'
DATASET = 'plans/a100-stage0-v1/sampling-stage0.dataset.json'
CUT_REFERENCE = 'runs/stage0/a100-stage0-cuts-v1/principal-cuts/config.json'
FIXED_INPUTS = 'plans/pilot3-inputs-v1/activation.prepared.json'
ONLINE_BEHAVIOUR = 'plans/pilot4-grpo-fp32-v1-behaviour-v1/grpo.evaluation.json'
ONLINE_TRAINING = 'plans/pilot4-grpo-fp32-v1-arm-v1/grpo.training.json'


def build_inputs(root, *, name, tolerance, repo, plan, training, snapshot, cache, versions, file_hash, long_min_tokens=100, prompt_count=8):
    directory = f'notebook/{name}'
    weights = 'inputs/models/'+plan['model_revision']
    zero = plan['source_evidence']['checkpoints']['0']['path']
    return dict(audit=AUDIT, historical_baseline=BASELINE, frozen=FROZEN, weights_dir=weights,
        stage0_runtime_file=f'{directory}/stage0-runtime.json', training_runtime_file=f'{directory}/training-runtime.json',
        checkpoint_zero=zero, checkpoint_weights_sha256=file_hash(root/zero/'adapter_model.safetensors'),
        checkpoint_config_sha256=file_hash(root/zero/'adapter_config.json'),
        dataset_source=DATASET, dataset_root=str(root), fixed_inputs=FIXED_INPUTS,
        qualification=f'plans/{name}-qualification/stage0.qualification.json',
        cut_reference=CUT_REFERENCE, cut_reference_root=str(root), long_min_tokens=long_min_tokens, prompt_count=prompt_count,
        fidelity_tolerance_nats=tolerance, model_identity=json.loads((root/f'plans/{name}-qualification/stage0.qualification.json').read_text())['model_identity'],
        eos_token_id=training['eos_token_id'], snapshot_hashes=snapshot, hf_cache_root=cache, minimum_cache_free_bytes=5*10**9,
        source_hashes={n: file_hash(root/n) for n in [AUDIT, BASELINE, FROZEN, DATASET, CUT_REFERENCE, FIXED_INPUTS]},
        software_versions=versions, minimum_free_bytes=50*10**9, hf_repo=repo,
        export_paths=[directory, f'{directory}-qualification', f'runs/stage0/{name}-qualification'],
        online_behaviour=ONLINE_BEHAVIOUR, online_training=ONLINE_TRAINING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--fidelity-tolerance', type=float, required=True)
    parser.add_argument('--hf-repo', required=True)
    args = parser.parse_args()
    # Stage 0 always records this value, while training records the environment's.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE
    from pilot_eval.grpo_gpu import GPUPreflight
    from pilot_eval.grpo_preflight import load_frozen
    from pilot_eval.grpo_prepare import load_grpo_prepared
    from pilot_eval.run import _write_json
    from pilot_eval.sft_backend import PINS
    from pilot_eval.stage0_qualification_backend import QualificationDependencies
    from pilot_eval.training import file_hash
    root = args.root.resolve()
    output = root/'notebook'/args.name
    frozen = load_frozen(root/FROZEN, root)
    plan, *_ = load_grpo_prepared(root/frozen['source_path'], root)
    stage0 = QualificationDependencies().runtime()
    _write_json(output/'stage0-runtime.json', stage0)
    training = GPUPreflight(plan, frozen['settings'], Path(tempfile.mkdtemp())).runtime()
    _write_json(output/'training-runtime.json', training)
    snapshot = {p.name: file_hash(p) for p in sorted((root/'inputs/models'/plan['model_revision']).iterdir())
                if p.is_file() and isinstance(try_to_load_from_cache(plan['model'], p.name, revision=plan['model_revision']), str)}
    inputs = build_inputs(root, name=args.name, tolerance=args.fidelity_tolerance, repo=args.hf_repo, plan=plan,
                          training=training, snapshot=snapshot, cache=HF_HUB_CACHE, versions=dict(PINS), file_hash=file_hash)
    _write_json(output/'inputs.json', inputs)
    print('written', output/'inputs.json', len(snapshot), 'model files;', training['device'], stage0['git_commit'][:7])


if __name__ == '__main__':
    main()

"""Pinned public lens provenance inspection; no paid model calls or lens fitting."""
import json
from pathlib import Path
from pilot_eval.training import file_hash
from pilot_eval.trace_corpus import _inside
from pilot_eval.trace_lenses import inspect_lens

REPOSITORY = 'qbz506/prabodha-lenses'
REVISION = 'bc6aec86a96e7e75ce286dfb8f3e59aaaca50cce'
FILENAME = 'qwen2.5-1.5b-instruct.pt'
SHA256 = '6a23a7460139a0d3b73fdcc3a204bdee57475223bdfa72a284e6c7c60ed3a9cc'
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
UPSTREAM_REVISION = '581d398613e5602a5af361e1c34d3a92ea82ba8e'


def inspect_community(output_root, name, *, artifact_dir=None):
    root = Path(output_root).resolve()
    from pilot_eval.sft import safe_name
    safe_name(name)
    directory = _inside(root, artifact_dir) if artifact_dir else root/'runs/pilot-5'/name/'community-inputs'
    if artifact_dir is None:
        from huggingface_hub import hf_hub_download
        for filename in [FILENAME, 'README.md', 'lenses_manifest.json']:
            hf_hub_download(REPOSITORY, filename, revision=REVISION, local_dir=directory)
    # These are declared claims from this exact pinned snapshot, not inferred fit metadata.
    manifest = json.loads((directory/'lenses_manifest.json').read_text())
    matches = [r for r in manifest['lenses'] if r['lens_file'] == FILENAME]
    if len(matches) != 1 or matches[0]['model'] != 'Qwen/Qwen2.5-1.5B-Instruct':
        raise ValueError('Pinned lens manifest does not identify the expected model.')
    provenance = dict(model=matches[0]['model'],model_revision=None,source_site='block-output',
        target=None,repository=REPOSITORY,artifact_revision=REVISION,
        upstream_repository='anthropics/jacobian-lens',upstream_revision=UPSTREAM_REVISION,
        upstream_convention='J maps source block column vectors toward target; row-vector transport v @ J.T',
        declared_steering_site=matches[0]['site_layer'],
        conflicts=['Card describes band-targeted/band-exit lenses while also declaring target_layer=n_layers-1; exact fitted target/norm contract is not serialized.'],
        files={str((directory/n).relative_to(root)):file_hash(directory/n) for n in ['README.md','lenses_manifest.json']})
    return inspect_lens(directory/FILENAME,root,name,expected_sha256=SHA256,provenance=provenance,
        expected_model='Qwen/Qwen2.5-1.5B-Instruct',expected_revision=MODEL_REVISION)

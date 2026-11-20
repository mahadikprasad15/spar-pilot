"""Ticket 08: the Pod input bundle (CPU, fake Hub)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from pilot_eval import c_attribution_bundle as cb
from test_neuron_audit_export import FakeHub

ARTIFACTS = Path(os.environ.get('PILOT_ARTIFACTS', '/Users/prasadmahadik/spar/pilot1/artifacts'))
REPO = Path(__file__).resolve().parents[1]


def _root(tmp_path):
    root = tmp_path / 'artifacts'
    entries = []
    for i, role in enumerate(['responses:online-grpo', 'run-config:online-grpo', 'adapter:online-grpo:64',
                              'generation-config', 'hot-layer-source']):
        path = f'inputs/{role.replace(":", "_")}.bin'
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(f'{role}-{i}'.encode() * 10)
        entries.append({'role': role, 'path': path})
    return root, entries


def test_export_clean_restore_and_verify_against_the_seal(tmp_path):
    root, entries = _root(tmp_path)
    bundle = cb.build(root, entries, repo_root=REPO, base_model='not-bundled-fixture')
    assert bundle['catalog']['commit'] and bundle['catalog']['files']
    hub = FakeHub()
    receipt = cb.export(root, bundle, label='test', client=hub, temp_dir=tmp_path, on_event=lambda e: None)
    restored = cb.restore(receipt, tmp_path / 'clean', client=hub, on_event=lambda e: None)
    assert restored['verified_files'] == len(entries) and restored['bundle_sha256'] == bundle['sha256']
    # The restored copy is verified against the bundle seal, not rebuilt from the originals.
    (root / entries[0]['path']).write_bytes(b'changed after export')
    assert cb.verify(tmp_path / 'clean', restored['bundle'])['verified_files'] == len(entries)


def test_missing_and_differing_files_are_named_distinctly(tmp_path):
    root, entries = _root(tmp_path)
    bundle = cb.build(root, entries, repo_root=REPO, base_model='not-bundled-fixture')
    (root / entries[1]['path']).unlink()
    with pytest.raises(cb.BundleError, match='^missing: run-config:online-grpo'):
        cb.verify(root, bundle)
    (root / entries[1]['path']).write_bytes(b'other')
    with pytest.raises(cb.BundleError, match='^differs: run-config:online-grpo'):
        cb.verify(root, bundle)
    with pytest.raises(cb.BundleError, match='^missing'):
        cb.build(root, entries + [{'role': 'tokenizer:x', 'path': 'nope.json'}], repo_root=REPO, base_model='not-bundled-fixture')
    with pytest.raises(cb.BundleError, match='^differs'):
        cb.build(root, entries, repo_root=REPO, expected={'generation-config': '0' * 64}, base_model='not-bundled-fixture')
    tampered = dict(bundle, files=bundle['files'][:-1])
    with pytest.raises(cb.BundleError, match='seal'):
        cb.verify(root, tampered)


def test_missing_catalog_revision_is_refused(tmp_path):
    root, entries = _root(tmp_path)
    with pytest.raises(cb.BundleError, match='missing catalog revision'):
        cb.build(root, entries, repo_root=tmp_path / 'not-a-checkout', base_model='not-bundled-fixture')
    plain = tmp_path / 'plain'
    (plain / 'catalog').mkdir(parents=True)
    (plain / 'catalog' / 'arms.yaml').write_text('arms: {}\n')
    with pytest.raises(cb.BundleError, match='missing catalog revision'):
        cb.catalog_revision(plain)
    subprocess.run(['git', 'init', '-q', str(plain)], check=True)
    with pytest.raises(cb.BundleError):
        cb.catalog_revision(plain)  # untracked catalog/: uncommitted changes or no HEAD


def test_base_model_files_are_checked_after_download(tmp_path):
    root, entries = _root(tmp_path)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'weights')
    pinned = cb.pinned_identities(model)  # this fixture's stand-in for HF's published per-file IDs
    bundle = cb.build(root, entries, repo_root=REPO, model_dir=model, pinned=pinned)
    assert cb.check_base_model(model, bundle, pinned=pinned)
    (model / 'model.safetensors').write_bytes(b'other')
    with pytest.raises(cb.BundleError, match='differs: base model'):
        cb.check_base_model(model, bundle, pinned=pinned)
    (model / 'model.safetensors').write_bytes(b'weights')
    (model / 'config.json').unlink()
    with pytest.raises(cb.BundleError, match='missing: base model'):
        cb.check_base_model(model, bundle, pinned=pinned)


def test_real_inputs_build_and_match_the_catalog():
    entries = cb.default_entries()
    if not all((ARTIFACTS / e['path']).is_file() for e in entries) or \
            not (ARTIFACTS / cb.MODEL_DIR / 'model.safetensors').is_file():
        pytest.skip('real C-attribution inputs are not all local')
    pytest.importorskip('yaml')
    expected = {'responses:online-grpo': '4af7926a3def742d9eef128f3b7306b783aada413f785624054dacb16ee971ec',
                'responses:offline-grpo': 'a1cd79aa24e9cf7a87d0d8db3bd15588b97e830229db662892038f4b7437be94',
                'adapter:online-grpo:64:adapter_model.safetensors': '70d2d56cd4636933020b107d6fb344ed451bc879b4a694e725c8608181f07226',
                'adapter:offline-grpo:64:adapter_model.safetensors': '6601551812ab754ccf35c0fbd5f6cf181786636683e9367b24c4f92451a1c93f',
                'adapter:untuned-zero:0:adapter_model.safetensors': 'b26c524ab41540b2380ad384bba6a41df5bc9c86d228a9f130a60e89ee2ef7d5',
                'hot-layer-source': '35747fda0615032d996e3671833ed656a4eebf7a9b047a5a27656f6002ffd30d',
                'generation-config': 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6'}
    bundle = cb.build(ARTIFACTS, entries, repo_root=REPO, expected=expected)  # the pinned base model is hashed (about 3 GB)
    assert cb.verify(ARTIFACTS, bundle)['verified_files'] == len(entries)
    assert cb.check_base_model(ARTIFACTS / cb.MODEL_DIR, bundle)
    report = cb.pod_catalog_check(REPO, bundle, ARTIFACTS, ['responses:online-grpo.s64.sampled-x8.test150',
                                                            'responses:offline-grpo.s64.sampled-x8.test150',
                                                            'cohort:gsm8k-test-cohort150'])
    assert set(report.values()) == {'pass'}


def test_base_model_hashes_are_required_and_fail_closed(tmp_path):
    """Review P1: an empty base-model hash map verified nothing; it now fails closed."""
    root, entries = _root(tmp_path)
    with pytest.raises(cb.BundleError, match='missing: base model'):
        cb.build(root, entries, repo_root=REPO, model_dir=tmp_path / 'nonexistent')
    partial = tmp_path / 'partial'
    partial.mkdir()
    (partial / 'config.json').write_text('{}')
    with pytest.raises(cb.BundleError, match='missing: base model model.safetensors'):
        cb.build(root, entries, repo_root=REPO, model_dir=partial)
    fixture = cb.build(root, entries, repo_root=REPO, base_model='not-bundled-fixture')
    with pytest.raises(cb.BundleError, match='no base-model hashes'):
        cb.check_base_model(tmp_path / 'nonexistent', fixture)
    tampered = dict(fixture, base_model=dict(fixture['base_model'], file_sha256={}))
    with pytest.raises(cb.BundleError, match='no base-model hashes'):
        cb.check_base_model(tmp_path / 'nonexistent', tampered)


def _fake_snapshot(directory, weights):
    directory.mkdir()
    (directory / 'config.json').write_text('{"model_type": "qwen2"}')
    (directory / 'model.safetensors').write_bytes(weights)
    return directory


def test_a_self_consistent_snapshot_cannot_be_labelled_the_pinned_revision(tmp_path):
    """Review 06-10 P1: expected base-model hashes came from the files being certified. They now come from
    HF's own per-file IDs for revision 989aa798 (LFS sha256, git blob id), fixed in the code."""
    root, entries = _root(tmp_path)
    assert cb.PINNED_BASE_MODEL['model.safetensors'] == {
        'sha256': 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'}
    assert cb.PINNED_BASE_MODEL['config.json'] == {'git_blob_sha1': 'f81ead14ab072d65a07817f83a3ee0e5a1890d10'}
    a = _fake_snapshot(tmp_path / 'a', b'weights-a')
    b = _fake_snapshot(tmp_path / 'b', b'weights-b')
    for snapshot in (a, b):  # neither valid-looking snapshot is accepted as the pinned revision
        with pytest.raises(cb.BundleError, match='differs from HF revision 989aa798'):
            cb.build(root, entries, repo_root=REPO, model_dir=snapshot)
    # a bundle sealed under other pins cannot pass the Pod check, which uses the code's pins
    own = cb.build(root, entries, repo_root=REPO, model_dir=a, pinned=cb.pinned_identities(a))
    with pytest.raises(cb.BundleError, match='differs from HF revision 989aa798'):
        cb.check_base_model(a, own)
    swapped = dict(own, base_model=dict(own['base_model'], pinned=cb.pinned_identities(b)))
    with pytest.raises(cb.BundleError, match='other base-model pins'):
        cb.check_base_model(a, swapped, pinned=cb.pinned_identities(a))


def test_the_local_pinned_model_matches_hf_identities():
    model = ARTIFACTS / cb.MODEL_DIR
    if not (model / 'model.safetensors').is_file():
        pytest.skip('the pinned base model is not local')
    assert cb.pinned_identities(model) == cb.PINNED_BASE_MODEL  # hashes about 3 GB

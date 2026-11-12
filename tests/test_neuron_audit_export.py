"""Ticket 07 export/restore with a mocked Hub: partial/complete, readback, failures, Pod-stop gate."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_eval import neuron_audit_export as ex
from pilot_eval.neuron_audit_blocks import BlockStore

REPO = 'owner/private-audit'


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f'HTTP {status}')
        self.response = SimpleNamespace(status_code=status)


class FakeHub:
    cache_root = None

    def __init__(self, private=True):
        self.private, self.store, self.publishes, self.fail_publish, self.corrupt_fetch = private, {}, 0, [], False

    def info(self, repo):
        return SimpleNamespace(private=self.private)

    def publish(self, repo, files):
        if self.fail_publish:
            raise self.fail_publish.pop(0)
        self.publishes += 1
        revision = hashlib.sha1(f'{self.publishes}'.encode()).hexdigest()
        self.store[revision] = {n: Path(p).read_bytes() for n, p in files.items()}
        return revision

    def fetch_to(self, repo, name, revision, directory):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.store[revision][name]
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]) if self.corrupt_fetch else data)
        return path


def sealed_run(directory, files):
    directory.mkdir(parents=True)
    for name, text in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)
    seal = {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in files}
    (directory / 'complete.json').write_text(json.dumps(dict(files=seal)))
    (directory / 'meta').mkdir()
    (directory / 'meta/status.json').write_text(json.dumps(dict(state='complete')))


def unit(store, key):
    rows = 2
    store.write_unit(key, identity=dict(k=key), example_ids=['a', 'b'], counts={'response_content': [3, 3]},
                     blocks={('response_content', 0, 'mean_z'): np.ones((rows, 4), np.float32)})


@pytest.fixture
def root(tmp_path):
    r = tmp_path / 'artifacts'
    base = r / 'runs/joint-neuron-adapter-audit'
    sealed_run(base / 'sequences/s1', {'sequences.jsonl': '{}\n', 'manifest.json': '{}'})
    store = BlockStore(base / 'collection/c1')
    unit(store, 'gold-rows-0-1')
    unit(store, 'untuned-rows-0-1')
    (base / 'collection/c1/units/gold-rows-2-3').mkdir(parents=True)  # killed mid-unit
    (base / 'collection/c1/units/gold-rows-2-3/blocks.f32').write_bytes(b'partial')
    (base / 'collection/c1/logs').mkdir()
    (base / 'collection/c1/logs/run.log').write_text('worker stopped\n')
    return r


INCLUDE = ['runs/joint-neuron-adapter-audit/sequences', 'runs/joint-neuron-adapter-audit/collection']
OMITTED = [dict(item='Qwen/Qwen2.5-1.5B-Instruct', revision='989aa79', recovery='hf download at pinned revision')]


def snap(root, **kw):
    return ex.snapshot(root, INCLUDE, 'fixture', omitted=OMITTED, **kw)


def test_partial_snapshot_lists_unsealed_and_archives_only_sealed(root):
    s = snap(root, expected_units=['gold-rows-0-1', 'untuned-rows-0-1', 'gold-rows-2-3'])
    assert s['completeness'] == 'partial'
    kinds = {}
    for name, rec in s['files'].items():
        kinds.setdefault(rec['kind'], []).append(name)
    assert any(n.endswith('gold-rows-0-1/seal.json') for n in kinds['sealed-unit'])
    assert not any('gold-rows-2-3' in n for n in s['files'])
    assert {u['state'] for u in s['unsealed']} == {'interrupted'}
    assert s['missing_expected_units'] == ['gold-rows-2-3']
    assert any(n.endswith('logs/run.log') for n in kinds['failure-evidence'])
    assert any(n.endswith('meta/status.json') for n in kinds['status'])
    assert s['omitted_reconstructible'] == OMITTED


def test_complete_after_resume_seals_the_missing_unit(root):
    store = BlockStore(root / 'runs/joint-neuron-adapter-audit/collection/c1')
    unit(store, 'gold-rows-2-3')  # moves the interrupted attempt aside, then seals
    s = snap(root, expected_units=['gold-rows-0-1', 'untuned-rows-0-1', 'gold-rows-2-3'])
    assert s['completeness'] == 'complete' and s['interrupted_evidence']


def test_upload_readback_restore_and_pod_stop_gate(root, tmp_path):
    hub = FakeHub()
    s = snap(root)
    record = ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub, on_event=lambda e: None)
    assert record['completeness'] == 'partial' and len(record['revision']) == 40
    proof = ex.restore(root / Path(s['snapshot_path']).with_name(Path(s['snapshot_path']).stem + '.hf-upload.json'),
                       tmp_path / 'clean', client=hub)
    assert proof['verified_files'] == len(s['files']) and proof['verified_units'] == 2
    assert proof['completeness'] == 'partial' and proof['missing']['unsealed']
    assert ex.may_stop_pod(record, proof) is True
    assert ex.may_stop_pod(record, proof, require_complete=True) is False  # partial is never complete
    restored = (tmp_path / 'clean' / 'runs/joint-neuron-adapter-audit/sequences/s1/sequences.jsonl').read_text()
    assert restored == '{}\n'
    with pytest.raises(ValueError, match='empty, clean root'):
        ex.restore(record, tmp_path / 'clean', client=hub)
    # Resume: an existing receipt is returned without a second commit.
    assert ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub) == record and hub.publishes == 1


def test_public_repository_refused_before_any_transfer(root):
    hub = FakeHub(private=False)
    s = snap(root)
    with pytest.raises(ValueError, match='private'):
        ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub)
    assert hub.publishes == 0


def test_readback_mismatch_preserves_local_export_and_blocks_stop(root):
    hub = FakeHub()
    hub.corrupt_fetch = True
    s = snap(root)
    with pytest.raises(ValueError, match='readback'):
        ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub, on_event=lambda e: None)
    assert not list((root / 'exports').rglob('*.hf-upload.json'))
    assert (root / s['snapshot_path']).exists()


def test_auth_failure_is_fatal_and_transient_failure_retried(root):
    s = snap(root)
    hub = FakeHub()
    hub.fail_publish = [HTTPError(401)]
    with pytest.raises(PermissionError, match='authentication'):
        ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub)
    hub = FakeHub()
    hub.fail_publish = [HTTPError(503)]
    events = []
    record = ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub, on_event=events.append,
                       sleep=lambda seconds: None)
    assert events[0]['kind'] == 'hf-retry' and events[0]['status'] == 503 and record['revision']


def test_file_changed_after_snapshot_refused(root):
    s = snap(root)
    (root / 'runs/joint-neuron-adapter-audit/collection/c1/logs/run.log').write_text('changed\n')
    with pytest.raises(ValueError, match='changed after snapshot'):
        ex.upload(root / s['snapshot_path'], root, repo=REPO, client=FakeHub())


def test_credential_in_status_file_refuses_snapshot(root):
    (root / 'runs/joint-neuron-adapter-audit/collection/c1/logs/run.log').write_text('token hf_' + 'a' * 34 + '\n')
    with pytest.raises(ValueError, match='credential'):
        snap(root)


def test_corrupt_sealed_unit_listed_and_not_archived(root):
    path = root / 'runs/joint-neuron-adapter-audit/collection/c1/units/untuned-rows-0-1/blocks.f32'
    path.write_bytes(b'\0' * path.stat().st_size)
    s = snap(root)
    assert s['completeness'] == 'partial' and s['corrupt'][0]['path'].endswith('untuned-rows-0-1')
    assert not any('untuned-rows-0-1' in n for n in s['files'])


def test_tampered_remote_archive_refused_at_restore(root, tmp_path):
    hub = FakeHub()
    s = snap(root)
    record = ex.upload(root / s['snapshot_path'], root, repo=REPO, client=hub)
    hub.corrupt_fetch = True
    with pytest.raises(ValueError, match='archive hash'):
        ex.restore(record, tmp_path / 'clean', client=hub)


def test_archive_built_in_local_temp_not_beside_the_snapshot(root, tmp_path, monkeypatch):
    seen = []
    real = ex.tempfile.TemporaryDirectory

    def spy(*args, **kwargs):
        seen.append(kwargs.get('dir'))
        return real(*args, **kwargs)

    monkeypatch.setattr(ex.tempfile, 'TemporaryDirectory', spy)
    s = snap(root)
    local = tmp_path / 'local-temp'
    local.mkdir()
    ex.upload(root / s['snapshot_path'], root, repo=REPO, client=FakeHub(), temp_dir=local)
    assert seen and all(Path(d) == local for d in seen)


# --- per-arm packages, reference-only final manifest, Global-Volume mirror -----------------------

from pilot_eval.neuron_audit_blocks import CorruptUnit, mirror_store  # noqa: E402


def arm_store(root):
    store = BlockStore(root)
    for arm in ['untuned', 'gold-sft']:
        for start in (0, 2):
            store.write_unit(f'{arm}-rows-{start}', identity=dict(arm=arm, k=start), example_ids=[f'e{start}', f'e{start + 1}'],
                             counts={'response_content': [3, 3]},
                             blocks={('response_content', 0, 'mean_z'): np.full((2, 4), start + 1, np.float32)})
    return store


def test_arm_packages_are_deterministic_and_verified_by_download(tmp_path):
    arm_store(tmp_path / 'local')
    p1, m1, manifest = ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path)
    p2, _, _ = ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path)
    assert p1.read_bytes() == p2.read_bytes() and manifest['unit_count'] == 2
    hub = FakeHub()
    record = ex.upload_package(p1, m1, manifest, repo=REPO, label='prod', client=hub, temp_dir=tmp_path)
    assert record['path'].startswith('neuron-audit/prod/arms/gold-sft/') and record['verified'].startswith('fresh')
    hub.corrupt_fetch = True
    with pytest.raises(ValueError, match='readback differs'):
        ex.upload_package(p1, m1, manifest, repo=REPO, label='prod', client=hub, temp_dir=tmp_path)


def test_final_manifest_references_packages_and_gates_pod_stop(tmp_path):
    arm_store(tmp_path / 'local')
    hub, records = FakeHub(), []
    for arm in ['untuned', 'gold-sft']:
        package, mpath, manifest = ex.package_arm(tmp_path / 'local', arm, 'prod', tmp_path)
        records.append(ex.upload_package(package, mpath, manifest, repo=REPO, label='prod', client=hub, temp_dir=tmp_path))
    partial = ex.publish_final_manifest(records[:1], repo=REPO, label='prod', required_arms=['untuned', 'gold-sft'],
                                        client=hub, temp_dir=tmp_path)
    assert partial['completeness'] == 'partial' and partial['missing_arms'] == ['gold-sft']
    assert ex.may_stop_pod_final(partial, records[:1], required_arms=['untuned', 'gold-sft']) is False
    final = ex.publish_final_manifest(records, repo=REPO, label='prod', required_arms=['untuned', 'gold-sft'],
                                      client=hub, temp_dir=tmp_path, extras=dict(qualification='q-sha'))
    stored = hub.store[final['revision']][final['path']]
    assert b'package.tar' in stored and len(stored) < 10_000  # references only, no package bytes
    assert ex.may_stop_pod_final(final, records, required_arms=['untuned', 'gold-sft']) is True
    proof = ex.restore_final(final, tmp_path / 'clean', client=hub, temp_dir=tmp_path)
    assert proof['verified_units'] == {'gold-sft': 2, 'untuned': 2} and proof['completeness'] == 'complete'
    assert ex.may_stop_pod_final(final, records, required_arms=['untuned', 'gold-sft'], restore_proof=proof) is True
    unverified = [dict(r, verified='not checked') for r in records]
    assert ex.may_stop_pod_final(final, unverified, required_arms=['untuned', 'gold-sft']) is False


def test_restore_final_refuses_tampered_package(tmp_path):
    arm_store(tmp_path / 'local')
    hub = FakeHub()
    package, mpath, manifest = ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path)
    record = ex.upload_package(package, mpath, manifest, repo=REPO, label='prod', client=hub, temp_dir=tmp_path)
    final = ex.publish_final_manifest([record], repo=REPO, label='prod', required_arms=['gold-sft'], client=hub,
                                      temp_dir=tmp_path)
    data = hub.store[record['revision']][record['path']]
    hub.store[record['revision']][record['path']] = data[:-1] + bytes([data[-1] ^ 1])
    with pytest.raises(ValueError, match='package hash differs'):
        ex.restore_final(final, tmp_path / 'clean', client=hub, temp_dir=tmp_path)


def test_mirror_copies_seal_last_verifies_without_listing_and_repairs_torn_copy(tmp_path):
    arm_store(tmp_path / 'local')
    assert set(mirror_store(tmp_path / 'local', tmp_path / 'volume').values()) == {'mirrored'}
    assert set(mirror_store(tmp_path / 'local', tmp_path / 'volume').values()) == {'already-mirrored'}
    torn = tmp_path / 'volume/units/gold-sft-rows-0/blocks.f32'
    torn.write_bytes(torn.read_bytes()[:-4])  # truncated copy on the volume
    with pytest.raises(CorruptUnit):
        BlockStore(tmp_path / 'volume').verify_unit('gold-sft-rows-0', force=True, use_listing=False)
    assert mirror_store(tmp_path / 'local', tmp_path / 'volume')['gold-sft-rows-0'] == 'mirrored'
    (tmp_path / 'volume/units/gold-sft-rows-2/seal.json').write_text('{"files": ')  # torn seal
    with pytest.raises(CorruptUnit, match='unreadable'):
        BlockStore(tmp_path / 'volume').verify_unit('gold-sft-rows-2', force=True, use_listing=False)
    assert mirror_store(tmp_path / 'local', tmp_path / 'volume')['gold-sft-rows-2'] == 'mirrored'


def test_pinned_inputs_are_hash_checked_and_archived(root, tmp_path):
    adapter = root / 'adapters/gold/adapter_model.safetensors'
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b'weights')
    digest = hashlib.sha256(b'weights').hexdigest()
    s = ex.snapshot(root, INCLUDE, 'prep', pinned_inputs=[dict(path='adapters/gold/adapter_model.safetensors', sha256=digest)])
    assert s['files']['adapters/gold/adapter_model.safetensors']['kind'] == 'pinned-input'
    with pytest.raises(ValueError, match='differs from its recorded hash'):
        ex.snapshot(root, INCLUDE, 'prep', pinned_inputs=[dict(path='adapters/gold/adapter_model.safetensors', sha256='0' * 64)])


def test_stage_seals_and_optional_unsealed_evidence(root):
    stage = root / 'runs/joint-neuron-adapter-audit/qualification/q/stages/masks'
    stage.mkdir(parents=True)
    (stage / 'result.json').write_text('{}')
    (stage / 'complete.json').write_text(json.dumps(dict(identity={}, result_sha256=hashlib.sha256(b'{}').hexdigest())))
    (stage.parent.parent / 'logs').mkdir()
    (stage.parent.parent / 'logs/progress.jsonl').write_text('{}\n')
    s = ex.snapshot(root, ['runs/joint-neuron-adapter-audit/qualification'], 'q')
    assert s['files']['runs/joint-neuron-adapter-audit/qualification/q/stages/masks/result.json']['kind'] == 'sealed-run'
    assert 'runs/joint-neuron-adapter-audit/qualification/q/logs/progress.jsonl' not in s['files']
    s = ex.snapshot(root, ['runs/joint-neuron-adapter-audit/qualification'], 'q', include_unsealed_evidence=True)
    assert s['files']['runs/joint-neuron-adapter-audit/qualification/q/logs/progress.jsonl']['kind'] == 'unsealed-evidence'
    assert s['completeness'] == 'partial'


def test_production_package_must_hold_exactly_the_expected_units(tmp_path):
    arm_store(tmp_path / 'local')
    with pytest.raises(ValueError, match='not sealed locally'):
        ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path,
                       expected_keys=['gold-sft-rows-0', 'gold-sft-rows-2', 'gold-sft-rows-4'])
    with pytest.raises(ValueError, match='another arm'):
        ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path, expected_keys=['untuned-rows-0'])
    hub = FakeHub()
    package, mpath, manifest = ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path,
                                              expected_keys=['gold-sft-rows-0'])  # a short package, built on purpose
    record = ex.upload_package(package, mpath, manifest, repo=REPO, label='prod', client=hub, temp_dir=tmp_path)
    expected = {'gold-sft': ['gold-sft-rows-0', 'gold-sft-rows-2']}
    final = ex.publish_final_manifest([record], repo=REPO, label='prod', required_arms=['gold-sft'], client=hub,
                                      temp_dir=tmp_path, expected_units=expected)
    assert final['completeness'] == 'partial' and final['incomplete_arms'] == {'gold-sft': ['gold-sft-rows-2']}
    assert ex.may_stop_pod_final(final, [record], required_arms=['gold-sft'], expected_units=expected) is False


def test_online_hub_process_runs_hub_calls_online_and_classifies_failures(tmp_path):
    calls = []

    def runner(command, env, **kw):
        request = json.loads(command[command.index('--request') + 1])
        calls.append((request, env))
        if request['op'] == 'publish':
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(ok=True, revision='a' * 40)), stderr='')
        if request['op'] == 'info':
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(ok=True, private=True)), stderr='')
        reply = dict(ok=False, error_type='HfHubHTTPError', message='503 hf_' + 'x' * 34, status=503, retryable=True)
        return SimpleNamespace(returncode=1, stdout=json.dumps(reply), stderr='')

    client = ex.OnlineHubProcess(runner=runner)
    assert client.info('o/r').private is True and client.publish('o/r', {'a.txt': tmp_path / 'a'}) == 'a' * 40
    assert all(env['HF_HUB_OFFLINE'] == '0' for _, env in calls)
    assert all('hf_' not in json.dumps(req) for req, _ in calls)  # no token on the command line
    with pytest.raises(ex.HubProcessError) as info:
        client.fetch_to('o/r', 'x', 'a' * 40, tmp_path)
    assert info.value.response.status_code == 503 and 'hf_x' not in str(info.value)
    events = []
    with pytest.raises(ex.HubProcessError):
        ex._with_retries(lambda: client.fetch_to('o/r', 'x', 'a' * 40, tmp_path), attempts=2,
                         on_event=events.append, sleep=lambda s: None)
    assert events and events[0]['kind'] == 'hf-retry'


def test_audit_hub_refuses_to_start_in_an_offline_process(monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, 'HF_HUB_OFFLINE', True)
    with pytest.raises(RuntimeError, match='OnlineHubProcess'):
        ex.AuditHub()


def test_restore_final_qualification_only_skips_arm_downloads(tmp_path):
    arm_store(tmp_path / 'local')
    hub = FakeHub()
    package, mpath, manifest = ex.package_arm(tmp_path / 'local', 'gold-sft', 'prod', tmp_path)
    record = ex.upload_package(package, mpath, manifest, repo=REPO, label='prod', client=hub, temp_dir=tmp_path)
    qual = tmp_path / 'q-v1'
    (qual / 'stages').mkdir(parents=True)
    (qual / 'qualification.json').write_text('{"status": "qualified"}')
    (qual / 'stages/x.json').write_text('{}')
    qpack = ex.upload_directory_package(qual, repo=REPO, label='prod', kind='qualification', client=hub,
                                        temp_dir=tmp_path)
    final = ex.publish_final_manifest([record], repo=REPO, label='prod', required_arms=['gold-sft'], client=hub,
                                      temp_dir=tmp_path, extras=dict(qualification_package=qpack))
    proof = ex.restore_final(final, tmp_path / 'clean', client=hub, temp_dir=tmp_path, arms=False)
    assert proof['verified_units'] == {} and not (tmp_path / 'clean/arms').exists()
    restored = Path(proof['restored_extras']['qualification_package'])
    assert restored == (tmp_path / 'clean/qualification/q-v1').resolve()
    assert (restored / 'qualification.json').read_text() == '{"status": "qualified"}'

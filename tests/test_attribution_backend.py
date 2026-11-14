"""Ticket 11: pure parts of the real backend (CPU; the CUDA loader itself runs only in 06a/06b)."""
import pytest

from pilot_eval import attribution_backend as ab
from pilot_eval.protocol import select_gsm8k_indices


def test_cohort_is_adr_0001():
    assert ab.cohort_indices(1319) == select_gsm8k_indices(1319, 150, 42)


def test_rehearsal_draw_rule_user_2026_11_14():
    import random
    cohort = set(select_gsm8k_indices(1319, 150, 42))
    expected = sorted(random.Random(42).sample([i for i in range(1319) if i not in cohort], 20))
    assert ab.rehearsal_draw(1319) == expected and not cohort & set(expected) and len(set(expected)) == 20


def test_rehearsal_ids_must_be_explicit_distinct_and_outside_the_cohort():
    cohort = set(ab.cohort_indices(1319))
    outside = [i for i in range(1319) if i not in cohort][:20]
    assert ab.rehearsal_indices(outside, 1319) == outside
    with pytest.raises(ValueError, match='overlap'):
        ab.rehearsal_indices(outside[:19] + [min(cohort)], 1319)
    for bad in (None, outside[:19], outside[:19] + outside[:1], outside[:19] + [5000]):
        with pytest.raises(ValueError):
            ab.rehearsal_indices(bad, 1319)


# --- review round 4 (3): actual inputs bound to their sealed identities ------------------------------

import hashlib  # noqa: E402
import json  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    arms = {}
    for arm in ('gold-sft', 'online-grpo'):
        d = tmp_path / 'adapters' / arm
        d.mkdir(parents=True)
        (d / 'adapter_model.safetensors').write_bytes(arm.encode() * 10)
        (d / 'adapter_config.json').write_text(json.dumps(dict(r=1, arm=arm)))
        arms[arm] = d
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'weights')
    (model / 'config.json').write_text('{}')
    parquet = tmp_path / 'test.parquet'
    parquet.write_bytes(b'test rows')
    inventory = dict(
        arms={arm: dict(checkpoints={'64': dict(status='verified', files={
            'adapter_model.safetensors': dict(sha256=sha(d / 'adapter_model.safetensors')),
            'adapter_config.json': dict(sha256=sha(d / 'adapter_config.json'))})}) for arm, d in arms.items()},
        base=dict(expected_file_sha256={'model.safetensors': sha(model / 'model.safetensors'),
                                        'config.json': sha(model / 'config.json')}))
    manifest = dict(inputs=dict(files=dict(gsm8k_test=sha(parquet))))
    return inventory, manifest, arms, model, parquet


def test_r4_3_inputs_bound_to_inventory_and_manifest(tmp_path):
    inventory, manifest, arms, model, parquet = fixture(tmp_path)
    record = ab.bind_sources(inventory, manifest, adapter_dirs=arms, model_dir=model, test_parquet=parquet)
    assert record['arms']['gold-sft']['adapter_model.safetensors'] == sha(arms['gold-sft'] / 'adapter_model.safetensors')
    assert record['step'] == '64' and record['test_parquet_sha256'] == sha(parquet)


def test_r4_3_swapped_arm_directories_are_refused(tmp_path):
    inventory, manifest, arms, model, parquet = fixture(tmp_path)
    swapped = {'gold-sft': arms['online-grpo'], 'online-grpo': arms['gold-sft']}
    with pytest.raises(ValueError, match='gold-sft'):
        ab.bind_sources(inventory, manifest, adapter_dirs=swapped, model_dir=model, test_parquet=parquet)


def test_r4_3_other_base_model_or_test_file_or_unverified_checkpoint_is_refused(tmp_path):
    inventory, manifest, arms, model, parquet = fixture(tmp_path)
    (model / 'model.safetensors').write_bytes(b'other weights')
    with pytest.raises(ValueError, match='base model'):
        ab.bind_sources(inventory, manifest, adapter_dirs=arms, model_dir=model, test_parquet=parquet)
    inventory, manifest, arms, model, parquet = fixture(tmp_path / 'b')
    parquet.write_bytes(b'another split')
    with pytest.raises(ValueError, match='test parquet'):
        ab.bind_sources(inventory, manifest, adapter_dirs=arms, model_dir=model, test_parquet=parquet)
    inventory, manifest, arms, model, parquet = fixture(tmp_path / 'c')
    inventory['arms']['online-grpo']['checkpoints']['64']['status'] = 'unresolved'
    with pytest.raises(ValueError, match='online-grpo'):
        ab.bind_sources(inventory, manifest, adapter_dirs=arms, model_dir=model, test_parquet=parquet)
    inventory, manifest, arms, model, parquet = fixture(tmp_path / 'd')
    with pytest.raises(ValueError, match='sampling-sft-k0'):
        ab.bind_sources(inventory, manifest, adapter_dirs=dict(arms, **{'sampling-sft-k0': arms['gold-sft']}),
                        model_dir=model, test_parquet=parquet)

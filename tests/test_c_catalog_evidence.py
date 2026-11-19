from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize('arm', ['online-grpo', 'offline-grpo'])
def test_sampled_arm_catalog_does_not_claim_unchecked_policy_is_verified(arm):
    root = Path(__file__).resolve().parents[1]
    path = root / 'catalog' / 'responses' / f'{arm}.s64.sampled-x8.test150.yaml'
    evidence = yaml.safe_load(path.read_text())['evidence']
    assert evidence['status'] == 'inferred'
    assert 'processor' in evidence['note'].lower()
    assert 'unhashed' in evidence['note'].lower()

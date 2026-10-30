"""Cross-CPU verification of sealed drift products (FP64 last-ulp differences)."""
import numpy as np
import pytest

import pilot_eval.stage0_drift as drift


def test_float_agreement_tolerates_last_ulp_but_not_real_change():
    saved = {'V0a': 12.859320802641816, 'n': 3, 'id': 'x', 'cells': [{'kl': 0.1, 'count': 2}]}
    ulp = {'V0a': 12.859320802641813, 'n': 3, 'id': 'x', 'cells': [{'kl': 0.1 * (1 + 3e-16), 'count': 2}]}
    assert drift._agrees(saved, ulp)
    assert not drift._agrees(saved, {**ulp, 'V0a': 12.8593208})
    assert not drift._agrees(saved, {**ulp, 'n': 4})
    assert not drift._agrees(saved, {**ulp, 'id': 'y'})
    assert not drift._agrees(saved, {**ulp, 'extra': 1})
    assert not drift._agrees({'a': [1., 2.]}, {'a': [1.]})
    assert drift._agrees({'a': 0.0, 'b': None}, {'a': 0.0, 'b': None})
    assert not drift._agrees({'a': 0.0}, {'a': 1e-300})
    assert not drift._agrees({'a': 1}, {'a': True})
    assert drift._agrees({'a': 0.5}, {'a': np.float64(0.5) * (1 + 2e-16)})


def test_only_the_named_verification_predecessor_keeps_its_identity():
    current = drift._implementation()
    assert drift._implementation_matches(current)
    predecessor = dict(current, **{'stage0_drift.py': sorted(drift._VERIFICATION_PREDECESSORS['stage0_drift.py'])[0]})
    assert drift._implementation_matches(predecessor)
    assert not drift._implementation_matches(dict(current, **{'stage0_drift.py': '0' * 64}))
    other = next(k for k in current if k != 'stage0_drift.py')
    assert not drift._implementation_matches(dict(predecessor, **{other: '0' * 64}))
    assert not drift._implementation_matches({k: v for k, v in current.items() if k != other})

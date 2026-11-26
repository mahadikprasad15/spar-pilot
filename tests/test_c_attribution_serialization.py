"""Review 06-10 P2: sealed JSON keeps NumPy values as numbers (it used ``default=str``)."""
import json

import numpy as np
import pytest

from pilot_eval import attribution_units as au


def test_numpy_values_round_trip_through_a_sealed_run_as_numbers(tmp_path):
    lo = np.array([-1.9625, -2.9625])
    value = {'lo': lo, 'mean': np.float64(0.1 + 0.2), 'count': np.int64(7), 'grid': np.arange(4.0).reshape(2, 2),
             'flag': np.bool_(True)}
    au.seal_run(tmp_path / 'run', {'boot.json': value})
    back = json.loads((tmp_path / 'run' / 'boot.json').read_text())
    assert back == {'lo': [-1.9625, -2.9625], 'mean': 0.1 + 0.2, 'count': 7, 'grid': [[0.0, 1.0], [2.0, 3.0]],
                    'flag': True}
    assert back['lo'] == lo.tolist()  # lossless: full float precision, not NumPy's display precision


def test_an_unknown_object_is_refused_rather_than_stringified(tmp_path):
    with pytest.raises(TypeError, match='object'):
        au.seal_run(tmp_path / 'run', {'x.json': {'thing': object()}})

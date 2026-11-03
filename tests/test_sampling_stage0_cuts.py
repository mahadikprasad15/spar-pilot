import numpy as np
import pytest

from pilot_eval.sampling_stage0_cuts import module_edits, variant_catalogue


def test_dense_oracle_and_controls():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 40)).astype(np.float32)
    a = rng.normal(size=(1, 40)).astype(np.float32)
    b = rng.normal(size=(40, 1)).astype(np.float32)
    edits, evidence = module_edits(matrix, {'arm': (a, b)})
    u, s, vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    k = np.searchsorted(np.cumsum(s*s), .5 * np.sum(s*s)) + 1
    original = b @ a
    np.testing.assert_allclose(edits['arm-output'][1] @ edits['arm-output'][0], (np.eye(40)-u[:,:k]@u[:,:k].T)@original, atol=2e-6)
    np.testing.assert_allclose(edits['arm-input'][1] @ edits['arm-input'][0], original@(np.eye(40)-vt[:k].T@vt[:k]), atol=2e-6)
    for control, cut in [('output-magnitude', 'output'), ('input-magnitude', 'input')]:
        ca, cb = edits['arm-'+control]; xa, xb = edits['arm-'+cut]
        np.testing.assert_allclose(np.linalg.norm(cb@ca), np.linalg.norm(xb@xa), rtol=1e-6)
    assert evidence['spectrum']['k50'] == k
    assert len(edits) == 7


def test_catalogue_is_arm_parameterized_and_rejects_unsafe_names():
    variants = variant_catalogue(['a', 'b'])
    assert len(variants) == 16
    assert variants['a-original']['operation'] == 'original'
    with pytest.raises(ValueError): variant_catalogue(['../escape'])
    with pytest.raises(ValueError): variant_catalogue(['a', 'a'])


def test_build_resumes_sealed_units_and_rejects_changed_payload(tmp_path):
    from pilot_eval.sampling_stage0_cuts import build_modules
    rng = np.random.default_rng(4)
    adapters = {'a': {m: (rng.normal(size=(1,40)).astype('f4'), rng.normal(size=(40,1)).astype('f4')) for m in ['one', 'two']}}
    matrices = {m: rng.normal(size=(40,40)).astype('f4') for m in ['one', 'two']}
    calls = []
    def load(m):
        calls.append(m)
        return matrices[m]
    assert build_modules(tmp_path, ['one','two'], adapters, load, identity='abc', limit=1)['state'] == 'incomplete'
    assert calls == ['one']
    assert build_modules(tmp_path, ['one','two'], adapters, load, identity='abc')['state'] == 'complete'
    assert calls == ['one', 'two']
    unit = next((tmp_path/'units').glob('*/factors.npz'))
    unit.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        build_modules(tmp_path, ['one','two'], adapters, load, identity='abc')

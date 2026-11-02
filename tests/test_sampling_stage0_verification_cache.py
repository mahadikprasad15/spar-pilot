from types import SimpleNamespace

import pytest


def test_operation_reuses_success_and_refreshes_after_exit(tmp_path):
    from pilot_eval.sampling_stage0_verification_cache import verification_cache
    source = tmp_path / 'source'
    source.write_text('valid')
    calls = []
    def verify(path, output_root):
        calls.append(path)
        if path.read_text() != 'valid':
            raise ValueError('changed')
        return {'verified': True}
    module = SimpleNamespace(verify=verify)
    with verification_cache([(module, 'verify')]):
        assert module.verify(source, tmp_path)['verified']
        assert module.verify(path=source, output_root=tmp_path)['verified']
        assert len(calls) == 1
    assert module.verify is verify
    source.write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        with verification_cache([(module, 'verify')]):
            module.verify(source, tmp_path)
    assert module.verify is verify
    assert len(calls) == 2


def test_failed_checks_are_not_cached(tmp_path):
    from pilot_eval.sampling_stage0_verification_cache import verification_cache
    calls = []
    def verify(path, output_root):
        calls.append(path)
        raise ValueError('bad')
    module = SimpleNamespace(verify=verify)
    with verification_cache([(module, 'verify')]):
        for _ in range(2):
            with pytest.raises(ValueError, match='bad'):
                module.verify(tmp_path, tmp_path)
    assert len(calls) == 2

"""Shared offline attempt lifecycle, sealed-unit reuse rule, hash reuse and source roots."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

from pilot_eval.run import _write_json
from pilot_eval.stage0_evaluation import _attempt_timing
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen


class IntegrityFailure(ValueError):
    """A model-integrity check failed, so another full check cannot change the outcome."""


def attempt_start(attempt, required):
    """Apply the Stage 0 reuse rule: a missing closing check is not a failed one.

    Each sealed unit carries its own integrity proof. An attempt that was killed or
    ran out of time therefore keeps its units, while a failed or altered closing
    record still invalidates them. Elapsed-time accounting stays with the reservation.
    """
    final = attempt/'final-integrity.json'
    timing = _attempt_timing(attempt, required=False)
    recorded = file_hash(final) if final.exists() else None
    if final.exists() and not all(json.loads(final.read_text()).get(key) is True for key in required):
        raise ValueError('A failed full integrity check invalidates units from this attempt')
    if timing is not None and timing.get('final_integrity_sha256') != recorded:
        raise ValueError('Attempt final-integrity record changed')
    return file_hash(attempt/'start.json')


@contextmanager
def attempt_scope(directory, reservation, runtime, remaining, *, real, on_failure):
    """Run one bounded attempt and always leave its timing record behind."""
    path = directory/'attempts'/uuid.uuid4().hex
    _save_frozen(path/'start.json', dict(reservation_id=reservation['reservation_id'], runtime_sha256=_hash(runtime)))
    scope = SimpleNamespace(path=path, engine=None, new=0, started=time.monotonic())
    try:
        with deadline_guard(remaining, real):
            yield scope
    except (Exception, KeyboardInterrupt) as exc:
        left = remaining-(time.monotonic()-scope.started)
        if scope.engine is not None and not isinstance(exc, IntegrityFailure) and left > 0:
            try:
                with deadline_guard(left, real):
                    _write_json(path/'final-integrity.json', scope.engine.check_integrity(full=True))
            except Exception:
                pass
        on_failure(exc)
        raise
    finally:
        try:
            if scope.engine is not None:
                scope.engine.close()
        finally:
            final = path/'final-integrity.json'
            _write_json(path/'timing.json', dict(seconds=time.monotonic()-scope.started, newly_sealed_units=scope.new,
                reservation_id=reservation['reservation_id'], start_sha256=file_hash(path/'start.json'),
                final_integrity_sha256=file_hash(final) if final.exists() else None,
                usage_reconciled=False, reservation_stays_charged=True))
            _write_json(path/'timing.complete.json', dict(sha256=file_hash(path/'timing.json')))


_pinned = {}


def pinned_hash(path):
    """Hash a large pinned file once per process while its size and modified time hold."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _pinned:
        _pinned[key] = file_hash(path)
    return _pinned[key]


def portable_root(root, source):
    """Store a source root relative to the artifact root whenever it lies inside it."""
    root, source = Path(root).resolve(), Path(source).resolve()
    return str(source.relative_to(root)) if source.is_relative_to(root) else str(source)


def resolve_root(root, value):
    return (Path(root)/value).resolve()

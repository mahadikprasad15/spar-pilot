"""Reuse read-only provenance checks only within one Stage 0 source audit.

The source tree must not be concurrently modified during an audit. No result or
exception is retained between audits. Historical verifier modules remain intact.
"""
import copy
import inspect
import threading
from contextlib import contextmanager
from pathlib import Path

_lock = threading.RLock()


@contextmanager
def verification_cache(targets):
    """Temporarily wrap specified read-only functions; restore on every exit."""
    with _lock:
        owner = threading.get_ident()
        saved = []
        cache = {}
        try:
            for module, name in targets:
                original = getattr(module, name)
                signature = inspect.signature(original)
                def wrapper(*args, _original=original, _signature=signature, **kwargs):
                    # Calls from other threads never share this operation's snapshot.
                    if threading.get_ident() != owner:
                        return _original(*args, **kwargs)
                    bound = _signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = list(bound.arguments.values())
                    root = Path(values[1]).resolve()
                    path = Path(values[0])
                    path = (path if path.is_absolute() else root / path).resolve()
                    key = (_original, str(path), str(root), *values[2:])
                    if key not in cache:
                        cache[key] = copy.deepcopy(_original(*args, **kwargs))
                    return copy.deepcopy(cache[key])
                saved.append((module, name, original))
                setattr(module, name, wrapper)
            yield
        finally:
            for module, name, original in reversed(saved):
                setattr(module, name, original)


def provenance_targets():
    from pilot_eval import sampling_sft as source
    from pilot_eval import sampling_sft_training_data as data
    from pilot_eval import sampling_sft_review as review
    from pilot_eval import sampling_sft_generation as generation
    from pilot_eval import sampling_sft_pilot as pilot
    from pilot_eval import sampling_sft_instrument as instrument
    return [(module, name) for module, names in [
        (source, ('load', 'verify')), (data, ('load', 'verify')),
        (review, ('load', 'verify')), (generation, ('load', 'verify')),
        (pilot, ('load',)), (instrument, ('load',))]
        for name in names]

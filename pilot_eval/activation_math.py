"""FP64 sufficient summaries; shared by the instrument and CPU reporting."""

import numpy as np


def _inputs(ordinary, change, mask):
    ordinary, change = np.asarray(ordinary, dtype=np.float64), np.asarray(change, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if ordinary.shape != change.shape or ordinary.ndim != 3 or mask.shape != ordinary.shape[:2]:
        raise ValueError('activation/mask shapes disagree')
    if not np.isfinite(ordinary).all() or not np.isfinite(change).all():
        raise ValueError('nonfinite activations')
    return ordinary, change, mask


def block_summary(baseline, delta, mask):
    """Return per-example sums for one block/view; never average examples early."""
    baseline, delta, mask = _inputs(baseline, delta, mask)
    return {'count': mask.sum(axis=1),
            'base_sum': (baseline * mask[..., None]).sum(axis=1),
            'delta_sum': (delta * mask[..., None]).sum(axis=1),
            'base_norm_sum': (np.linalg.norm(baseline, axis=-1) * mask).sum(axis=1),
            'delta_norm_sum': (np.linalg.norm(delta, axis=-1) * mask).sum(axis=1)}


def module_summary(ordinary, direct, mask):
    """Keep zero denominator tokens in primary sums; conditional ratios separate."""
    ordinary, direct, mask = _inputs(ordinary, direct, mask)
    base, delta = np.linalg.norm(ordinary, axis=-1), np.linalg.norm(direct, axis=-1)
    defined = mask & (base != 0)
    ratios = np.divide(delta, base, out=np.zeros_like(base), where=defined)
    return {'count': mask.sum(axis=1), 'base_norm_sum': (base * mask).sum(axis=1),
            'delta_norm_sum': (delta * mask).sum(axis=1),
            'ratio_sum': ratios.sum(axis=1), 'defined_count': defined.sum(axis=1)}


def derive_measurements(summary, weighting):
    """Derive one block/module view from per-example sufficient summaries."""
    if weighting not in ('token', 'example'):
        raise ValueError('unknown weighting')
    count = np.asarray(summary['count'])
    nonempty = count > 0

    def mean(values, counts=count):
        values, counts = np.asarray(values), np.asarray(counts)
        valid = counts > 0
        if not valid.any():
            return None
        if weighting == 'token':
            return values[valid].sum(axis=0) / counts[valid].sum()
        divisor = counts[valid].reshape((-1,) + (1,) * (values.ndim - 1))
        return (values[valid] / divisor).mean(axis=0)

    base, change = mean(summary['base_norm_sum']), mean(summary['delta_norm_sum'])
    result = {'token_count': int(count.sum()), 'example_count': int(nonempty.sum()),
              'empty_example_count': int((~nonempty).sum()),
              'mean_base_norm': None if base is None else float(base),
              'mean_delta_norm': None if change is None else float(change)}
    numerator = change
    if 'delta_sum' in summary:
        vector, base_vector = mean(summary['delta_sum']), mean(summary['base_sum'])
        numerator = None if vector is None else float(np.linalg.norm(vector))
        alternative = None if base_vector is None else float(np.linalg.norm(base_vector))
        result.update(mean_delta=None if vector is None else vector.tolist(),
                      mean_base=None if base_vector is None else base_vector.tolist(),
                      mean_base_vector_norm=alternative,
                      relative_write_alternative=None if not alternative else numerator / alternative)
    result['relative_write'] = None if base is None or base == 0 else float(numerator / base)
    if 'ratio_sum' in summary:
        defined = np.asarray(summary['defined_count'])
        ratio = mean(summary['ratio_sum'], defined)
        result.update(mean_token_ratio=None if ratio is None else float(ratio),
                      defined_token_count=int(defined.sum()),
                      undefined_token_count=int(count.sum() - defined.sum()),
                      ratio_example_count=int((defined > 0).sum()))
    return result

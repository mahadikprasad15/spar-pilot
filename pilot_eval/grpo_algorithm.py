"""Pilot 4 GRPO window contract, pinned TRL loss and gradient diagnostics.

Eight prompt groups are normalized together before any memory microbatching.
Direct backward on DAPO sums contributions; never divide again by accumulation.
"""
import math
import numpy as np


def group_advantages(rewards):
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or len(values) % 8 or not np.isfinite(values).all():
        raise ValueError('finite rewards in complete groups of eight required')
    groups = values.reshape(-1, 8)
    return ((groups - groups.mean(1, keepdims=True)) /
            (groups.std(1, ddof=1, keepdims=True) + 1e-4)).reshape(-1).tolist()

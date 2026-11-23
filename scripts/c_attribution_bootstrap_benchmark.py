"""Benchmark the C-attribution bootstraps at production shape on synthetic sums (ticket 11's benchmark tiers).

Calls the real functions the aggregate stage calls (``resample_counts``, ``neuron_bootstrap`` per tier,
``adapter_bootstrap``) on random sums of the production shape and records wall seconds, tracemalloc peak and
process max RSS per step. The inputs are synthetic: only timing and memory are meaningful, not the intervals.

    python scripts/c_attribution_bootstrap_benchmark.py --questions 150 --out <file>.json
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from pilot_eval import c_attribution_stability as st


def run(*, questions, layers, width, adapters, selected, resamples, seed, tiers):
    rng = np.random.default_rng(0)
    reads = rng.standard_normal((questions, layers, width)) * 1e-4
    out = dict(shape=dict(questions=questions, layers=layers, width=width, adapters=adapters, selected=selected),
               resamples=resamples, host=dict(machine=platform.machine(), processor=platform.processor(),
                                              cpus=os.cpu_count(), numpy=np.__version__),
               note='synthetic sums at production shape: timing and memory only', steps={})
    t0 = time.perf_counter()
    counts = st.resample_counts(questions, B=resamples, seed=seed)
    out['steps']['resample_counts'] = dict(seconds=time.perf_counter() - t0)
    pairs = [(int(li), int(j)) for li, j in zip(rng.integers(0, layers, selected), rng.integers(0, width, selected))]
    bench = st.neuron_bootstrap(reads, counts, selected=pairs, tiers=tiers, benchmark=True)['benchmark']
    out['steps'].update({f'neurons:{k}': v for k, v in bench['tiers'].items()})
    S = {arm: (list(range(questions)), rng.standard_normal((questions, adapters)) * 1e-3)
         for arm in ('online-grpo', 'offline-grpo')}
    t0 = time.perf_counter()
    st.adapter_bootstrap(S, B=resamples, seed=seed)
    out['steps']['adapter_bootstrap (2 arms + paired)'] = dict(seconds=time.perf_counter() - t0,
                                                               max_rss_bytes=st._max_rss())
    per_arm = sum(out['steps'][k]['seconds'] for k in ('resample_counts', 'neurons:per_layer', 'neurons:selected'))
    out['production_estimate_seconds'] = dict(
        per_arm_neurons=per_arm, adapters=out['steps']['adapter_bootstrap (2 arms + paired)']['seconds'],
        total=2 * per_arm + out['steps']['adapter_bootstrap (2 arms + paired)']['seconds'],
        note="production runs tiers per_layer and selected for each arm, then the paired adapter bootstrap; "
             "'all' is reported for scale only")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--questions', type=int, default=150)
    p.add_argument('--layers', type=int, default=28)
    p.add_argument('--width', type=int, default=8960)        # Qwen2.5-1.5B MLP intermediate size
    p.add_argument('--adapters', type=int, default=196)      # 28 layers x 7 projections
    p.add_argument('--selected', type=int, default=512)      # reads in target and control groups, upper bound
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--tiers', default='per_layer,selected,all')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    result = run(questions=a.questions, layers=a.layers, width=a.width, adapters=a.adapters, selected=a.selected,
                 resamples=a.resamples, seed=a.seed, tiers=tuple(a.tiers.split(',')))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()

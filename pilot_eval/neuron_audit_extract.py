"""Compact per-text extract of sealed neuron-audit block units, for off-Pod analysis and plotting.

Reads sealed units (hash-checked by BlockStore.read_block) and writes, per arm and layer, one float32 matrix
[examples, neurons] for each requested statistic, rows in frozen sequences order (discovery + heldout only;
calibration rows are skipped). Resumable: files already listed as written in status.json are reused.

  python -m pilot_eval.neuron_audit_extract --stores <blocks> --sequences_run <seq_dir> --out_dir <out> \
      --layers 0-27 [--arms untuned,gold-sft,...] [--stats mean_abs_z,mean_abs_term] [--group response_content]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from pilot_eval.neuron_audit_blocks import BlockStore

SPLITS = ('discovery', 'heldout')
DEFAULT_ARMS = ['untuned', 'gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2',
                'sampling-sft-k5']


def _atomic_bytes(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_npy(path, array):
    tmp = path.with_name(path.name + '.tmp.npy')
    np.save(tmp, array)
    os.replace(tmp, path)


def _where(store_roots):
    """(arm, example) -> (store, unit key, row) over sealed units only; duplicates refused."""
    where = {}
    for root in store_roots:
        store = BlockStore(root)
        for key, state in store.status().items():
            if state != 'sealed':
                continue
            index = store.index(key)
            arm_rows = where.setdefault(index['identity']['arm'], {})
            for row, example in enumerate(index['example_ids']):
                if example in arm_rows:
                    raise ValueError(f"arm {index['identity']['arm']} example {example} appears twice")
                arm_rows[example] = (store, key, row)
    return where


def extract(store_roots, sequences, *, out_dir, arms, layers, stats, group='response_content'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in sequences if r['split'] in SPLITS]
    ids = [r['id'] for r in rows]
    index = dict(group=group, example_ids=ids, groups=[r['group'] for r in rows], splits=[r['split'] for r in rows],
                 arms=list(arms), layers=list(layers), stats=list(stats), dtype='float32',
                 note='per-text mean over the group tokens; rows in frozen sequences order')
    _atomic_bytes(out_dir / 'index.json', (json.dumps(index, indent=2) + '\n').encode())
    status_path = out_dir / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(files={})
    if status.get('example_ids_sha256') not in (None, hashlib.sha256('\n'.join(ids).encode()).hexdigest()):
        raise ValueError('existing extract was built from different rows; use a new out_dir')
    status['example_ids_sha256'] = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    where = _where(store_roots)
    written = reused = 0
    for arm in arms:
        if arm not in where:
            raise ValueError(f'arm {arm} has no sealed units')
        missing = [i for i in ids if i not in where[arm]]
        if missing:
            raise ValueError(f'arm {arm} lacks {len(missing)} rows (first: {missing[0]})')
        arm_dir = out_dir / arm
        arm_dir.mkdir(exist_ok=True)
        counts = [int(where[arm][i][0].index(where[arm][i][1])['counts'][group][where[arm][i][2]]) for i in ids]
        _atomic_bytes(arm_dir / 'counts.json', json.dumps(counts).encode())
        arm_stats = [s for s in stats if not (arm == 'untuned' and 'term' in s)]  # untuned has no adapter
        for layer in layers:
            for stat in arm_stats:
                name = f'{arm}/{stat}_layer{layer:02d}.npy'
                path = out_dir / name
                if name in status['files'] and path.exists():
                    reused += 1
                    continue
                cache, out = {}, []
                for i in ids:
                    store, key, row = where[arm][i]
                    if key not in cache:
                        cache[key] = store.read_block(key, group, layer, stat)
                    out.append(cache[key][row])
                array = np.stack(out).astype(np.float32)
                _atomic_npy(path, array)
                status['files'][name] = dict(shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())
                _atomic_bytes(status_path, (json.dumps(status, indent=2) + '\n').encode())
                written += 1
                print(json.dumps(dict(kind='extracted', file=name)), flush=True)
    status.update(complete=True, written=written, reused=reused)
    _atomic_bytes(status_path, (json.dumps(status, indent=2) + '\n').encode())
    return status


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stores', nargs='+', required=True)
    p.add_argument('--sequences_run', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--layers', required=True, help='e.g. 0-27')
    p.add_argument('--arms', default=','.join(DEFAULT_ARMS))
    p.add_argument('--stats', default='mean_abs_z')
    p.add_argument('--group', default='response_content')
    a = p.parse_args(argv)
    seq = [json.loads(l) for l in (Path(a.sequences_run) / 'sequences.jsonl').read_text().splitlines()]
    lo, hi = (int(x) for x in a.layers.split('-'))
    status = extract(a.stores, seq, out_dir=a.out_dir, arms=a.arms.split(','), layers=list(range(lo, hi + 1)),
                     stats=a.stats.split(','), group=a.group)
    print('EXTRACT', 'complete' if status.get('complete') else 'incomplete', 'written', status['written'],
          'reused', status['reused'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

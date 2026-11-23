"""Local-only synthetic journal comparison; never alters experiment runs."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from pilot_eval.sampling_sft_buffered_journal import BufferedJournal


def benchmark(directory, *, rows, token_count, max_bytes):
    directory = Path(directory)
    if rows <= 0 or token_count <= 0:
        raise ValueError('rows and token count must be positive')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError('benchmark directory must be empty')
    values = [dict(problem_id=i, block=3, step=i % 5, old_ids=list(range(token_count)),
                   proposal_ids=list(range(token_count)), old_score=-0.62,
                   proposal_score=-0.61, accepted=True) for i in range(rows)]
    results = {}
    for mode in ('old', 'buffered'):
        path = directory / (mode + '.jsonl')
        metrics = dict(encode_seconds=0.0, write_seconds=0.0, sync_seconds=0.0,
                       flush_seconds=0.0, write_count=0, fsync_count=0)
        start = time.monotonic()
        if mode == 'old':
            # Same open/JSON/write/flush/fsync/close sequence as instrument.append.
            for value in values:
                with path.open('a' if path.exists() else 'x', encoding='utf-8') as stream:
                    t = time.monotonic(); line = json.dumps(value, allow_nan=False) + '\n'; metrics['encode_seconds'] += time.monotonic()-t
                    t = time.monotonic(); stream.write(line); metrics['write_seconds'] += time.monotonic()-t; metrics['write_count'] += 1
                    t = time.monotonic(); stream.flush(); metrics['flush_seconds'] += time.monotonic()-t
                    t = time.monotonic(); os.fsync(stream.fileno()); metrics['sync_seconds'] += time.monotonic()-t; metrics['fsync_count'] += 1
        else:
            # Wrap the prototype's actual writes and fsync to measure its calls.
            real_write, real_sync, real_dumps = os.write, os.fsync, json.dumps
            def write(fd, data):
                t = time.monotonic()
                try: return real_write(fd, data)
                finally: metrics['write_seconds'] += time.monotonic()-t; metrics['write_count'] += 1
            def sync(fd):
                t = time.monotonic()
                try: return real_sync(fd)
                finally: metrics['sync_seconds'] += time.monotonic()-t; metrics['fsync_count'] += 1
            def dumps(*args, **kwargs):
                t = time.monotonic()
                try: return real_dumps(*args, **kwargs)
                finally: metrics['encode_seconds'] += time.monotonic()-t
            try:
                os.write, os.fsync, json.dumps = write, sync, dumps
                with BufferedJournal(path, max_bytes=max_bytes) as journal:
                    for value in values: journal.append(value)
                    journal.sync()
            finally:
                os.write, os.fsync, json.dumps = real_write, real_sync, real_dumps
        metrics['wall_seconds'] = time.monotonic()-start
        metrics['bytes'] = path.stat().st_size
        metrics['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Isolated simple state-file write, not a complete production seal.
        t = time.monotonic()
        with (directory/(mode+'-checkpoint.json')).open('x') as f:
            json.dump({'next_block': 1, 'rows': rows}, f); f.flush(); os.fsync(f.fileno())
        metrics['checkpoint_seconds'] = time.monotonic()-t
        results[mode] = metrics
    return dict(schema_version=1, platform=platform.platform(), directory=str(directory.resolve()),
                rows=rows, token_count=token_count, max_bytes=max_bytes,
                equal_bytes=results['old']['sha256']==results['buffered']['sha256'], **results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, help='new empty local benchmark directory')
    p.add_argument('--rows', type=int, default=2500)
    p.add_argument('--token-count', type=int, default=256)
    p.add_argument('--max-bytes', type=int, default=1048576)
    a = p.parse_args()
    target = Path(a.output_dir)
    target.mkdir(parents=True, exist_ok=False)
    result = benchmark(target, rows=a.rows, token_count=a.token_count, max_bytes=a.max_bytes)
    with (target/'summary.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

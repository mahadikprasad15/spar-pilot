"""Ticket 03a immutable dense FP32 block store for the joint neuron/adapter audit.

One sealed *unit* per (arm, batch/row range). A unit's ``blocks.f32`` holds the logical
blocks [rows, intermediate_size] for every (position group, layer, statistic) back to
back; ``index.json`` maps each logical block to its byte offset, shape and SHA-256, and
rows to example IDs and per-group token counts. Per-token reads c_t live in separate
compact arrays with (row, position, group) mapping. Write magnitude |s| ||b|| |c_t| is
derived later from c_t and the recorded s and ||b||, not stored.

A unit is published by writing ``seal.json`` last. A sealed unit is never rewritten.
An unsealed directory is interrupted work: it is moved aside as evidence, not reused.
A sealed unit whose hashes fail is *corrupt*, which is reported distinctly.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

UNIT_SCHEMA = 'joint-neuron-adapter-audit/block-unit-v1'
TOKEN_FILES = {'ct': '<f4', 'token_rows': '<i4', 'token_positions': '<i4', 'token_groups': '|u1'}


class CorruptUnit(ValueError):
    pass


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_write(path, data):
    with Path(path).open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


class BlockStore:
    def __init__(self, root):
        self.root = Path(root)
        self.units = self.root / 'units'
        self._verified = set()  # hash each sealed unit at most once per process

    def unit_dir(self, key):
        if not key or '/' in key or key.startswith('.'):
            raise ValueError('unit key must be a simple name')
        return self.units / key

    def state(self, key):
        directory = self.unit_dir(key)
        if (directory / 'seal.json').is_file():
            return 'sealed'
        return 'interrupted' if directory.exists() else 'missing'

    def write_unit(self, key, *, identity, example_ids, counts, blocks, tokens=None, meta=None):
        """Seal one unit. ``blocks`` maps (group, layer, stat) -> FP32 array [rows, J].

        Returns 'written' or 'reused' (sealed unit with identical identity, verified).
        """
        directory = self.unit_dir(key)
        state = self.state(key)
        if state == 'sealed':
            self.verify_unit(key)
            seal = json.loads((directory / 'seal.json').read_text())
            if seal['identity'] != identity:
                raise ValueError(f'Sealed unit {key} has a different identity; refusing to rewrite.')
            return 'reused'
        if state == 'interrupted':
            evidence = self.root / 'interrupted' / f'{key}-{time.time_ns()}'
            evidence.parent.mkdir(parents=True, exist_ok=True)
            os.replace(directory, evidence)
        rows = len(example_ids)
        if len(set(example_ids)) != rows or rows == 0:
            raise ValueError('unit needs unique, nonempty example IDs')
        counts = {g: [int(x) for x in v] for g, v in counts.items()}
        if any(len(v) != rows for v in counts.values()):
            raise ValueError('counts must have one entry per row')
        for (group, layer, stat), array in blocks.items():
            array = np.asarray(array)
            if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != rows:
                raise ValueError(f'block {(group, layer, stat)} must be FP32 [rows, J]')
            if group not in counts:
                raise ValueError(f'block group {group} has no counts')
            empty = np.asarray(counts[group]) == 0
            # Undefined means (zero counted tokens) are NaN, never zero; every other value is finite.
            if not np.isfinite(array[~empty]).all() or not np.isnan(array[empty]).all():
                raise ValueError(f'block {(group, layer, stat)} must be finite except all-NaN zero-count rows')
        directory.mkdir(parents=True)
        entries, offset = [], 0
        with (directory / 'blocks.f32').open('wb') as stream:
            for (group, layer, stat), array in sorted(blocks.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
                array = np.asarray(array)
                raw = np.ascontiguousarray(array, dtype='<f4').tobytes()
                stream.write(raw)
                entries.append(dict(group=group, layer=int(layer), stat=stat, offset=offset,
                                    shape=list(array.shape), sha256=hashlib.sha256(raw).hexdigest()))
                offset += len(raw)
            stream.flush()
            os.fsync(stream.fileno())
        token_index = None
        if tokens is not None:
            token_index = {}
            for name, dtype in TOKEN_FILES.items():
                array = np.ascontiguousarray(tokens[name], dtype=dtype)
                _fsync_write(directory / f'{name}.bin', array.tobytes())
                token_index[name] = dict(dtype=dtype, shape=list(array.shape))
        index = dict(schema=UNIT_SCHEMA, key=key, identity=identity, example_ids=list(example_ids),
                     counts=counts, blocks=entries, tokens=token_index, meta=meta or {})
        _fsync_write(directory / 'index.json', _json_bytes(index))
        files = {p.name: dict(sha256=_sha(p), size=p.stat().st_size) for p in sorted(directory.iterdir())}
        tmp = directory / 'seal.json.tmp'
        _fsync_write(tmp, _json_bytes(dict(identity=identity, files=files)))
        os.replace(tmp, directory / 'seal.json')
        self._verified.add(key)
        return 'written'

    def verify_unit(self, key, *, force=False, use_listing=True):
        """Verify every sealed file. ``use_listing=False`` checks only explicit seal paths: on a Runpod
        Global Volume directory listings may be up to a minute stale, so extra-file detection is skipped there."""
        directory = self.unit_dir(key)
        if self.state(key) != 'sealed':
            raise FileNotFoundError(f'unit {key} is {self.state(key)}')
        if key in self._verified and not force:
            return True
        try:
            seal = json.loads((directory / 'seal.json').read_text())
        except json.JSONDecodeError as exc:
            raise CorruptUnit(f'unit {key} seal is unreadable (torn write?)') from exc
        if use_listing:
            present = {p.name for p in directory.iterdir()} - {'seal.json'}
            if present != set(seal['files']):
                raise CorruptUnit(f'unit {key} file set differs from its seal')
        for name, record in seal['files'].items():
            path = directory / name
            if not path.is_file() or path.stat().st_size != record['size'] or _sha(path) != record['sha256']:
                raise CorruptUnit(f'unit {key} file {name} failed its hash')
        self._verified.add(key)
        return True

    def index(self, key):
        self.verify_unit(key)
        return json.loads((self.unit_dir(key) / 'index.json').read_text())

    def read_block(self, key, group, layer, stat):
        index = self.index(key)
        for entry in index['blocks']:
            if (entry['group'], entry['layer'], entry['stat']) == (group, layer, stat):
                rows, width = entry['shape']
                array = np.fromfile(self.unit_dir(key) / 'blocks.f32', dtype='<f4', count=rows * width,
                                    offset=entry['offset']).reshape(rows, width)
                if hashlib.sha256(array.tobytes()).hexdigest() != entry['sha256']:
                    raise CorruptUnit(f'block {(group, layer, stat)} in {key} failed its hash')
                return array
        raise KeyError((group, layer, stat))

    def read_tokens(self, key):
        index = self.index(key)
        if index['tokens'] is None:
            return None
        return {name: np.fromfile(self.unit_dir(key) / f'{name}.bin', dtype=spec['dtype']).reshape(spec['shape'])
                for name, spec in index['tokens'].items()}

    def status(self):
        keys = sorted(p.name for p in self.units.iterdir()) if self.units.exists() else []
        return {k: self.state(k) for k in keys}


def mirror_store(local_root, volume_root, *, keys=None, source_listing=True):
    """Copy sealed local units to a Global-Volume store without relying on atomic rename or listings.

    Data files are copied with shutil.copyfile, the seal file last; the copy is then verified from
    explicit seal paths. A torn or mismatched copy is re-copied once, then reported as an error.
    Returns {key: 'mirrored' | 'already-mirrored'}.
    """
    import shutil
    local, volume = BlockStore(local_root), BlockStore(volume_root)
    keys = keys if keys is not None else [k for k, s in local.status().items() if s == 'sealed']
    out = {}
    for key in keys:
        local.verify_unit(key, force=True, use_listing=source_listing)
        source, target = local.unit_dir(key), volume.unit_dir(key)
        seal_bytes = (source / 'seal.json').read_bytes()
        if (target / 'seal.json').is_file() and (target / 'seal.json').read_bytes() == seal_bytes:
            try:
                volume.verify_unit(key, force=True, use_listing=False)
                out[key] = 'already-mirrored'
                continue
            except CorruptUnit:
                pass
        for attempt in range(2):
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'seal.json').exists():
                (target / 'seal.json').unlink()  # unseal before rewriting data
            for name in json.loads(seal_bytes)['files']:
                shutil.copyfile(source / name, target / name)
            with (target / 'seal.json').open('wb') as stream:
                stream.write(seal_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                volume.verify_unit(key, force=True, use_listing=False)
                out[key] = 'mirrored'
                break
            except CorruptUnit:
                if attempt == 1:
                    raise
    return out

"""Unintegrated block journal prototype; the caller owns sealing and recovery.

Only a successful sync permits a block seal. A failed or interrupted partial
block must be preserved and discarded as a whole before replay. This module is
not wired into the frozen generation instrument.
"""
import json
import os


class BufferedJournal:
    def __init__(self, path, *, max_bytes=1024 * 1024):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('max_bytes must be positive')
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        self.max_bytes = max_bytes
        self.buffer = bytearray()
        self.failed = False
        self.closed = False

    def _check(self):
        if self.failed or self.closed:
            raise RuntimeError('journal is failed or closed')

    def append(self, value):
        self._check()
        try:
            encoded = (json.dumps(value, allow_nan=False) + '\n').encode('utf-8')
            if len(self.buffer) + len(encoded) > self.max_bytes:
                self._write()
            self.buffer.extend(encoded)
            if len(self.buffer) >= self.max_bytes:
                self._write()
        except BaseException:
            self.failed = True
            raise

    def _write(self):
        try:
            data = bytes(self.buffer)
            offset = 0
            while offset < len(data):
                count = os.write(self.fd, data[offset:])
                if count <= 0:
                    raise OSError('short write made no progress')
                offset += count
            self.buffer.clear()
        except BaseException:
            self.failed = True
            raise

    def sync(self):
        self._check()
        try:
            self._write()
            os.fsync(self.fd)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BlockWriter:
    """Shared scientific-record callback for the v2 pilot and generation."""
    def __init__(self, partial, *, max_bytes, metadata=None):
        self.partial = partial
        self.max_bytes = max_bytes
        self.metadata = dict(metadata or {})
        self.journal = None

    def record(self, value):
        self.partial.mkdir(exist_ok=True)
        value = json.loads(json.dumps(value))
        value.pop('replay', None)
        for output in value.get('outputs', []):
            output.pop('num_cached_tokens', None)
        value.update(self.metadata)
        if self.journal is None:
            self.journal = BufferedJournal(self.partial/'scientific.jsonl', max_bytes=self.max_bytes)
        self.journal.append(value)
        return value

    def sync_block(self):
        if self.journal is None:raise ValueError('Missing scientific journal.')
        self.journal.sync()
        self.journal.close()
        self.journal = None

    def close(self):
        if self.journal is not None:
            self.journal.close()
            self.journal = None

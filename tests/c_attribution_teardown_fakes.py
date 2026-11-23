"""Module-level teardown fakes: the chain runs its volume copy in a child process, so they must be importable."""
import os
import time
from pathlib import Path


def slow_copy(local_root, volume_root, sealed_only=True, skip_present=False):
    """Record this process's PID, then hang (a stuck FUSE copy)."""
    if skip_present:  # the in-process mirror at a unit boundary: nothing to simulate
        return dict(verified=True, copied=[], problems=[])
    Path(local_root, 'copy.pid').write_text(str(os.getpid()))
    time.sleep(120)
    return dict(verified=True, copied=[], problems=[])


def empty_copy(local_root, volume_root, sealed_only=True, skip_present=False):
    """Copies nothing and reports no problems (production-1 called that verified)."""
    return dict(verified=True, copied=[], problems=[])

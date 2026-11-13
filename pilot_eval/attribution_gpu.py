"""Ticket 05: GPU utilization measurement for adapter attribution (spec §GPU utilization).

Performance diagnostics only: busy %, memory and power from NVML at 1 Hz (a summary needs >= 30 s of
samples), synchronized throughput, padding fraction and an MFU estimate. None of these is a validity
gate or a reason to change a qualified configuration.

FLOP model: forward ~ 2N per unpadded token; each backward without weight gradients ~ 2N per token.
Excludes attention-score FLOPs and padding. Denominator: the measured device's FP32 dense peak.
"""
import statistics
import threading
import time
from contextlib import contextmanager
from pathlib import Path

MIN_SAMPLE_SECONDS = 30


def _load_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:  # no NVML on CPU or without the package: recorded as unavailable
        return None


class NvmlSampler:
    """1 Hz background sampler; ``nvml`` is injectable (``None`` = unavailable)."""

    def __init__(self, *, nvml='auto', device_index=0, hz=1.0, clock=time.monotonic):
        self.nvml = _load_nvml() if nvml == 'auto' else nvml
        if self.nvml is not None and nvml != 'auto':
            self.nvml.nvmlInit()
        self.handle = self.nvml.nvmlDeviceGetHandleByIndex(device_index) if self.nvml is not None else None
        self.interval, self.clock = 1.0 / hz, clock
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def sample(self, now=None):
        if self.nvml is None:
            return None
        util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        power = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
        record = dict(t=self.clock() if now is None else now, busy=float(util.gpu),
                      memory_fraction=memory.used / memory.total, power_watts=power / 1000.0)
        self.samples.append(record)
        return record

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self):
        if self.nvml is not None:
            self.sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self):
        if self.nvml is None:
            return dict(available=False, reason='NVML unavailable (CPU or no pynvml)', samples=0)
        span = self.samples[-1]['t'] - self.samples[0]['t'] if self.samples else 0.0
        if span < MIN_SAMPLE_SECONDS:
            return dict(available=False, reason=f'fewer than {MIN_SAMPLE_SECONDS} s of samples ({span:.0f} s)',
                        samples=len(self.samples))
        return dict(available=True, samples=len(self.samples), span_seconds=span,
                    median_busy_percent=statistics.median(s['busy'] for s in self.samples),
                    peak_memory_fraction=max(s['memory_fraction'] for s in self.samples),
                    median_power_watts=statistics.median(s['power_watts'] for s in self.samples),
                    targets=dict(median_busy=0.85, peak_memory_range=[0.50, 0.80]), targets_are_gates=False)


@contextmanager
def synchronized(sync, *, now=time.perf_counter):
    """Time a block between two device synchronizations (call after warm-up)."""
    record = {}
    sync()
    start = now()
    try:
        yield record
    finally:
        sync()
        record['seconds'] = now() - start


def throughput(*, tokens, seconds):
    if seconds <= 0:
        raise ValueError('positive synchronized time required')
    return dict(tokens=int(tokens), seconds=float(seconds), tokens_per_second=tokens / seconds)


def padding_fraction(lengths, *, padded_width):
    total = padded_width * len(lengths)
    if not total or any(l > padded_width for l in lengths):
        raise ValueError('lengths must fit the padded width')
    return 1.0 - sum(lengths) / total


def mfu(*, parameters, tokens, seconds, peak_flops, backward_passes=0):
    """Estimate: (2N forward + 2N per activation-only backward) x unpadded tokens / (seconds x peak)."""
    if seconds <= 0 or peak_flops <= 0 or parameters <= 0:
        raise ValueError('positive seconds, peak FLOP/s and parameter count required')
    flops = 2.0 * parameters * tokens * (1 + backward_passes)
    return dict(flops=flops, seconds=seconds, peak_flops=peak_flops, mfu=flops / seconds / peak_flops,
                model='forward 2N + 2N per backward without weight gradients, per unpadded token',
                excludes=['attention_scores', 'padding'], estimate=True)


@contextmanager
def profiler_trace(path, *, enabled=True):
    """One torch profiler trace per stage, exported as Chrome JSON."""
    if not enabled:
        yield None
        return
    import torch
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        yield profiler
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))

"""Cross-machine comparison of recomputed JSON-like results (ADR 0015/0016)."""


def agrees(saved, recomputed, *, rtol=1e-12):
    """Exact structure, integers and strings; floats agree to rtol (BLAS/SIMD order differs across CPUs)."""
    if isinstance(saved, float) or isinstance(recomputed, float):
        return (isinstance(saved, float) and isinstance(recomputed, float)
                and (saved == recomputed or abs(saved - recomputed) <= rtol * max(abs(saved), abs(recomputed))))
    if type(saved) != type(recomputed):
        return False
    if isinstance(saved, dict):
        return set(saved) == set(recomputed) and all(agrees(saved[k], recomputed[k], rtol=rtol) for k in saved)
    if isinstance(saved, list):
        return len(saved) == len(recomputed) and all(agrees(a, b, rtol=rtol) for a, b in zip(saved, recomputed))
    return saved == recomputed


def environment(runtime):
    """Runtime fields that must match within a frozen session; the code commit is recorded per unit instead."""
    return {k: v for k, v in runtime.items() if k != 'git_commit'}

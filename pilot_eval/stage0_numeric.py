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


def dependency_fingerprint(names, folder):
    """SHA-256 of each named module and every pilot_eval module it imports, transitively.

    Lazy imports inside functions count. A loader or backend change therefore changes the
    identity even when only the code commit would otherwise differ.
    """
    import ast
    from pathlib import Path
    from pilot_eval.training import file_hash
    folder = Path(folder); pending = list(names); seen = {}
    while pending:
        name = pending.pop()
        if name in seen or not (folder / name).is_file():
            continue
        seen[name] = file_hash(folder / name)
        for node in ast.walk(ast.parse((folder / name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == 'pilot_eval':
                    pending += [alias.name + '.py' for alias in node.names]
                elif node.module.startswith('pilot_eval.'):
                    pending.append(node.module.split('.')[1] + '.py')
            elif isinstance(node, ast.Import):
                pending += [a.name.split('.')[1] + '.py' for a in node.names if a.name.startswith('pilot_eval.')]
    return dict(sorted(seen.items()))


def require_eager_attention(model):
    """Inspect the loaded model rather than trusting the declared runtime field."""
    actual = getattr(model.config, '_attn_implementation', None)
    if actual != 'eager':
        raise ValueError(f'Loaded model uses {actual!r} attention; the frozen protocol requires eager.')

"""Evidence and child-process controls for the bounded IEEE diagnostic."""
import ast
from pathlib import Path
from pilot_eval.training import file_hash


def inspect_sources(vllm_root, triton_root, vllm_version, triton_version):
    if vllm_version != '0.11.1':
        raise ValueError('Expected vLLM 0.11.1.')
    files = {'vllm_attention': Path(vllm_root)/'attention/ops/triton_unified_attention.py',
             'triton_core': Path(triton_root)/'language/core.py',
             'triton_knobs': Path(triton_root)/'knobs.py'}
    try:
        texts = {k: p.read_text() for k,p in files.items()}
    except OSError as exc:
        raise ValueError('Missing installed precision source.') from exc
    if ('knobs.language.fp32_default' not in texts['triton_core'] or
            'TRITON_F32_DEFAULT' not in texts['triton_knobs'] or
            'fp32_default' not in texts['triton_knobs']):
        raise ValueError('Installed Triton precision mechanism is missing or ambiguous.')
    tree=ast.parse(texts['vllm_attention'])
    dots=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
          isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and
          n.func.value.id=='tl' and n.func.attr=='dot']
    unqualified=[n for n in dots if len(n.args)==2 and not any(
        k.arg in ('input_precision','allow_tf32') or k.arg is None for k in n.keywords)]
    if not unqualified:
        raise ValueError('No unqualified attention dot calls found.')
    return dict(vllm_version=vllm_version,triton_version=triton_version,
                unqualified_dots=len(unqualified),
                files={k:dict(path=str(p),sha256=file_hash(p)) for k,p in files.items()})

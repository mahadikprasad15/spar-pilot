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


def installed_sources():
    import importlib.metadata as metadata
    v=metadata.distribution('vllm'); t=metadata.distribution('triton')
    return inspect_sources(v.locate_file('vllm'),t.locate_file('triton'),v.version,t.version)


def execution_identity(runtime_hash, plan_path):
    return dict(legacy_runtime_hash=runtime_hash,precision_plan_sha256=file_hash(plan_path),
                production_qualified=False)


def configure_child(plan_path, root, expected_hash=None):
    import json, os, sys
    path=Path(plan_path).resolve(); root=Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_hash(path)!=expected_hash:
        raise ValueError('Invalid precision plan or hash.')
    plan=json.loads(path.read_text())
    if (plan.get('sources')!=installed_sources() or plan.get('precision')!='ieee'
            or plan.get('cache_initially_empty') is not True
            or plan.get('diagnostic_code_sha256')!=file_hash(Path(__file__))):
        raise ValueError('Precision plan source fingerprint differs.')
    cache=Path(plan['triton_cache']).resolve()
    if not cache.is_relative_to(path.parent) or not cache.is_dir() or any(cache.iterdir()):
        raise ValueError('Invalid isolated Triton cache.')
    if any(x == 'triton' or x.startswith('triton.') or x == 'vllm' or x.startswith('vllm.') for x in sys.modules):
        raise ValueError('Precision must be configured before Triton or vLLM import.')
    os.environ['TRITON_F32_DEFAULT']='ieee'
    os.environ['TRITON_CACHE_DIR']=str(cache)
    os.environ['VLLM_CACHE_ROOT']=str(path.parent/'vllm-cache')
    os.environ['SAMPLING_SFT_IEEE_PLAN']=str(path)
    os.environ['SAMPLING_SFT_IEEE_HASH']=expected_hash
    from pilot_eval import sampling_sft_engine as engine
    original=engine.open_engine
    def wrapped(*args):
        import vllm
        llm=vllm.LLM
        def construct(*a,**kw):
            kw['worker_extension_cls']='pilot_eval.sampling_sft_ieee_diagnostic.IEEEWorkerExtension'
            return llm(*a,**kw)
        vllm.LLM=construct
        try:return original(*args)
        finally:vllm.LLM=llm
    engine.open_engine=wrapped


from pilot_eval.sampling_sft_engine import SamplingSFTWorkerExtension


class IEEEWorkerExtension(SamplingSFTWorkerExtension):
    def sampling_sft_worker_evidence(self):
        import json, os
        import triton
        path=Path(os.environ['SAMPLING_SFT_IEEE_PLAN'])
        if file_hash(path)!=os.environ['SAMPLING_SFT_IEEE_HASH']:
            raise ValueError('Worker precision plan changed.')
        plan=json.loads(path.read_text())
        if (os.environ.get('TRITON_F32_DEFAULT')!='ieee' or
            os.environ.get('TRITON_CACHE_DIR')!=plan['triton_cache'] or
            os.environ.get('VLLM_CACHE_ROOT')!=str(path.parent/'vllm-cache') or
            file_hash(Path(__file__))!=plan['diagnostic_code_sha256'] or
            triton.knobs.language.fp32_default!='ieee' or installed_sources()!=plan['sources']):
            raise ValueError('Worker precision or source evidence differs.')
        evidence=super().sampling_sft_worker_evidence()
        evidence['triton_precision']=dict(environment='ieee',effective_knob=triton.knobs.language.fp32_default,
            cache=os.environ['TRITON_CACHE_DIR'],vllm_cache=os.environ['VLLM_CACHE_ROOT'],
            plan_sha256=file_hash(path),sources=plan['sources'])
        return evidence

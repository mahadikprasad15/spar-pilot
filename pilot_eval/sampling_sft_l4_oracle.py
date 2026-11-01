"""Versioned HF-only L4 gate runtime; never changes the sampler runtime."""
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from pilot_eval.workflow import _hash

FLAGS={'matmul_tf32':False,'cudnn_tf32':False,'matmul_precision':'highest'}


def validate_runtime(value):
    fields={'protocol','gpu','driver','cuda','torch_version','transformers_version','worker_numeric_flags'}
    if (set(value)!=fields or value['protocol']!='sampling-sft-hf-oracle-l4-v1'
            or value['gpu']!='NVIDIA L4' or value['worker_numeric_flags']!=FLAGS
            or value['transformers_version']!='4.57.6' or value['torch_version'].split('+')[0]!='2.9.0'
            or not all(isinstance(value[k],str) and value[k] for k in ('driver','cuda'))):
        raise ValueError('Invalid L4 HF oracle runtime contract.')


def gate_runtime(sampling,oracle):
    validate_runtime(oracle)
    return dict(sampling_runtime_sha256=_hash(sampling),runtime_hash=_hash(oracle))


def capture():
    import torch
    from pilot_eval.sampling_sft_engine import numeric_flags
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Require exactly one CUDA device.')
    drivers=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip().splitlines()
    if len(drivers)!=1:raise ValueError('Require exactly one visible GPU driver.')
    value=dict(protocol='sampling-sft-hf-oracle-l4-v1',gpu=torch.cuda.get_device_name(),driver=drivers[0],
        cuda=torch.version.cuda,torch_version=importlib.metadata.version('torch'),
        transformers_version=importlib.metadata.version('transformers'),worker_numeric_flags=numeric_flags())
    validate_runtime(value)
    return value


def open_oracle(plan,root):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    actual=capture();expected=plan['oracle_runtime']
    if actual!=expected:raise ValueError('Actual L4 HF oracle runtime differs from frozen scoring runtime.')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from pilot_eval.sampling_sft import verify
    from pilot_eval.sampling_sft_engine import HFOracle
    audit=verify(root/plan['source_plan'],root)['audit'];identity=audit['training']['identity']
    settings=plan['settings']; model_path=root/settings['local_model'];tokenizer_path=root/settings['local_tokenizer']
    if model_path.name!=identity['model_revision'] or tokenizer_path.name!=identity['tokenizer_revision']:
        raise ValueError('Oracle snapshot revision differs from audited source.')
    if json.loads((model_path/'config.json').read_text()).get('quantization_config'):raise ValueError('Quantized model is forbidden.')
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
    if _hash(tokenizer.chat_template)!=audit['training']['prompt_mask_contract']['chat_template_sha256']:
        raise ValueError('Pinned tokenizer chat template changed.')
    stops=[tokenizer.convert_tokens_to_ids(s) for s in ['<|im_end|>','<|endoftext|>']]
    if stops!=settings['runtime']['actual_stop_ids'] or tokenizer.eos_token_id not in stops:raise ValueError('Stop IDs changed.')
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
    if capture()!=expected:raise ValueError('Actual L4 numeric runtime changed while loading the oracle.')
    dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    devices=sorted({str(p.device) for p in model.parameters()})
    if dtypes!=['torch.float32'] or any(not d.startswith('cuda') for d in devices):raise ValueError('Oracle weights must be FP32 on CUDA.')
    return HFOracle(model),tokenizer,dict(execution_kind='real-cuda',worker=actual,
        parameter_dtypes=dtypes,parameter_devices=devices,runtime_hash=plan['runtime_hash'])


if __name__=='__main__':
    try:print(json.dumps(capture(),indent=2))
    except (ValueError,RuntimeError) as exc:print(str(exc),file=sys.stderr);sys.exit(1)

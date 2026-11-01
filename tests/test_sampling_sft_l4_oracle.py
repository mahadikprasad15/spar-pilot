import pytest


def test_runtime_contract_is_separate_and_rejects_changed_gpu():
    from pilot_eval.sampling_sft_l4_oracle import validate_runtime
    runtime={'protocol':'sampling-sft-hf-oracle-l4-v1','gpu':'NVIDIA L4','driver':'550',
             'cuda':'12.8','torch_version':'2.9.0','transformers_version':'4.57.6',
             'worker_numeric_flags':{'matmul_tf32':False,'cudnn_tf32':False,'matmul_precision':'highest'}}
    validate_runtime(runtime)
    with pytest.raises(ValueError):validate_runtime({**runtime,'gpu':'NVIDIA A40'})
    with pytest.raises(ValueError):validate_runtime({**runtime,'worker_numeric_flags':{**runtime['worker_numeric_flags'],'matmul_tf32':True}})


def test_l4_gate_identity_preserves_original_sampling_runtime():
    from pilot_eval.sampling_sft_l4_oracle import gate_runtime
    from pilot_eval.workflow import _hash
    sampling={'gpu':'NVIDIA A40','max_model_len':4096,'actual_stop_ids':[3,4]}
    oracle={'protocol':'sampling-sft-hf-oracle-l4-v1','gpu':'NVIDIA L4','driver':'550',
             'cuda':'12.8','torch_version':'2.9.0','transformers_version':'4.57.6',
             'worker_numeric_flags':{'matmul_tf32':False,'cudnn_tf32':False,'matmul_precision':'highest'}}
    out=gate_runtime(sampling,oracle)
    assert out['sampling_runtime_sha256']==_hash(sampling) and out['runtime_hash']==_hash(oracle)
    assert sampling['gpu']=='NVIDIA A40'

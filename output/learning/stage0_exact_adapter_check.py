"""Exact-name adapter verification; extracted from the tested production verifier.

No model math or acceptance threshold is changed.
"""

def verify_loaded_adapter(model,variant,directory):
    """Check every actual PEFT factor against its saved FP32 tensor payload."""
    import numpy as np
    from safetensors.numpy import load_file
    expected=load_file(str(directory/'adapter_model.safetensors'))
    # PEFT0.18's export selects by substring: V4 also selects V4-input.
    # Select the exact final LoRA adapter segment, then remove only that segment.
    suffix=f'.{variant}.weight'
    actual={name[:-len(suffix)]+'.weight':value for name,value in model.state_dict().items()
            if any(name.endswith(f'.{factor}{suffix}') for factor in ['lora_A','lora_B'])}
    if set(actual)!=set(expected):
        raise ValueError(f'Actual adapter mapping differs from saved payload: {variant}')
    for name,value in actual.items():
        array=value.detach().cpu().numpy()
        if array.dtype!=np.float32 or not np.array_equal(array,expected[name]):
            raise ValueError(f'Actual loaded adapter factor differs from saved payload: {variant}; {name}')
    return dict(variant=variant,tensor_count=len(expected),exact_payload_match=True)

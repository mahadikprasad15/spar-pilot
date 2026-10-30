"""Adapter-name prefixes must not contaminate loaded-factor verification."""
import sys
import ast
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('uploaded_helper', [False, True])
def test_verify_v4_does_not_export_v4_input_or_ignore_real_corruption(tmp_path, monkeypatch, uploaded_helper):
    if uploaded_helper:
        verify_loaded_adapter = runpy.run_path(str(Path(__file__).parents[1] /
            'output/learning/stage0_exact_adapter_check.py'))['verify_loaded_adapter']
    else:
        from pilot_eval.stage0_evaluation_backend import verify_loaded_adapter

    class Tensor:
        def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    expected = {'linear.lora_A.weight': np.array([[1., 2.]], dtype=np.float32),
                'linear.lora_B.weight': np.array([[3.], [4.]], dtype=np.float32)}
    raw = {'linear.lora_A.V4.weight': Tensor([[1., 2.]]),
           'linear.lora_B.V4.weight': Tensor([[3.], [4.]]),
           'linear.lora_A.V4-input.weight': Tensor([[9., 9.]]),
           'linear.lora_B.V4-input.weight': Tensor([[8.], [8.]])}
    model = SimpleNamespace(state_dict=lambda: raw)

    def legacy_export(model, adapter_name):
        # PEFT0.18 selects by substring, then strips only exact adapter segments.
        return {key.replace(f'.{adapter_name}.weight', '.weight'): value
                for key, value in model.state_dict().items() if adapter_name in key}

    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(get_peft_model_state_dict=legacy_export))
    monkeypatch.setitem(sys.modules, 'safetensors.numpy', SimpleNamespace(load_file=lambda _: expected))
    assert verify_loaded_adapter(model, 'V4', tmp_path)['tensor_count'] == 2
    # The sibling must itself remain correctly selectable, not simply dropped.
    sibling = {'linear.lora_A.weight': np.array([[9., 9.]], dtype=np.float32),
               'linear.lora_B.weight': np.array([[8.], [8.]], dtype=np.float32)}
    monkeypatch.setitem(sys.modules, 'safetensors.numpy', SimpleNamespace(load_file=lambda _: sibling))
    assert verify_loaded_adapter(model, 'V4-input', tmp_path)['tensor_count'] == 2
    monkeypatch.setitem(sys.modules, 'safetensors.numpy', SimpleNamespace(load_file=lambda _: expected))
    raw['linear.lora_B.V4.weight'].value[0, 0] = 999
    with pytest.raises(ValueError, match='loaded adapter factor'):
        verify_loaded_adapter(model, 'V4', tmp_path)
    del raw['linear.lora_B.V4.weight']
    with pytest.raises(ValueError, match='mapping'):
        verify_loaded_adapter(model, 'V4', tmp_path)


def test_uploaded_verifier_matches_production_function():
    root = Path(__file__).parents[1]
    source = ast.parse((root / 'pilot_eval/stage0_evaluation_backend.py').read_text())
    helper = ast.parse((root / 'output/learning/stage0_exact_adapter_check.py').read_text())
    function = lambda tree: next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                 and node.name == 'verify_loaded_adapter')
    assert ast.dump(function(source)) == ast.dump(function(helper))

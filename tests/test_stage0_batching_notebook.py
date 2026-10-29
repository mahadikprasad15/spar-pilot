import ast
import json
from pathlib import Path


def test_saved_notebooks_expose_batch16_handoff_and_keep_internals_singleton():
    root=Path(__file__).parents[1]
    for kind in ['cpu','gpu']:
        notebook=json.loads((root/f'notebooks/stage-0-{kind}-colab.ipynb').read_text())
        text='\n'.join(''.join(c['source']) for c in notebook['cells'])
        assert 'USE_BATCH16 = not CONDITIONAL_EXTENSION' in text
        assert '-batch16-primary-findings-v1' in text
        for cell in notebook['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        if kind=='gpu':
            for operation in ['batch-prepare','batch-run','batch-verify','batch-approve','batch-freeze']:
                assert f"'{operation}'" in text
            assert 'INFERENCE_BATCH_SIZE = 16 if USE_BATCH16 else 1' in text
            assert "'--context-chunk', 32" in text
            assert 'BATCH_PRODUCTION_APPROVED = False' in text

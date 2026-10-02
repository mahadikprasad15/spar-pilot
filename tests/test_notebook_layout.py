"""Both pilots share one clearly named workspace and expose offline scoring."""

import ast
import json
from pathlib import Path


def test_shared_notebook_locations_and_cpu_score_audit():
    root = Path(__file__).parents[1]
    for pilot in (1, 2):
        notebook = json.loads((root / f'notebooks/pilot-{pilot}-colab.ipynb').read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells']]
        text = '\n'.join(sources)
        assert 'SPAR/spar-pilot' in text
        assert 'SPAR/pilot1' not in text
        assert '/content/spar-pilot2' not in text
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        if pilot == 2:
            audit = next(s for s in sources if "'audit-sft-scores'" in s)
            assert 'drive.mount(' in audit
            assert '0fc7f29' in audit
            assert 'subprocess.run(' in audit
            assert 'sft-train' not in audit and 'sft-evaluate' not in audit

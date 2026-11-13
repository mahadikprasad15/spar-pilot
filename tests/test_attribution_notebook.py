"""Ticket 06a notebook (written, not run): structure and scope checks only."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'adapter-attribution-colab.ipynb'


def cells():
    return json.loads(NOTEBOOK.read_text())['cells']


def test_notebook_code_cells_parse_and_never_touch_production_inputs():
    code = [''.join(c['source']) for c in cells() if c['cell_type'] == 'code']
    for source in code:
        ast.parse(source)
    text = '\n'.join(code)
    assert 'attribution_rehearsal' in text and 'rehearsal-colab' in text and 'mode="rehearsal"' in text
    for forbidden in ['attribution_chain run', 'attribution_overnight', "'heldout'", "'discovery'", 'cohort_items']:
        assert forbidden not in text


def test_notebook_refuses_unset_user_decisions():
    settings = ''.join(cells()[2]['source'])
    for name in ['COLAB_CAP_HOURS', 'REHEARSAL_GENERATION_IDS', 'RANDOM_ORDER_SEEDS', 'NULL_SEED', 'TIMEOUTS_SECONDS',
                 'FP32_PEAK_FLOPS', 'EARLY_STOP_QUANTILE']:
        assert f'{name} = None' in settings
    assert any('assert not missing' in ''.join(c['source']) for c in cells())

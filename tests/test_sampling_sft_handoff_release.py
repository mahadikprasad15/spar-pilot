"""The shared notebook pins code that actually contains its public workflow."""
import ast
import json
from pathlib import Path
import subprocess


def test_notebook_release_pin_matches_local_execution_sources():
    from pilot_eval.sampling_sft_handoff import check_release
    repo=Path(__file__).parents[1]
    book=json.loads((repo/'notebooks/sampling-sft.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    values=[node.value.value for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCE_RELEASE' for t in node.targets)
        and isinstance(node.value,ast.Constant)]
    assert len(values)==1
    result=check_release(values[0])
    assert 'pilot_eval/sampling_sft_handoff.py' in result['source_files']
    assert 'pilot_eval/sampling_sft_training.py' in result['source_files']
    cli=subprocess.check_output(['git','show',values[0]+':pilot_eval/cli.py'],cwd=repo,text=True)
    assert "sampling-sft-handoff-" in cli and "['prepare','build','verify','status','release']" in cli
    assert result['release_commit']==values[0]

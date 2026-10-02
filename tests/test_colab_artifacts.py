"""Git setup must accept the notebook-created Drive artifact symlink."""

import subprocess
import ast
import json
from pathlib import Path


def test_notebook_progress_survives_transient_empty_status(tmp_path):
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/pilot-1-colab.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and "def run_cli(" in "".join(c["source"]))
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    config = dict(run_id="run", experiment="pilot", model="qwen", dataset="mmlu", cohort="test", variant="v")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    run_dir = tmp_path / "runs/pilot/qwen/mmlu/test/v/run"
    (run_dir / "meta").mkdir(parents=True)
    (run_dir / "results").mkdir()
    (run_dir / "results/results.json").write_text('{"total": 2}')
    class Process:
        polls = 0
        def wait(self, timeout):
            self.polls += 1
            if self.polls <= 2:
                (run_dir / "meta/status.json").write_text("" if self.polls == 1 else '{"state":"running","completed":1,"total":2}')
                raise subprocess.TimeoutExpired("fake", timeout)
            return 0
    process = Process()
    class Subprocess:
        TimeoutExpired = subprocess.TimeoutExpired
        STDOUT = subprocess.STDOUT
        @staticmethod
        def Popen(*args, **kwargs):
            return process
    import sys
    scope = dict(json=json, subprocess=Subprocess, sys=sys, ARTIFACT_ROOT=tmp_path,
                 SESSION_META=tmp_path, REPO_DIR=tmp_path)
    exec(compile(tree, "notebook-helper", "exec"), scope)
    assert scope["run_cli"](config_path) == {"total": 2}
    assert process.polls == 3


def test_drive_artifact_symlink_is_ignored_by_git(tmp_path):
    repo = tmp_path / "repo"
    storage = tmp_path / "drive-artifacts"
    storage.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    ignore = Path(__file__).parents[1] / ".gitignore"
    (repo / ".gitignore").write_bytes(ignore.read_bytes())
    (repo / "artifacts").symlink_to(storage, target_is_directory=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True)
    assert "artifacts" not in untracked.splitlines()

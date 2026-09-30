"""Git setup must accept the notebook-created Drive artifact symlink."""

import subprocess
from pathlib import Path


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

"""Archived unfinished tests: private-repository work was cancelled when the repo became public."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def notebook_git_namespace(monkeypatch):
    notebook = json.loads((Path(__file__).parents[1] / "output/jupyter-notebook/pilot-1-colab.ipynb").read_text())
    source = next(("".join(cell["source"]) for cell in notebook["cells"]
                   if "def github_git(" in "".join(cell["source"])), "")
    assert source, "notebook must support authenticated Git operations"
    colab = ModuleType("google.colab")
    colab.userdata = type("Userdata", (), {"get": staticmethod(lambda name: "test-only-secret")})()
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {"os": __import__("os"), "sys": sys, "subprocess": subprocess, "Path": Path}
    exec(source, namespace)
    return namespace


def test_private_clone_uses_temporary_askpass_without_persisting_token(monkeypatch):
    namespace = notebook_git_namespace(monkeypatch)
    original_run = subprocess.run
    helpers = []

    def fake_git(command, **kwargs):
        assert "test-only-secret" not in " ".join(command)
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert "credential.helper=" in command
        helper = Path(kwargs["env"]["GIT_ASKPASS"])
        helpers.append(helper)
        assert "test-only-secret" not in helper.read_text()
        reply = original_run([str(helper), "Password for https://github.com:"], env=kwargs["env"],
                             capture_output=True, text=True, check=True)
        assert reply.stdout.strip() == "test-only-secret"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_git)
    namespace["github_git"](["clone", "https://github.com/owner/private.git", "/tmp/fake"])
    assert helpers and not helpers[0].exists()
    assert "PILOT1_GITHUB_TOKEN" not in namespace["os"].environ


def test_clone_error_is_actionable_and_redacts_credentials(monkeypatch):
    namespace = notebook_git_namespace(monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 128, "", "fatal: auth failed test-only-secret"))
    with pytest.raises(RuntimeError, match="fatal: auth failed") as exc:
        namespace["github_git"](["clone", "https://github.com/owner/private.git", "/tmp/fake"])
    assert "test-only-secret" not in str(exc.value)

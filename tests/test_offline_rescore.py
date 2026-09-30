import json

from pilot_eval.cli import main
from pilot_eval.workflow import prepare_plan, execute_config
from test_workflow import Dependencies


def test_offline_rescore_preserves_source_and_is_idempotent(tmp_path):
    class Prose(Dependencies):
        def load_backend(self, config):
            class Backend:
                def generate_batch(self, prompts, decoding):
                    return [dict(text="Final answer: It takes 2 bolts.", token_count=8, stop_reason="eos") for p in prompts]
            return Backend()
    path = prepare_plan(tmp_path, "source", dependencies=Prose())[0]
    execute_config(path, tmp_path, dependencies=Prose())
    run = next(tmp_path.glob("runs/**/source-gsm8k-0shot"))
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    args = ["rescore-gsm8k", "--responses", str(run / "results/responses.jsonl"),
            "--config", str(run / "config.json"), "--summary", str(run / "results/results.json"),
            "--name", "v2", "--output-root", str(tmp_path)]
    assert main(args) == 0
    report = tmp_path / "reports/v2"
    summary = json.loads((report / "results/results.json").read_text())
    assert summary["old_flexible_correct"] == 0
    assert summary["flexible_v2_correct"] == 150
    assert summary["flexible_v2_accuracy"] == 1.
    assert summary["changed_extraction_count"] == 150
    assert len((report / "results/changes.jsonl").read_text().splitlines()) == 150
    assert all(p.read_bytes() == content for p, content in before.items())
    assert main(args) == 0
    # Corruption must stop before publishing a new report.
    response = run / "results/responses.jsonl"
    response.write_text(response.read_text() + response.read_text().splitlines()[0] + "\n")
    args[args.index("v2")] = "corrupt"
    assert main(args) == 1
    assert not (tmp_path / "reports/corrupt").exists()

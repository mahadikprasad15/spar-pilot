import os
from pathlib import Path
import subprocess


def run_precheck(tmp_path, *, production='1 passed in 1.0s', gpu_exit=0, pytest_exit=0):
    root = tmp_path / 'artifacts'
    (root / 'job').mkdir(parents=True)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    python = bindir / 'python'
    python.write_text('''#!/bin/sh
if [ "$1" = "-m" ]; then
  case "$*" in *test_sampling_sft_training_production_path.py*) echo "$PRODUCTION"; exit "$PYTEST_EXIT";; esac
  echo "5 passed"; exit 0
fi
exit 0
''')
    (bindir / 'nvidia-smi').write_text('#!/bin/sh\nexit "$GPU_EXIT"\n')
    (bindir / 'df').write_text('#!/bin/sh\necho disk-ok\n')
    for path in bindir.iterdir():
        path.chmod(0o755)
    env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', PY=str(python), ROOT=str(root), JOB='job',
               PRODUCTION=production, GPU_EXIT=str(gpu_exit), PYTEST_EXIT=str(pytest_exit), RUNPOD_POD_ID='test')
    script = Path(__file__).resolve().parents[1] / 'scripts/sampling_sft_overnight_precheck.sh'
    return subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)


def test_precheck_propagates_failure_and_rejects_skipped_production(tmp_path):
    for i, kwargs in enumerate([{'production': '1 skipped in 1s'}, {'pytest_exit': 1}, {'gpu_exit': 1}]):
        case = tmp_path / str(i)
        case.mkdir()
        assert run_precheck(case, **kwargs).returncode != 0
    case = tmp_path / 'success'
    case.mkdir()
    assert run_precheck(case).returncode == 0

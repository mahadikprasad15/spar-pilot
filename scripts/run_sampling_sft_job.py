"""CLI-first bounded job/export/stop driver. No provisioning or deletion."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_eval.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['sampling-sft-job-execute', *sys.argv[1:]]))

"""Create explicit, source-fingerprinted IEEE settings from prior settings."""
import argparse
import json
import sys
from pathlib import Path
from pilot_eval import sampling_sft as source
from pilot_eval.sampling_sft_ieee_diagnostic import installed_sources
from pilot_eval.sampling_sft_instrument import validate_runtime
from pilot_eval.sampling_sft_sources import inside


def prepare(root, input_path, output_path):
    root=Path(root).resolve()
    old=inside(root,input_path); target=inside(root,output_path)
    if old==target or target.exists():
        raise ValueError('Write IEEE settings to a fresh path; preserve existing settings.')
    data=json.loads(old.read_text())
    runtime=data['runtime']
    if 'triton_precision' in runtime:
        raise ValueError('Input already has a precision identity; use its existing settings.')
    runtime['triton_precision']=dict(mode='ieee',cache_policy='isolated-per-engine',sources=installed_sources())
    validate_runtime(runtime)
    source.frozen(target,data)
    return target


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    try:print(prepare(a.output_root,a.input,a.output))
    except (ValueError,OSError,KeyError) as exc:
        print('IEEE settings stopped:',exc,file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())

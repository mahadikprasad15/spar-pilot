"""Small CLI wrapper: benchmark batch sizes before committing to a full run."""
import argparse
import json
from pathlib import Path
from pilot_eval.inference_profile import profile_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='saved untuned GSM8K plan config')
    parser.add_argument('--output-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--name', required=True, help='unique profile name; rerun unchanged to resume')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--sample-size', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    result = profile_inference(args.config, args.output_root, args.name, dtype=args.dtype,
        batch_sizes=args.batch_sizes, sample_size=args.sample_size, repeats=args.repeats)
    print('\nBatch | status | tokens/sec | peak allocated GiB | min/150 | min/six evaluations')
    for row in result['measurements']:
        if row['status'] == 'completed':
            print(f"{row['batch_size']:5} | completed | {row['tokens_per_second']:10.2f} | "
                  f"{row['peak_allocated_bytes']/2**30:18.2f} | {row['estimated_150_minutes']:7.1f} | "
                  f"{row['estimated_six_evaluations_minutes']:19.1f}")
        else:
            print(f"{row['batch_size']:5} | OOM — skipped")
    print('Provisional fastest batch:', result['recommended_batch_size'])
    print('Output differences versus smallest successful batch:', result['output_differences'])
    print('Saved profile:', args.output_root / 'runs/diagnostics/inference-profile' / args.name)
    print(result['limitations'])


if __name__ == '__main__':
    main()

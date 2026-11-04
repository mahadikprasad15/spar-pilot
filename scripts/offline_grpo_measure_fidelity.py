"""Measure generation-score versus teacher-forced differences on the real model.

Run this on the GPU session before choosing the fidelity tolerance. It saves the
observations under --output and changes nothing else.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def summary(result):
    values = sorted(abs(x) for row in result['observations'] for x in row['differences'])
    return dict(answers=len(result['observations']), tokens=len(values),
                lengths=[len(row['generated_token_ids']) for row in result['observations']],
                eos=sum(row['eos'] for row in result['observations']),
                max=values[-1], p99=values[int(.99*(len(values)-1))], median=statistics.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-min-tokens', type=int, default=100)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, enable_full_determinism
    from pilot_eval.grpo_preflight import load_frozen
    from pilot_eval.grpo_prepare import load_grpo_prepared
    from pilot_eval.offline_grpo_diagnostics import measure_generation_scores
    root = args.root.resolve()
    frozen = load_frozen(root/args.frozen, root)
    plan, training, _, _ = load_grpo_prepared(root/frozen['source_path'], root)
    rows = {row['id']: row for row in training}
    first = [rows[i] for i in frozen['optimizer_groups'][0]]
    weights = root/'inputs/models'/plan['model_revision']
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    enable_full_determinism(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True, dtype=torch.float32,
                                                attn_implementation='eager', device_map={'': 0})
    # Match the pipeline's loader: the saved generation config sets a repetition penalty.
    base.generation_config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.0, no_repeat_ngram_size=0)
    model = PeftModel.from_pretrained(base, root/plan['source_evidence']['checkpoints']['0']['path'])
    result = measure_generation_scores(model, [r['input_ids'][:r['prompt_tokens']] for r in first],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=frozen['settings']['completion_limit'], long_min_tokens=args.long_min_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(result=result, summary=summary(result))))
    print(json.dumps(summary(result)))


if __name__ == '__main__':
    main()

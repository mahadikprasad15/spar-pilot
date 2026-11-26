"""Reusable multi-lens down_proj decoder for any sparse-rewiring survivor set (Gold, k=2, GRPO).

Applies Logit Lens, J-Lens, J++, and R-Lens (neuronpedia/jacobian-lens :: qwen2.5-1.5b-it/)
to frozen Qwen2.5-1.5B-Instruct down_proj columns for any JSON survivor registry.

Two zero-bloat weight modes:
  1. --model-dir /path/to/qwen   (Pod mode: reads safetensors directly from local disk in ~5s)
  2. --stream-hf                 (Mac mode: HTTP byte-range streams only needed tensors -> ~25 MB cache)

Input --survivors-json format (list of dicts, or dict with key "survivors"):
  [{"layer": 17, "idx": 6230, "group": "shared", "in_arm_a": true, "in_arm_b": true, ...}, ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

LENS_REPO = 'neuronpedia/jacobian-lens'
LENS_FILES = {
    'jlens': 'qwen2.5-1.5b-it/jlens/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_jacobian_lens.pt',
    'jpp':   'qwen2.5-1.5b-it/jpp/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_jpp_lens.pt',
    'rlens': 'qwen2.5-1.5b-it/rlens/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_rlens.pt',
}
ST_URL = 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors'
DTYPE_MAP = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}

PROBE_GROUPS = [
    ('####',          ['####', ' ####'],           'gsm8k'),
    ('<<',            ['<<'],                      'gsm8k'),
    ('>>',            ['>>'],                      'gsm8k'),
    ('digits 0-9',    [str(d) for d in range(10)], 'gsm8k'),
    ('=',             ['=', ' ='],                 'gsm8k'),
    ('$',             ['$', ' $'],                 'gsm8k'),
    ('\\n',           ['\n'],                      'gsm8k'),
    ('\\n\\n',        ['\n\n'],                    'gsm8k'),
    ('** (bold)',     ['**', ' **'],               'style'),
    ('### (hdr)',     ['###', '##'],               'style'),
    ('\\[ (LaTeX)',   ['\\[', '\\('],              'style'),
    ('boxed',         ['boxed'],                   'style'),
    ('Step',          ['Step', ' Step'],           'style'),
    ('Therefore',     ['Therefore', ' Therefore'], 'style'),
    ('So',            ['So', ' So'],               'style'),
    ('Final/Ans',     ['Answer', ' Answer', 'Final', ' Final'], 'style'),
    ('<|im_end|>',    ['<|im_end|>'],              'eos'),
    ('<|endoftext|>', ['<|endoftext|>'],           'eos'),
]


def load_survivor_registry(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    items = raw['survivors'] if isinstance(raw, dict) and 'survivors' in raw else raw
    if not isinstance(items, list) or not items:
        raise ValueError('survivors JSON must contain a non-empty list of survivor dicts')
    out = []
    for rec in items:
        s = dict(rec)
        s['layer'] = int(s['layer'])
        s['idx'] = int(s['idx'])
        s.setdefault('label', f"L{s['layer']} #{s['idx']}")
        s.setdefault('group', 'survivor')
        out.append(s)
    return out


def compute_probe_zscores(logits_by_lens: dict[str, torch.Tensor], id_to_str: list[str]) -> dict[str, Any]:
    """Compute standardized vocab z-scores across the 18 GSM8K/style/EOS probe columns for all lenses."""
    str_to_id: dict[str, int] = {}
    for tid, tok_str in enumerate(id_to_str):
        if tok_str not in str_to_id:
            str_to_id[tok_str] = tid
    resolved = []
    for name, cands, cat in PROBE_GROUPS:
        tids = [str_to_id[c] for c in cands if c in str_to_id]
        if tids:
            resolved.append({'name': name, 'cat': cat, 'tids': tids})
    mats: dict[str, list[list[float]]] = {}
    for ln, L in logits_by_lens.items():
        Lf = L.float()
        Z = (Lf - Lf.mean(dim=-1, keepdim=True)) / (Lf.std(dim=-1, keepdim=True) + 1e-8)
        mat = np.zeros((Lf.shape[0], len(resolved)), dtype=np.float32)
        for ci, item in enumerate(resolved):
            mat[:, ci] = Z[:, item['tids']].mean(dim=-1).numpy()
        mats[ln] = np.round(mat, 3).tolist()
    return {'columns': [{'name': r['name'], 'cat': r['cat']} for r in resolved], 'zscores': mats}


def _download_hf(repo: str, rel_path: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{repo}/resolve/main/{rel_path}'
    req = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'python-urllib'})
    part = dst.with_suffix(dst.suffix + '.part')
    with urllib.request.urlopen(req, timeout=300) as resp, open(part, 'wb') as out_f:
        while True:
            chunk = resp.read(2 * 1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
    os.replace(part, dst)
    return dst


def _fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity', 'User-Agent': 'python-urllib'}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        chunks = []
        while True:
            c = resp.read(2 * 1024 * 1024)
            if not c:
                break
            chunks.append(c)
        return b''.join(chunks)


def _load_weights_local(model_dir: Path, needed_layers: list[int]):
    st_path = model_dir / 'model.safetensors'
    with safe_open(str(st_path), framework='pt', device='cpu') as f:
        norm_weight = f.get_tensor('model.norm.weight').float()
        W_U = f.get_tensor('model.embed_tokens.weight').float()
        downs = {L: f.get_tensor(f'model.layers.{L}.mlp.down_proj.weight').float() for L in needed_layers}
    tok = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    return norm_weight, W_U, downs, tok


def _load_weights_stream_hf(tmp_dir: Path, needed_layers: list[int]):
    tok_path = _download_hf('Qwen/Qwen2.5-1.5B-Instruct', 'tokenizer.json', tmp_dir / 'tokenizer.json')
    tok = Tokenizer.from_file(str(tok_path))
    raw = _fetch_range(ST_URL, 0, 65535)
    header_len = struct.unpack('<Q', raw[:8])[0]
    if 8 + header_len > len(raw):
        raw = _fetch_range(ST_URL, 0, 8 + header_len - 1)
    header = json.loads(raw[8:8 + header_len].decode('utf-8'))
    data_base = 8 + header_len

    def grab(name: str) -> torch.Tensor:
        meta = header[name]
        s, e = meta['data_offsets']
        buf = _fetch_range(ST_URL, data_base + s, data_base + e - 1)
        return torch.frombuffer(bytearray(buf), dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape']).float()

    norm_weight = grab('model.norm.weight')
    W_U = grab('model.embed_tokens.weight')
    downs = {L: grab(f'model.layers.{L}.mlp.down_proj.weight') for L in needed_layers}
    return norm_weight, W_U, downs, tok


def decode_survivors(
    survivors: list[dict[str, Any]],
    *,
    model_dir: str | Path | None = None,
    lens_dir: str | Path | None = None,
    out_path: str | Path,
    top_k: int = 30,
    keep_tmp: bool = False,
) -> Path:
    """Extract frozen down_proj columns for ``survivors``, apply all 4 lenses, and save a compact .pt cache."""
    t0 = time.time()
    tmp_dir = Path('/tmp/jlens_decode_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    needed_layers = sorted({s['layer'] for s in survivors})

    try:
        if model_dir is not None:
            norm_weight, W_U, downs, tok = _load_weights_local(Path(model_dir), needed_layers)
        else:
            norm_weight, W_U, downs, tok = _load_weights_stream_hf(tmp_dir, needed_layers)

        lenses: dict[str, dict[int, torch.Tensor]] = {}
        lens_meta: dict[str, Any] = {}
        for lname, rel in LENS_FILES.items():
            local_pt = (Path(lens_dir) / f'{lname}.pt') if lens_dir else _download_hf(LENS_REPO, rel, tmp_dir / f'{lname}.pt')
            obj = torch.load(local_pt, map_location='cpu', weights_only=False)
            lenses[lname] = obj['J']
            lens_meta[lname] = {'source_layers': obj['source_layers'], 'n_prompts': obj['n_prompts'],
                                'provenance': obj.get('provenance')}

        vocab_size, d_model = W_U.shape
        id_to_str = [tok.decode([i], skip_special_tokens=False) for i in range(vocab_size)]

        layer_norm_stats: dict[int, dict[str, float]] = {}
        down_vecs: list[torch.Tensor] = []
        for L, W_down in downs.items():
            cn = W_down.norm(dim=0)
            layer_norm_stats[L] = {
                'mean': float(cn.mean()), 'median': float(cn.median()),
                'p90': float(torch.quantile(cn, 0.90)), 'p99': float(torch.quantile(cn, 0.99)),
                'max': float(cn.max()),
            }
        for s in survivors:
            W_down = downs[s['layer']]
            cn = W_down.norm(dim=0)
            v = W_down[:, s['idx']].clone()
            vn = float(v.norm())
            s['v_norm'] = vn
            s['v_norm_pct'] = float((cn <= vn).float().mean() * 100.0)
            down_vecs.append(v)

        lens_names = ['logit_lens', 'jlens', 'jpp', 'rlens']
        n = len(survivors)
        raw_v = torch.stack(down_vecs, dim=0)
        transported_u = {ln: torch.zeros(n, d_model, dtype=torch.float32) for ln in lens_names}
        u_raw_norms = {ln: torch.zeros(n, dtype=torch.float32) for ln in lens_names}
        logits_fp16 = {ln: torch.zeros(n, vocab_size, dtype=torch.float16) for ln in lens_names}
        top_k_records: dict[str, list[dict[str, Any]]] = {ln: [] for ln in lens_names}

        for i, s in enumerate(survivors):
            L = s['layer']
            v = raw_v[i]
            for ln in lens_names:
                u = v if (ln == 'logit_lens' or L not in lenses[ln]) else (lenses[ln][L].float() @ v)
                u_raw_norms[ln][i] = float(u.norm())
                u_unit = u / (u.pow(2).mean().sqrt() + 1e-8)
                transported_u[ln][i] = u_unit
                logits = W_U @ (u_unit * norm_weight)
                logits = logits - logits.mean()
                logits_fp16[ln][i] = logits.to(torch.float16)
                pv, pi = torch.topk(logits, top_k)
                nv, ni = torch.topk(-logits, top_k)
                top_k_records[ln].append({
                    'pos': [(int(t), id_to_str[int(t)], float(val)) for t, val in zip(pi, pv)],
                    'neg': [(int(t), id_to_str[int(t)], float(-val)) for t, val in zip(ni, nv)],
                    'std': float(logits.std()),
                })

        probe = compute_probe_zscores(logits_fp16, id_to_str)
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
            'lens_repo': LENS_REPO,
            'survivors': survivors,
            'layer_norm_stats': layer_norm_stats,
            'lens_names': lens_names,
            'lens_meta': lens_meta,
            'raw_v': raw_v,
            'transported_u': transported_u,
            'u_raw_norms': u_raw_norms,
            'logits_fp16': logits_fp16,
            'top_k_records': top_k_records,
            'probe': probe,
            'id_to_str': id_to_str,
        }, out_p)
        print(f'Decoded {n} survivors x 4 lenses -> {out_p} ({out_p.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.1f}s')
        return out_p
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Decode frozen down_proj vectors with Logit/J/J++/R lenses.')
    parser.add_argument('--survivors-json', required=True, help='JSON with survivor (layer, idx) records')
    parser.add_argument('--out', required=True, help='Output .pt decoded cache path')
    parser.add_argument('--model-dir', default=None, help='Local Qwen2.5-1.5B-Instruct dir (omit to stream via HTTP Range)')
    parser.add_argument('--lens-dir', default=None, help='Optional local dir with jlens.pt, jpp.pt, rlens.pt')
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--keep-tmp', action='store_true')
    args = parser.parse_args(argv)

    survivors = load_survivor_registry(args.survivors_json)
    decode_survivors(survivors, model_dir=args.model_dir, lens_dir=args.lens_dir, out_path=args.out,
                     top_k=args.top_k, keep_tmp=args.keep_tmp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

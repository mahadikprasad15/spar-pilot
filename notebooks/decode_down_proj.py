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
import hashlib
import json
import os
import re
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

QWEN_MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
QWEN_MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
LENS_REPO = 'neuronpedia/jacobian-lens'
LENS_FILES = {
    'jlens': 'qwen2.5-1.5b-it/jlens/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_jacobian_lens.pt',
    'jpp':   'qwen2.5-1.5b-it/jpp/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_jpp_lens.pt',
    'rlens': 'qwen2.5-1.5b-it/rlens/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_rlens.pt',
}
ST_URL = f'https://huggingface.co/{QWEN_MODEL_ID}/resolve/{QWEN_MODEL_REVISION}/model.safetensors'
DTYPE_MAP = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
_HEX_SHA_RE = re.compile(r'^[0-9a-f]{40}$')


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _require_immutable_revision(rev: str, *, name: str) -> str:
    if not isinstance(rev, str) or not _HEX_SHA_RE.fullmatch(rev):
        raise ValueError(f'{name} must be an immutable 40-char hex commit SHA, got {rev!r}')
    return rev

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


def _download_hf(repo: str, rel_path: str, dst: Path, *, revision: str) -> Path:
    _require_immutable_revision(revision, name=f'{repo} revision')
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{rel_path}'
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


def _safetensors_header_sha256(st_path: Path) -> str:
    """Hash the safetensors JSON header so local and HTTP-streamed decodes share an identical model fingerprint."""
    with st_path.open('rb') as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f'{st_path}: truncated safetensors header')
        header_len = struct.unpack('<Q', raw_len)[0]
        header_bytes = fh.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f'{st_path}: truncated safetensors header bytes')
    return hashlib.sha256(header_bytes).hexdigest()


def _load_weights_local(model_dir: Path, needed_layers: list[int]):
    st_path = model_dir / 'model.safetensors'
    tok_path = model_dir / 'tokenizer.json'
    with safe_open(str(st_path), framework='pt', device='cpu') as f:
        norm_weight = f.get_tensor('model.norm.weight').float()
        W_U = f.get_tensor('model.embed_tokens.weight').float()
        downs = {L: f.get_tensor(f'model.layers.{L}.mlp.down_proj.weight').float() for L in needed_layers}
    tok = Tokenizer.from_file(str(tok_path))
    hashes = {
        'model.safetensors': _sha256_file(st_path),
        'model.safetensors:header': _safetensors_header_sha256(st_path),
        'tokenizer.json': _sha256_file(tok_path),
    }
    return norm_weight, W_U, downs, tok, hashes


def _load_weights_stream_hf(tmp_dir: Path, needed_layers: list[int], *, model_revision: str = QWEN_MODEL_REVISION):
    _require_immutable_revision(model_revision, name='model_revision')
    st_url = f'https://huggingface.co/{QWEN_MODEL_ID}/resolve/{model_revision}/model.safetensors'
    tok_path = _download_hf(QWEN_MODEL_ID, 'tokenizer.json', tmp_dir / 'tokenizer.json', revision=model_revision)
    tok = Tokenizer.from_file(str(tok_path))
    raw = _fetch_range(st_url, 0, 65535)
    header_len = struct.unpack('<Q', raw[:8])[0]
    if 8 + header_len > len(raw):
        raw = _fetch_range(st_url, 0, 8 + header_len - 1)
    header_bytes = raw[8:8 + header_len]
    header = json.loads(header_bytes.decode('utf-8'))
    data_base = 8 + header_len

    def grab(name: str) -> torch.Tensor:
        meta = header[name]
        s, e = meta['data_offsets']
        buf = _fetch_range(st_url, data_base + s, data_base + e - 1)
        return torch.frombuffer(bytearray(buf), dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape']).float()

    norm_weight = grab('model.norm.weight')
    W_U = grab('model.embed_tokens.weight')
    downs = {L: grab(f'model.layers.{L}.mlp.down_proj.weight') for L in needed_layers}
    hashes = {
        'model.safetensors:header': hashlib.sha256(header_bytes).hexdigest(),
        'tokenizer.json': _sha256_file(tok_path),
    }
    return norm_weight, W_U, downs, tok, hashes


DEFAULT_FINAL_LAYER = 27


def transport_vector(
    lenses: dict[str, dict[int, torch.Tensor]],
    lens_name: str,
    layer: int,
    vec: torch.Tensor,
    *,
    final_layer: int = DEFAULT_FINAL_LAYER,
) -> torch.Tensor:
    """Transport a frozen ``down_proj`` column ``vec`` through ``lens_name`` at ``layer``.

    Identity is permitted ONLY for ``logit_lens`` or the final transformer layer (``layer == final_layer``).
    Missing a fitted matrix for any non-final layer raises ``ValueError`` rather than silently falling back.
    """
    if lens_name == 'logit_lens' or int(layer) == int(final_layer):
        return vec
    layer_mats = lenses.get(lens_name) or {}
    if int(layer) not in layer_mats:
        raise ValueError(
            f'fitted lens {lens_name!r} is missing non-final layer {layer} '
            f'(only final layer {final_layer} uses identity)'
        )
    return layer_mats[int(layer)].float() @ vec


def verify_lens_orientation(
    lenses: dict[str, dict[int, torch.Tensor]],
    *,
    norm_weight: torch.Tensor | None = None,
    W_U: torch.Tensor | None = None,
    probe_pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    probe_layers: list[int] | None = None,
    final_layer: int = DEFAULT_FINAL_LAYER,
) -> dict[str, dict[str, Any]]:
    """Asymmetric orientation check comparing ``J @ x`` against ``J.T @ x`` toward a known target.

    Unlike the symmetric inner product ``(J @ v) · v == (J.T @ v) · v``, this tests either:
    - explicit asymmetric ``probe_pairs = (X [m, d], Y [m, d])`` measuring cosine similarity to target ``Y``, or
    - the exact RMSNorm+unembedding logit readout ``((u / RMS(u)) * norm_weight) @ W_U[tid]`` on token
      embeddings ``e_tid = W_U[tid]`` (where ``norm_weight`` and RMS scaling break symmetry).

    Raises ``ValueError`` if any lens scores lower under ``J @ x`` than under ``J.T @ x`` (transposed lens).
    """
    summary: dict[str, dict[str, Any]] = {}
    for lname, layer_mats in lenses.items():
        cand_layers = (
            [int(L) for L in probe_layers if int(L) != int(final_layer) and int(L) in layer_mats]
            if probe_layers is not None
            else [int(L) for L in sorted(layer_mats) if int(L) != int(final_layer)]
        )
        if not cand_layers:
            summary[lname] = {'n_checked': 0, 'orientation': 'J @ v', 'verified': True}
            continue
        fwd_scores: list[float] = []
        rev_scores: list[float] = []
        for L in cand_layers:
            J = layer_mats[L].float()
            if probe_pairs is not None:
                X, Y = probe_pairs[0].float(), probe_pairs[1].float()
                U_fwd = X @ J.T     # row i is J @ X[i]
                U_rev = X @ J       # row i is J.T @ X[i]
                Yn = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)
                cf = ((U_fwd / (U_fwd.norm(dim=-1, keepdim=True) + 1e-8)) * Yn).sum(dim=-1)
                cr = ((U_rev / (U_rev.norm(dim=-1, keepdim=True) + 1e-8)) * Yn).sum(dim=-1)
                fwd_scores.extend(cf.tolist())
                rev_scores.extend(cr.tolist())
            else:
                if norm_weight is None or W_U is None:
                    raise ValueError('verify_lens_orientation requires either probe_pairs or (norm_weight, W_U)')
                nw = norm_weight.float()
                step = max(W_U.shape[0] // 32, 1)
                tids = list(range(0, W_U.shape[0], step))[:32]
                E = W_U[tids].float()
                for row_i in range(E.shape[0]):
                    e = E[row_i]
                    u_f = J @ e
                    u_r = J.T @ e
                    uf_norm = (u_f / (u_f.pow(2).mean().sqrt() + 1e-8)) * nw
                    ur_norm = (u_r / (u_r.pow(2).mean().sqrt() + 1e-8)) * nw
                    fwd_scores.append(float((uf_norm @ e).item()))
                    rev_scores.append(float((ur_norm @ e).item()))
        mean_fwd = float(np.mean(fwd_scores))
        mean_rev = float(np.mean(rev_scores))
        if mean_fwd <= mean_rev:
            raise ValueError(
                f'lens {lname!r} failed asymmetric orientation check (mean J@x={mean_fwd:.4f} <= '
                f'J.T@x={mean_rev:.4f}); matrix appears transposed'
            )
        summary[lname] = {
            'n_checked': len(fwd_scores),
            'mean_score_J_x': round(mean_fwd, 4),
            'mean_score_JT_x': round(mean_rev, 4),
            'orientation': 'J @ v',
            'verified': True,
        }
    return summary


def decode_survivors(
    survivors: list[dict[str, Any]],
    *,
    model_dir: str | Path | None = None,
    lens_dir: str | Path | None = None,
    out_path: str | Path,
    top_k: int = 30,
    final_layer: int = DEFAULT_FINAL_LAYER,
    model_revision: str = QWEN_MODEL_REVISION,
    lens_revision: str | None = None,
    keep_tmp: bool = False,
) -> Path:
    """Extract frozen down_proj columns for ``survivors``, apply all 4 lenses, and save a compact .pt cache."""
    t0 = time.time()
    tmp_dir = Path('/tmp/jlens_decode_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    needed_layers = sorted({int(s['layer']) for s in survivors})

    try:
        if model_dir is not None:
            norm_weight, W_U, downs, tok, asset_hashes = _load_weights_local(Path(model_dir), needed_layers)
        else:
            norm_weight, W_U, downs, tok, asset_hashes = _load_weights_stream_hf(
                tmp_dir, needed_layers, model_revision=model_revision
            )

        lenses: dict[str, dict[int, torch.Tensor]] = {}
        lens_meta: dict[str, Any] = {}
        lens_hashes: dict[str, str] = {}
        for lname, rel in LENS_FILES.items():
            if lens_dir:
                local_pt = Path(lens_dir) / f'{lname}.pt'
            else:
                if not lens_revision:
                    raise ValueError('streaming lenses from HF requires an explicit immutable 40-char --lens-revision')
                local_pt = _download_hf(LENS_REPO, rel, tmp_dir / f'{lname}.pt', revision=lens_revision)
            lens_hashes[lname] = _sha256_file(local_pt)
            obj = torch.load(local_pt, map_location='cpu', weights_only=False)
            lenses[lname] = obj['J']
            lens_meta[lname] = {'source_layers': obj['source_layers'], 'n_prompts': obj['n_prompts'],
                                'provenance': obj.get('provenance')}
        asset_hashes['lenses'] = lens_hashes

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
            W_down = downs[int(s['layer'])]
            cn = W_down.norm(dim=0)
            v = W_down[:, int(s['idx'])].clone()
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

        orientation_check = verify_lens_orientation(
            lenses, norm_weight=norm_weight, W_U=W_U, probe_layers=needed_layers, final_layer=final_layer
        )
        for i, s in enumerate(survivors):
            L = int(s['layer'])
            v = raw_v[i]
            for ln in lens_names:
                u = transport_vector(lenses, ln, L, v, final_layer=final_layer)
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
            'model_id': QWEN_MODEL_ID,
            'model_revision': str(model_revision),
            'lens_repo': LENS_REPO,
            'lens_revision': str(lens_revision) if lens_revision else 'local-verified-assets',
            'asset_sha256': asset_hashes,
            'final_layer': int(final_layer),
            'orientation_check': orientation_check,
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
    parser.add_argument('--model-revision', default=QWEN_MODEL_REVISION, help='Immutable 40-char HF commit SHA for base model')
    parser.add_argument('--lens-revision', default=None, help='Immutable 40-char HF commit SHA for jacobian-lens (required when streaming)')
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--keep-tmp', action='store_true')
    args = parser.parse_args(argv)

    survivors = load_survivor_registry(args.survivors_json)
    decode_survivors(survivors, model_dir=args.model_dir, lens_dir=args.lens_dir, out_path=args.out,
                     model_revision=args.model_revision, lens_revision=args.lens_revision,
                     top_k=args.top_k, keep_tmp=args.keep_tmp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""verify_qwen4exp.py -- check a Qwen3.8-Flash-Next (qwen4_exp) checkpoint before the engine loads it, with no weight on
the GPU. It reads the config, the index and the safetensors headers, and only the small tensors: the n-gram hash
buffers, the experts' global and input scales, and a few rows of the n-gram table.

The checks:
- the config parses (engine.model.qwen4exp.Qwen4ExpConfig.from_checkpoint and its refusals)
- every tensor of the language model is in the index, with the dtype and shape the loader expects; no other tensor
  of the language model is there
- the hash buffers equal the values that engine/weights/ple.py computes
- the n-gram table: its shards, rows, dims and scale; a sample of rows dequantizes to finite values
- the experts' scales are finite and positive; the tool reports whether gate and up share their global scale and
  whether each layer has one input scale per projection (the MoE kernels assume both)

Usage: uv run python tools/verify_qwen4exp.py [<repo id or dir>]   (default: nvidia/Qwen3.8-Flash-Next-NVFP4)
The exit status is non-zero if a check fails.
"""
from __future__ import annotations

import os
import struct
import sys
import time

import numpy as np
import torch

from engine.model.qwen4exp import Qwen4ExpConfig, Qwen4ExpForCausalLM
from engine.weights.direct import header
from engine.weights.loader import PREFIX, resolve, weight_map
from engine.weights.ple import NgramTable, ngram_params
from engine.weights.qwen4exp_loader import PROJS, skipped

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"


def expected_tensors(cfg: Qwen4ExpConfig) -> dict[str, tuple[str, tuple[int, ...]]]:
    """{checkpoint name: (dtype, shape)} of the language model's tensors, as the nvidia checkpoint stores them."""
    with torch.device("meta"):
        m = Qwen4ExpForCausalLM(cfg)
    out = {}
    for k, v in m.state_dict().items():
        if ".mlp.experts." in k or k.endswith(".ngram_embedding.weight"):
            continue
        out["lm_head.weight" if k == "lm_head.weight" else PREFIX + k] = ("BF16" if v.is_floating_point() else "I64", tuple(v.shape))
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    for li in range(cfg.num_hidden_layers):
        for e in range(cfg.num_experts):
            for p in PROJS:
                n, k = shapes[p]
                b = f"{PREFIX}layers.{li}.mlp.experts.{e}.{p}."
                out |= {b + "weight": ("U8", (n, k // 2)), b + "weight_scale": ("F8_E4M3", (n, k // 16)),
                        b + "weight_scale_2": ("F32", ()), b + "input_scale": ("F32", ())}
    return out


def read_scalars(path: str, names: list[str], h: dict, base: int) -> dict[str, float]:
    """The f32 scalars `names` of one file, by pread at their offsets (no tensor library, no page cache kept)."""
    out = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        for n in names:
            b, e = h[n]["data_offsets"]
            out[n] = struct.unpack("<f", os.pread(fd, e - b, base + b))[0]
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return out


def read_small(path: str, name: str, h: dict, base: int) -> np.ndarray:
    v = h[name]
    b, e = v["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + b)
        return np.frombuffer(f.read(e - b), dtype={"I64": np.int64, "F32": np.float32}[v["dtype"]]).reshape(v["shape"])


def verify(path_or_repo: str, log=print, sample_rows: int = 64) -> list[str]:
    """The failed checks (empty: the checkpoint is as the engine expects)."""
    errors: list[str] = []
    t0 = time.time()
    path = resolve(path_or_repo)
    try:
        cfg = Qwen4ExpConfig.from_checkpoint(path)
    except (ValueError, NotImplementedError, KeyError) as e:
        return [f"config: {e}"]
    wm = weight_map(path)
    want = expected_tensors(cfg)
    headers = {f: header(os.path.join(path, f)) for f in sorted(set(wm.values()))}
    for n, (dt, shape) in want.items():
        f = wm.get(n)
        if f is None:
            errors.append(f"missing: {n}")
            continue
        v = headers[f][0].get(n)
        if v is None:
            errors.append(f"{n}: in the index, not in {f}")
        elif (v["dtype"], tuple(v["shape"])) != (dt, shape):
            errors.append(f"{n}: {v['dtype']} {v['shape']}, expected {dt} {list(shape)}")
    extra = sorted(n for n in wm if n not in want and not skipped(n))
    errors += [f"unexpected: {n}" for n in extra[:20]]
    log(f"[verify] {len(want)} tensors of the language model checked against the headers of {len(headers)} files")

    for li, j in cfg.ple_layers.items():
        p = ngram_params(cfg, j)
        b = f"{PREFIX}layers.{li}.ple.ple_embedding."
        for name, ref in (("layer_multipliers", p.multipliers), ("ngram_heads_vocab_sizes", p.sizes), ("ngram_heads_offsets", p.offsets)):
            f = wm.get(b + name)
            if f is None:
                continue  # reported as missing
            got = read_small(os.path.join(path, f), b + name, *headers[f])
            if got.tolist() != list(ref):
                errors.append(f"{b + name}: {got.tolist()} != the computed {list(ref)}")
        f = wm.get(b + "ngram_embedding.shard_0.weight")
        if f is None:
            errors.append(f"{b}ngram_embedding: no shards")
            continue
        t = NgramTable(os.path.join(path, f), b + "ngram_embedding")
        if (t.n_rows, t.dim, t.n_shards) != (p.padded_rows, cfg.ngram_dim, cfg.split_ngram_parts):
            errors.append(f"{b}ngram_embedding: {t.n_shards} shards, {t.n_rows} rows x {t.dim}; expected {cfg.split_ngram_parts} shards, "
                          f"{p.padded_rows} rows x {cfg.ngram_dim}")
        ids = np.random.default_rng(0).integers(0, t.n_rows, sample_rows)
        v = t.dequant(torch.from_numpy(t.rows(np.concatenate([ids, [0, t.n_rows - 1]]))), torch.float32)
        if not torch.isfinite(v).all() or not np.isfinite(t.scale) or t.scale <= 0:
            errors.append(f"{b}ngram_embedding: non-finite rows or scale {t.scale}")
        log(f"[verify] n-gram table: {t.n_shards} shards x {t.rows_per_shard} rows x {t.dim}, scale {t.scale:.6g}, "
            f"{sample_rows + 2} rows read, |row| max {v.abs().max().item():.4g}")
        t.close()

    L, E = cfg.num_hidden_layers, cfg.num_experts
    gs, ins = np.full((L, E, 3), np.nan), np.full((L, E, 3), np.nan)
    by_file: dict[str, list[str]] = {}
    for li in range(L):
        for e in range(E):
            for p in PROJS:
                for s in ("weight_scale_2", "input_scale"):
                    n = f"{PREFIX}layers.{li}.mlp.experts.{e}.{p}.{s}"
                    if n in wm and n in headers[wm[n]][0]:
                        by_file.setdefault(wm[n], []).append(n)
    for f, names in by_file.items():
        for n, x in read_scalars(os.path.join(path, f), names, *headers[f]).items():
            parts = n.split(".")
            li, e, j = int(parts[3]), int(parts[6]), PROJS.index(parts[7])
            (gs if parts[8] == "weight_scale_2" else ins)[li, e, j] = x
    if not (np.isfinite(gs).all() and np.isfinite(ins).all() and (gs > 0).all() and (ins > 0).all()):
        errors.append("expert scales: non-finite or non-positive values")
    shared = bool(np.array_equal(gs[..., 0], gs[..., 1]))
    per_layer = bool((ins == ins[:, :1]).all())
    log(f"[verify] expert scales: gate/up global scales equal: {shared}; one input scale per layer and projection: {per_layer}; "
        f"gate == up input scale: {bool(np.array_equal(ins[..., 0], ins[..., 1]))}")
    log(f"[verify] {'OK' if not errors else f'{len(errors)} errors'} in {time.time() - t0:.1f}s")
    return errors


def main(argv: list[str]) -> int:
    errors = verify(argv[0] if argv else MODEL)
    for e in errors[:50]:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

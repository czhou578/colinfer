#!/usr/bin/env python3
"""Skinny GEMM (csrc/skinny.cu) weight-streaming rate per decode shape and format, at M = 1 / 8 / 16 rows. The bench
cycles over several copies of each weight, so nothing runs from L2 (LPDDR5x peak ~238 GB/s).

   uv run python bench/skinny_bench.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kernels import ops  # noqa: E402
from engine.weights.quantize import nvfp4_global_scale, pack5, pack6, quantize, quantize_int  # noqa: E402

# (name, format, N, K) of the decode linears of Qwen3.8-27B (lm_head and the MLP in NVFP4, attention INT6, GDN INT5)
SHAPES = [("mlp gate|up", "nvfp4", 34816, 5120), ("mlp down", "nvfp4", 5120, 17408), ("lm_head", "nvfp4", 248320, 5120),
          ("gdn qkv|z", "int5", 16384, 5120), ("gdn out", "int5", 5120, 6144), ("attn q|k|v", "int6", 14336, 5120),
          ("attn o", "int6", 5120, 6144)]


def make(fmt, N, K):
    w = torch.randn(N, K, device="cuda") * 0.02
    if fmt == "nvfp4":
        gs = nvfp4_global_scale(w)
        p, sf = quantize(w, gs)
        return (lambda x, out: ops().skinny_nvfp4(x, p, sf, gs, None, out)), N * K * 0.5625
    bits = 6 if fmt == "int6" else 5
    gs = float(w.abs().max()) / (448 * (2 ** (bits - 1) - 1))
    codes, sf = quantize_int(w, gs, bits)
    lo, hi = (pack6 if bits == 6 else pack5)(codes)
    return (lambda x, out: ops().skinny_int(x, lo, hi, sf, gs, None, out)), N * K * (0.8125 if bits == 6 else 0.6875)


def main():
    for name, fmt, N, K in SHAPES:
        fn, nbytes = make(fmt, N, K)
        fns = [fn] + [make(fmt, N, K)[0] for _ in range(min(8, max(1, int(400e6 // nbytes))) - 1)]
        line = f"{name:12s} {fmt:5s} {nbytes / 1e6:6.1f} MB x{len(fns)}"
        for M in (1, 8, 16):
            x = torch.randn(M, K, device="cuda").bfloat16()
            out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            for f in fns:
                f(x, out)
            torch.cuda.synchronize()
            n = 40
            t = time.perf_counter()
            for i in range(n):
                fns[i % len(fns)](x, out)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / n
            line += f" | M={M:2d} {dt * 1e3:6.3f} ms {nbytes / dt / 1e9:4.0f} GB/s"
        print(line, flush=True)
        del fns
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

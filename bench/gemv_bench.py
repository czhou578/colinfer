#!/usr/bin/env python3
"""gemv_bench.py -- bandwidth of the decode GEMV kernels on every linear shape of Qwen3.8-27B.

GB/s = weight bytes streamed (incl. block scales) / median kernel time. Each measurement cycles
through enough distinct weight copies (>= 160 MB) that the 24 MB L2 holds none of them, as in a
real decode step. Targets (docs/baseline.md): 233 GB/s measured DRAM read, >= 90% (210 GB/s);
stock CUTLASS NVFP4 GEMM at M=1 reached 217 GB/s.

  uv run python bench/gemv_bench.py [--m 1 2 3 4] [--iters 50]
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kernels import ops  # noqa: E402

BW = 233.0
# name, kind, N, K, count per decode step
SHAPES = [
    ("mlp gate+up (swiglu)", "swiglu", 17408, 5120, 64),
    ("mlp down", "nvfp4", 5120, 17408, 64),
    ("lm_head", "nvfp4_f32", 248320, 5120, 1),
    ("gdn in_proj_qkv", "fp8", 10240, 5120, 48),
    ("gdn in_proj_z", "fp8", 6144, 5120, 48),
    ("gdn out_proj", "fp8", 5120, 6144, 48),
    ("attn q_proj", "fp8", 12288, 5120, 16),
    ("attn k/v_proj", "fp8", 1024, 5120, 32),
    ("attn o_proj", "fp8", 5120, 6144, 16),
]


def make(kind, N, K):
    if kind.startswith("nvfp4") or kind == "swiglu":
        def one():
            return (torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda"),
                    (torch.rand(N, K // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn))
        mats = 2 if kind == "swiglu" else 1
        nbytes = mats * N * (K // 2 + K // 16)
        return one, nbytes, mats
    def one():
        return (torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn),)
    return one, N * K, 1


def bench(kind, N, K, M, iters):
    one, nbytes, mats = make(kind, N, K)
    copies = max(2, int(160e6 // nbytes) + 1)
    ws = [[one() for _ in range(mats)] for _ in range(copies)]
    x = torch.randn(M, K, device="cuda").bfloat16()
    out = torch.empty(M, N, device="cuda", dtype=torch.float32 if kind == "nvfp4_f32" else torch.bfloat16)
    o = ops()

    def run(i):
        w = ws[i % copies]
        if kind == "swiglu":
            o.nvfp4_swiglu(x, w[0][0], w[0][1], 0.5, w[1][0], w[1][1], 0.5, out)
        elif kind.startswith("nvfp4"):
            o.nvfp4_gemv(x, w[0][0], w[0][1], 0.5, None, out)
        else:
            o.fp8_gemv(x, w[0][0], 0.01, None, out)

    for i in range(copies * 2):
        run(i)
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); run(i); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    med = statistics.median(ts)
    del ws
    torch.cuda.empty_cache()
    return med, nbytes / med / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 3, 4])  # up to 8
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    print(f"{'shape':24s} {'N x K':>14s} {'MB':>7s} " + " ".join(f"{'M=' + str(m) + ' us':>9s} {'GB/s':>6s} {'%BW':>4s}" for m in a.m))
    step_us = {m: 0.0 for m in a.m}
    step_bytes = 0
    for name, kind, N, K, count in SHAPES:
        row = f"{name:24s} {f'{N}x{K}':>14s} {make(kind, N, K)[1] / 1e6:7.1f} "
        for m in a.m:
            us, gbs = bench(kind, N, K, m, a.iters)
            us *= 1000
            step_us[m] += us * count
            row += f"{us:9.1f} {gbs:6.1f} {100 * gbs / BW:4.0f}"
        step_bytes += make(kind, N, K)[1] * count
        print(row, flush=True)
    print(f"\nall linears of one decode step: {step_bytes / 1e9:.2f} GB")
    for m in a.m:
        t = step_us[m] / 1e3
        print(f"  M={m}: {t:6.1f} ms  ->  {step_bytes / t / 1e6:6.1f} GB/s effective ({100 * step_bytes / t / 1e6 / BW:.0f}% of {BW:.0f}),"
              f"  linears-only ceiling {1000 / t:5.2f} steps/s")


if __name__ == "__main__":
    main()

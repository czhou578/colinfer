#!/usr/bin/env python3
"""Decode attention: the split-KV CUDA-core kernel (one block per query row) vs the tensor-core multi-row kernel
(csrc/attn_decode.cu, namespace tc) at T = 1 / 4 / 8 query rows per slot (plain decode, k=3 and k=7 verify).
Time per layer and KV read rate, cycling over several layers' caches so short contexts do not run from L2.

   uv run python bench/attn_bench.py [--ctx 8192 32768 131072] [--kv fp8 fp4] [--B 1 3]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kernels import ops  # noqa: E402

Hq, Hkv, D = 24, 4, 256


def timeit(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--kv", nargs="+", default=["fp8", "fp4"])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--T", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--splits", type=int, default=32)
    a = ap.parse_args()
    print(f"{'kv':4} {'B':>2} {'ctx':>7} {'T':>2} | {'split-KV ms':>11} {'GB/s':>6} | {'tensor-core ms':>14} {'GB/s':>6} | speedup")
    for kind in a.kv:
        row = D if kind == "fp8" else D // 2 + D // 16
        for B in a.B:
            for ctx in a.ctx:
                Lmax = ctx + 16
                layers = max(1, min(16, (2 << 30) // (B * Hkv * Lmax * row * 2)))
                caches = []
                for _ in range(layers):
                    if kind == "fp8":
                        k = torch.randn(B, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
                        v = torch.randn(B, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
                    else:
                        k = torch.randint(0, 120, (B, Hkv, Lmax, row), dtype=torch.uint8, device="cuda")
                        v = torch.randint(0, 120, (B, Hkv, Lmax, row), dtype=torch.uint8, device="cuda")
                    caches.append((k, v))
                for T in a.T:
                    q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
                    gate = torch.randn(B, T, Hq, 2 * D, device="cuda").bfloat16()
                    out = torch.empty(B, T, Hq * D, device="cuda", dtype=torch.bfloat16)
                    sl = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
                    it = [0]

                    def old():
                        k, v = caches[it[0] % layers]
                        it[0] += 1
                        ops().attn_decode(q, k, v, sl, out, a.splits, D ** -0.5, gate)

                    def tc():
                        k, v = caches[it[0] % layers]
                        it[0] += 1
                        ops().attn_decode_tc(q, k, v, sl, out, D ** -0.5, gate)
                    iters = max(20, 4 * layers)
                    t_old, t_tc = timeit(old, iters), timeit(tc, iters)
                    gb = B * Hkv * ctx * row * 2 / 1e9
                    print(f"{kind:4} {B:>2} {ctx:>7} {T:>2} | {t_old:>11.3f} {gb / t_old * 1e3:>6.0f} | {t_tc:>14.3f} {gb / t_tc * 1e3:>6.0f} | "
                          f"{t_old / t_tc:.2f}x", flush=True)
                del caches
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

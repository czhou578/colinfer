#!/usr/bin/env python3
"""Decode attention (csrc/attn_decode.cu, fp8 KV) at T = 1 / 4 / 8 query rows per slot (plain decode, k=3 and k=7
verify): time per layer and KV read rate, cycling over several layers' caches so short contexts do not run from L2.

   uv run python bench/attn_bench.py [--ctx 8192 32768 131072] [--B 1 3] [--T 1 4 8]
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
    ap.add_argument("--B", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--T", type=int, nargs="+", default=[1, 4, 8])
    a = ap.parse_args()
    print(f"{'B':>2} {'ctx':>7} {'T':>2} | {'ms / layer':>10} {'GB/s':>6}")
    for B in a.B:
        for ctx in a.ctx:
            Lmax = ctx + 16
            layers = max(1, min(16, (2 << 30) // (B * Hkv * Lmax * D * 2)))
            caches = [tuple(torch.randn(B, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)) for _ in range(layers)]
            for T in a.T:
                q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
                gate = torch.randn(B, T, Hq, 2 * D, device="cuda").bfloat16()
                out = torch.empty(B, T, Hq * D, device="cuda", dtype=torch.bfloat16)
                sl = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
                it = [0]

                def run():
                    k, v = caches[it[0] % layers]
                    it[0] += 1
                    ops().attn_decode(q, k, v, sl, out, D ** -0.5, gate)
                t = timeit(run, max(20, 4 * layers))
                gb = B * Hkv * ctx * D * 2 / 1e9
                print(f"{B:>2} {ctx:>7} {T:>2} | {t:>10.3f} {gb / t * 1e3:>6.0f}", flush=True)
            del caches
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

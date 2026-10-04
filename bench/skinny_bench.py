#!/usr/bin/env python3
"""Tensor-core skinny GEMM (csrc/skinny.cu) vs the CUDA-core GEMV (csrc/gemv.cu) on the model's decode shapes:
weight-streaming GB/s at M = 1..16, accuracy against an fp32 reference, and row bit-identity across M.

   uv run python bench/skinny_bench.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.kernels import ops  # noqa: E402
from engine.weights.loader import dequant_nvfp4  # noqa: E402

SHAPES = [  # (name, kind, N, K)
    ("mlp gate+up (swiglu)", "swiglu", 17408, 5120),
    ("mlp down", "fp4", 5120, 17408),
    ("lm_head", "fp4", 248320, 5120),
    ("gdn qkvz", "fp8", 16384, 5120),
    ("gdn out", "fp8", 5120, 6144),
    ("attn qkv", "fp8", 14336, 5120),
]


def rand_fp4(N, K):
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    sf = (torch.rand(N, K // 16, device="cuda") * 2 + 0.25).to(torch.float8_e4m3fn)
    return w, sf


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e-3


def main():
    torch.manual_seed(0)
    print("| shape | bytes | " + " | ".join(f"M={m} gemv / skinny GB/s" for m in (1, 4, 8, 12, 16)) + " |")
    print("|---|---|" + "---|" * 5)
    for name, kind, N, K in SHAPES:
        if kind in ("fp4", "swiglu"):
            w, sf = rand_fp4(N, K)
            w2, sf2 = rand_fp4(N, K) if kind == "swiglu" else (None, None)
            nbytes = (w.numel() + sf.numel()) * (2 if kind == "swiglu" else 1)
        else:
            w = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
            nbytes = w.numel()
        row = []
        ref_rows = None
        for M in (1, 4, 8, 12, 16):
            x = (torch.randn(M, K, device="cuda")).bfloat16()
            out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            out_g = torch.empty_like(out)
            if kind == "fp4":
                sk = lambda: ops().skinny_nvfp4(x, w, sf, 0.01, None, out)  # noqa: E731
                gv = (lambda: [ops().nvfp4_gemv(x[i:i + 8], w, sf, 0.01, None, out_g[i:i + 8]) for i in range(0, M, 8)])  # noqa: E731
            elif kind == "swiglu":
                sk = lambda: ops().skinny_swiglu(x, w, sf, 0.01, w2, sf2, 0.01, out)  # noqa: E731
                gv = (lambda: [ops().nvfp4_swiglu(x[i:i + 8], w, sf, 0.01, w2, sf2, 0.01, out_g[i:i + 8]) for i in range(0, M, 8)])  # noqa: E731
            else:
                sk = lambda: ops().skinny_fp8(x, w, 0.02, None, out)  # noqa: E731
                gv = (lambda: [ops().fp8_gemv(x[i:i + 8], w, 0.02, None, out_g[i:i + 8]) for i in range(0, M, 8)])  # noqa: E731
            sk()
            gv()
            torch.cuda.synchronize()
            # accuracy vs fp32 reference on a column slice
            cols = slice(0, 4096)
            if kind == "fp8":
                ref = (x.float() @ (w[cols].float() * 0.02).t())
            elif kind == "fp4":
                ref = x.float() @ dequant_nvfp4(w[cols], sf[cols], torch.tensor(0.01, device="cuda"), torch.float32).t()
            else:
                g = x.float() @ dequant_nvfp4(w[cols], sf[cols], torch.tensor(0.01, device="cuda"), torch.float32).t()
                u = x.float() @ dequant_nvfp4(w2[cols], sf2[cols], torch.tensor(0.01, device="cuda"), torch.float32).t()
                ref = torch.nn.functional.silu(g) * u
            err = ((out[:, cols].float() - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-2, (name, M, err)
            # bit identity of rows across M (x rows are prefixes of the same random matrix? no: compare row 0 with M=1)
            if ref_rows is None:
                x0, ref_rows = x[:1].clone(), out[:1].clone()
            else:
                x[:1] = x0
                sk()
                torch.cuda.synchronize()
                assert torch.equal(out[:1], ref_rows), (name, M, "row 0 differs from the M=1 result")
            tg, ts = timeit(gv), timeit(sk)
            row.append(f"{nbytes / tg / 1e9:.0f} / **{nbytes / ts / 1e9:.0f}**")
        print(f"| {name} {N}x{K} | {nbytes / 1e6:.0f} MB | " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gemm_peak.py -- GEMM ceiling on DGX Spark (GB10) at the Qwen3.8-27B linear-layer shapes.

Backends
  bf16  cuBLAS            torch.matmul
  fp8   cuBLASLt          torch._scaled_mm, e4m3 x e4m3, tensor-wise scales, bf16 out
  nvfp4 flashinfer <be>   flashinfer.mm_fp4 with backend b12x / cutlass (cross-check of the CUTLASS numbers)
  fp8 / nvfp4 cutlass     bench/gemm_sm120.cu, one binary per tile config (build: make -C bench gemm -k)

Shapes: N x K in {17408 x 5120 (gate/up), 5120 x 17408 (down)}, M in {1, 16, 256, 2048, 4096}.
Every measurement is verified (bf16/fp8 against an fp32 matmul, CUTLASS inside the binary, FlashInfer
by relative error against fp32 since NVFP4 quantization of random data is lossy).

Usage: uv run python bench/gemm_peak.py [--iters 20] [--warmup 3] [--m 1 16 256] [--only REGEX] [--json FILE]
                                        [--swizzle 8] [--raster h|m|n]   (CUTLASS tile-scheduler raster; see docs/baseline.md)
"""
import argparse
import datetime
import glob
import json
import os
import re
import statistics
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
SHAPES = [(17408, 5120), (5120, 17408)]
DEFAULT_M = [1, 16, 256, 2048, 4096]
BYTES_PER_WEIGHT = {"bf16": 2.0, "fp8": 1.0, "nvfp4": 0.5 + 1.0 / 16}


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts)


def result(kind, m, n, k, med, best, verify):
    return dict(ms=med, best_ms=best, tflops=2.0 * m * n * k / med / 1e9,
                gbps=n * k * BYTES_PER_WEIGHT[kind] / med / 1e6, verify=verify)


def bench_bf16(m, n, k, warmup, iters):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    wt = w.t()
    fn = lambda: a @ wt
    out = fn()
    ref = a.float() @ w.float().t()
    ok = torch.allclose(out.float(), ref, rtol=2e-2, atol=1.0)
    med, best = time_fn(fn, warmup, iters)
    return result("bf16", m, n, k, med, best, "ok" if ok else "FAIL")


def bench_fp8(m, n, k, warmup, iters):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, device="cuda")
    sb = torch.tensor(1.0, device="cuda")
    wt = w.t()
    fn = lambda: torch._scaled_mm(a, wt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    out = fn()
    torch.cuda.synchronize()
    ref = a.float() @ w.float().t()
    ok = torch.allclose(out.float(), ref, rtol=2e-2, atol=1.0)
    med, best = time_fn(fn, warmup, iters)
    return result("fp8", m, n, k, med, best, "ok" if ok else "FAIL")


def bench_fi_nvfp4(backend):
    def run(m, n, k, warmup, iters):
        import flashinfer
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        a_gs = (448.0 * 6.0) / a.abs().max().float()
        w_gs = (448.0 * 6.0) / w.abs().max().float()
        a_q, a_sf = flashinfer.fp4_quantize(a, a_gs, sf_vec_size=16)
        w_q, w_sf = flashinfer.fp4_quantize(w, w_gs, sf_vec_size=16)
        alpha = (1.0 / (a_gs * w_gs)).float()
        fn = lambda: flashinfer.mm_fp4(a_q, w_q.t(), a_sf, w_sf.t(), alpha, torch.bfloat16,
                                       block_size=16, backend=backend)
        out = fn()
        torch.cuda.synchronize()
        ref = a.float() @ w.float().t()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        med, best = time_fn(fn, warmup, iters)
        # NVFP4 of gaussian data is lossy; a correct kernel lands well under 0.3, a broken one near 1+.
        return result("nvfp4", m, n, k, med, best, f"ok(rel={rel:.2f})" if rel < 0.3 else f"FAIL(rel={rel:.2f})")
    return run


def bench_cutlass(binary, swizzle, raster):
    def run(m, n, k, warmup, iters):
        p = subprocess.run([binary, "--m", str(m), "--n", str(n), "--k", str(k), "--iters", str(iters),
                            "--warmup", str(warmup), "--swizzle", str(swizzle), "--raster", raster],
                           capture_output=True, text=True, timeout=900)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not lines:
            raise RuntimeError((p.stderr or p.stdout).strip().splitlines()[-1][:120] if (p.stderr or p.stdout).strip() else f"exit {p.returncode}")
        kv = dict(tok.split("=", 1) for tok in lines[-1][len("RESULT "):].split())
        if "status" in kv:
            raise RuntimeError(kv["status"])
        kind = kv["kind"]
        r = result(kind, m, n, k, float(kv["ms"]), float(kv["best_ms"]), kv["verify"])
        r["maxerr"] = float(kv["maxerr"])
        if p.returncode != 0 and r["verify"] == "ok":
            r["verify"] = f"exit{p.returncode}"
        return r
    return run


def backends(only, swizzle, raster):
    bk = [("bf16 cuBLAS (torch.matmul)", "bf16", bench_bf16),
          ("fp8 cuBLASLt (torch._scaled_mm)", "fp8", bench_fp8),
          ("nvfp4 flashinfer b12x", "nvfp4", bench_fi_nvfp4("b12x")),
          ("nvfp4 flashinfer cutlass", "nvfp4", bench_fi_nvfp4("cutlass"))]
    for b in sorted(glob.glob(os.path.join(BUILD, "gemm_sm120_*"))):
        name = os.path.basename(b)[len("gemm_sm120_"):]  # e.g. nvfp4_128x128x128_pp
        kind, tile, sched = name.split("_")
        bk.append((f"{kind} cutlass {tile} {sched}", kind, bench_cutlass(b, swizzle, raster)))
    if only:
        bk = [b for b in bk if re.search(only, b[0])]
    return bk


def cell(r, field, fmt):
    if r is None or "error" in r:
        return "-".rjust(8)
    s = fmt % r[field]
    v = r.get("verify", "ok")
    return (s + ("" if v.startswith("ok") else "!")).rjust(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    ap.add_argument("--only", help="regex on backend label")
    ap.add_argument("--json")
    ap.add_argument("--swizzle", type=int, default=8, help="CUTLASS scheduler max_swizzle_size (0 = CUTLASS default)")
    ap.add_argument("--raster", default="h", choices=["h", "m", "n"])
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"gemm_peak  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {p.name} cc {p.major}.{p.minor}  "
          f"torch {torch.__version__}  iters {args.iters} (+{args.warmup} warmup)  medians  "
          f"cutlass swizzle={args.swizzle} raster={args.raster}")
    bk = backends(args.only, args.swizzle, args.raster)
    print(f"backends: {len(bk)}  ({sum(1 for b in bk if 'cutlass' in b[0] and 'flashinfer' not in b[0])} CUTLASS configs)")

    res = {}   # (label, n, k, m) -> result
    errors = {}
    for (n, k) in SHAPES:
        for m in args.m:
            for label, kind, fn in bk:
                key = (label, n, k, m)
                if label in errors and errors[label][0] == "build":
                    continue
                try:
                    res[key] = fn(m, n, k, args.warmup, args.iters)
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:110]}" if str(e).strip() else type(e).__name__
                    res[key] = dict(error=msg)
                    errors.setdefault(label, ("run", msg))
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    for (n, k) in SHAPES:
        print(f"\n=== N x K = {n} x {k}   TFLOPS (median)   '!' = verification failed, '-' = unsupported/error")
        print("  " + "backend".ljust(36) + "".join(f"M={m}".rjust(8) for m in args.m))
        for label, kind, _ in bk:
            print("  " + label.ljust(36) + "".join(cell(res.get((label, n, k, m)), "tflops", "%.1f") for m in args.m))
        small = [m for m in args.m if m <= 16]
        if small:
            print(f"  -- weight-stream GB/s for small M (N*K weight bytes / time; decode-relevant, bandwidth measured at 233)")
            for label, kind, _ in bk:
                print("  " + label.ljust(36) + "".join(cell(res.get((label, n, k, m)), "gbps", "%.1f") for m in small))

    print("\n=== best per shape / M")
    for (n, k) in SHAPES:
        for m in args.m:
            parts = []
            for kind in ("nvfp4", "fp8", "bf16"):
                cands = [(r["tflops"], label) for (label, kind_, fn) in bk if kind_ == kind
                         for r in [res.get((label, n, k, m))] if r and "error" not in r and r["verify"].startswith("ok")]
                if cands:
                    t, label = max(cands)
                    parts.append(f"{kind} {t:6.1f} TF ({label.replace(kind + ' ', '')})")
            print(f"  {n}x{k} M={m:<5d} " + "   ".join(parts))

    if errors:
        print("\n=== backends with errors (first occurrence)")
        for label, (_, msg) in errors.items():
            print(f"  {label}: {msg}")

    if args.json:
        out = [dict(backend=label, n=n, k=k, m=m, **r) for (label, n, k, m), r in res.items()]
        with open(args.json, "w") as f:
            json.dump(dict(date=str(datetime.datetime.now()), device=p.name, torch=torch.__version__,
                           iters=args.iters, swizzle=args.swizzle, raster=args.raster, results=out), f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

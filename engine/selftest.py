"""Startup self-test of each matmul path.

A CUTLASS FP4 kernel compiled for the wrong ISA has given wrong answers on sm_121 with no CUDA error. Library upgrades
can also change the dispatch with no warning. Thus, before the server starts, each path runs a small random problem
against an fp32 reference built from independently dequantized operands. A mismatch stops the startup. The test takes
much less than a second.
"""
from __future__ import annotations

import time

import torch

from engine.kernels import ops
from engine.weights.loader import dequant_nvfp4
from engine.weights.quant_emul import fake_quant_nvfp4_unscaled
from engine.weights.quantize import dequant_int, int_global_scale, nvfp4_global_scale, pack5, pack6, quantize_int

TOL = 5e-3


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def run_selftest(verbose: bool = False) -> dict:
    o = ops()
    g = torch.Generator(device="cuda").manual_seed(1234)
    dev = "cuda"
    res = {}
    t0 = time.perf_counter()

    def rand_nvfp4(N, K):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g)
        sf = (torch.rand(N, K // 16, device=dev, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
        return w, sf

    w, sf = rand_nvfp4(512, 1024)
    w2, sf2 = rand_nvfp4(512, 1024)
    w8 = (torch.randn(384, 1024, device=dev, generator=g) * 30).to(torch.float8_e4m3fn)
    rs = torch.rand(384, device=dev, generator=g) * 0.01 + 0.001
    # BF16 GEMV (the GDN b / a gates)
    x3 = torch.randn(3, 1024, device=dev, generator=g).bfloat16()
    wb = torch.randn(96, 1024, device=dev, generator=g).bfloat16()
    out = torch.empty(3, 96, device=dev, dtype=torch.bfloat16)
    o.bf16_gemv(x3, wb, out)
    res["bf16_gemv"] = _rel(out, x3.float() @ wb.float().T)
    # tensor-core skinny GEMM (decode / verify), M = 1 and 12, residual, fp32 out, row scales, SwiGLU
    ws, wss = rand_nvfp4(264, 1024)
    resid = torch.randn(12, 264, device=dev, generator=g).bfloat16()
    for M in (1, 12):
        x = torch.randn(M, 1024, device=dev, generator=g).bfloat16()
        out = torch.empty(M, 264, device=dev)
        o.skinny_nvfp4(x, ws, wss, 0.3, None, out)
        res[f"skinny_nvfp4_m{M}"] = _rel(out, x.float() @ dequant_nvfp4(ws, wss, torch.tensor(0.3), torch.float32).T)
    x = torch.randn(12, 1024, device=dev, generator=g).bfloat16()
    out = torch.empty(12, 512, device=dev, dtype=torch.bfloat16)
    o.skinny_swiglu(x, w, sf, 0.3, w2, sf2, 0.2, out)
    res["skinny_swiglu"] = _rel(out, torch.nn.functional.silu(x.float() @ dequant_nvfp4(w, sf, torch.tensor(0.3), torch.float32).T)
                                * (x.float() @ dequant_nvfp4(w2, sf2, torch.tensor(0.2), torch.float32).T))
    w8s = w8[:264].contiguous()
    out = torch.empty(12, 264, device=dev, dtype=torch.bfloat16)
    o.skinny_fp8(x, w8s, 1.0, resid, out, rs[:264].contiguous())
    res["skinny_fp8_rowscale"] = _rel(out, x.float() @ (w8s.float() * rs[:264, None]).T + resid.float())
    # INT6 / INT5 decode copies
    wf = torch.randn(264, 1024, device=dev, generator=g) * 0.02
    for bits, pack in ((6, pack6), (5, pack5)):
        gs = int_global_scale(float(wf.abs().max()), bits)
        codes, isf = quantize_int(wf, gs, bits)
        lo, hi = pack(codes)
        out = torch.empty(12, 264, device=dev, dtype=torch.bfloat16)
        o.skinny_int(x, lo, hi, isf, gs, None, out)
        res[f"skinny_int{bits}"] = _rel(out, x.float() @ dequant_int(codes, isf, gs, bits).T)
    # NVFP4 x NVFP4 CUTLASS GEMM (prefill), both tiles, with residual
    M, N, K = 384, 256, 1024
    xa = torch.randn(M, K, device=dev, generator=g).bfloat16()
    s_in = nvfp4_global_scale(xa.float())
    a = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
    sfa = torch.empty(o.nvfp4_sf_size(M, K), dtype=torch.uint8, device=dev)
    o.nvfp4_quant(xa, s_in, a, sfa)
    wq, wsf = rand_nvfp4(N, K)
    sfb = torch.empty(o.nvfp4_sf_size(N, K), dtype=torch.uint8, device=dev)
    o.nvfp4_swizzle_sf(wsf.view(torch.uint8), sfb, K)
    resid = torch.randn(M, N, device=dev, generator=g).bfloat16()
    ref = (fake_quant_nvfp4_unscaled(xa, s_in).float() * s_in) @ dequant_nvfp4(wq, wsf, torch.tensor(0.01), torch.float32).T + resid.float()
    for tile in (0, 1):
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        o.nvfp4_gemm(a, sfa, wq, sfb, s_in * 0.01, resid, out, tile)
        res[f"nvfp4_gemm_tile{tile}"] = _rel(out, ref)
    # FP8 W8A8 cuBLASLt (prefill), row-wise scales
    x8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=dev)
    o.fp8_quant(xa, 0.02, x8)
    out = torch._scaled_mm(x8, w8.t(), scale_a=torch.full((M, 1), 0.02, device=dev),
                           scale_b=rs.view(1, -1), out_dtype=torch.bfloat16)
    res["fp8_scaled_mm"] = _rel(out, (x8.float() * 0.02) @ (w8.float() * rs[:, None]).T)
    # decode attention (FP8 KV)
    D = o.HEAD_DIM
    q = torch.randn(1, 24, 1, D, device=dev, generator=g).bfloat16()
    kc = torch.randn(1, 4, 600, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    vc = torch.randn(1, 4, 600, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    out = torch.empty_like(q)
    o.attn_decode(q, kc, vc, torch.tensor([600], dtype=torch.int32, device=dev), out, D ** -0.5)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), kc.float(), vc.float(), enable_gqa=True)
    res["attn_decode_fp8kv"] = _rel(out, ref)
    torch.cuda.synchronize()
    bad = {k: v for k, v in res.items() if not v < TOL}
    if verbose:
        print(f"[selftest] {len(res)} paths in {(time.perf_counter() - t0) * 1e3:.0f} ms: " +
              ", ".join(f"{k} {v:.1e}" for k, v in res.items()))
    if bad:
        raise RuntimeError(f"kernel self-test failed (relative error >= {TOL}): {bad}")
    return res

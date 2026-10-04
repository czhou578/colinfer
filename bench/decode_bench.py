#!/usr/bin/env python3
"""Decode step and speculative cycle times on the real model (PLAN.md 4.7 bench/decode_bench.py).

Plain decode graph (T=1 per slot) at 1-3 slots, and the MTP cycle graph at widths 1-3 and draft lengths k,
at a given context length. COLINFER_SKINNY=0 switches the decode linears to the CUDA-core GEMV.

   uv run python bench/decode_bench.py [--ctx 8192] [--ks 1 3 5 7]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.model.fast import SKINNY, DecodeGraph, load_fast_model, to_fast  # noqa: E402
from engine.weights.loader import resolve  # noqa: E402


def timed(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--requant", action="store_true", help="decode streams the NVFP4 re-quantized attention / GDN projections")
    a = ap.parse_args()
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    model = to_fast(load_fast_model(path), kv_fp8=True)
    if a.requant:
        from engine.model.fast import attach_requant, requant_path
        print(f"requant: {attach_requant(model, requant_path(path))} linears decode from NVFP4")
    from engine.model.prefill import prepare_prefill
    from engine.spec.mtp import Mtp, MtpCycle, MtpState
    prepare_prefill(model)
    print(f"linears: {'tensor-core skinny GEMM' if SKINNY else 'CUDA-core GEMV'}; context {a.ctx}")
    st = model.new_state(3, a.ctx + 64)
    for B in a.widths:
        v = st.view(0, B)
        g = DecodeGraph(model, v)
        v.pos_t.fill_(a.ctx)
        tok = torch.zeros(B, dtype=torch.long, device="cuda")

        def step():
            g.state.pos = 0
            g.step(tok)
            v.pos_t.fill_(a.ctx)
        dt = timed(step)
        print(f"plain decode  width {B}: {dt * 1e3:6.1f} ms/step  -> {B / dt:5.1f} tok/s aggregate")
    mtp = Mtp(model, path, fp8=True, draft_vocab=65536)
    mst = MtpState(model.cfg, a.ctx + 64, "cuda", batch=3, active=st.active)
    for B in a.widths:
        for k in a.ks:
            if B * (k + 1) > 16:
                continue
            v, mv = st.view(0, B), mst.view(0, B)
            cyc = MtpCycle(model, mtp, v, mv, k)

            def run():
                cyc.graph.replay()
                v.pos_t.fill_(a.ctx)
            dt = timed(run)
            print(f"MTP cycle     width {B} k={k}: {dt * 1e3:6.1f} ms/cycle ({B * (k + 1)} verify rows)", flush=True)
            del cyc


if __name__ == "__main__":
    main()

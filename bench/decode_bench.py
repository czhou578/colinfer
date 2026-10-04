#!/usr/bin/env python3
"""decode_bench.py -- end-to-end decode speed of the Phase 2 path (one CUDA graph per step).

Reports ms/token and tok/s at each context length (the KV cache and GDN state are synthetic: the
step cost does not depend on their contents), with the host reading back every token as a real
generation loop does, plus a per-kernel GPU-time breakdown at the last context length.

  uv run python bench/decode_bench.py [--ctx 64 8192 32768 131072] [--steps 40] [--profile]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.model.fast import DecodeGraph, load_fast_model, to_fast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--ctx", type=int, nargs="+", default=[64, 8192, 32768])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--kv-fp8", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="concurrent slots decoded in one step (<= 4)")
    a = ap.parse_args()
    m = to_fast(load_fast_model(a.ckpt), kv_fp8=a.kv_fp8)
    print(f"KV cache: {'fp8 e4m3' if a.kv_fp8 else 'bf16'}")
    st = m.new_state(a.batch, max(a.ctx) + a.steps + 8)
    g = DecodeGraph(m, st)
    tok = torch.full((a.batch,), 42, device="cuda")
    print(f"slots: {a.batch}")
    print(f"{'context':>8s} {'ms/step':>9s} {'tok/s/slot':>11s} {'aggregate':>10s}")
    for ctx in a.ctx:
        st.pos = ctx
        st.pos_t.fill_(ctx)
        for _ in range(3):
            tok = g.step(tok)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            tok = g.step(tok)
            tok.tolist()
        dt = (time.perf_counter() - t0) / a.steps
        print(f"{ctx:8d} {dt * 1e3:9.2f} {1 / dt:11.2f} {a.batch / dt:10.2f}", flush=True)
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                g.step(tok)
            torch.cuda.synchronize()
        ev = [(e.key, e.self_device_time_total / 3, e.count // 3) for e in prof.key_averages()]
        print(f"\nGPU time per step at ctx {a.ctx[-1]}: {sum(t for _, t, _ in ev) / 1e3:.1f} ms")
        for k, t, c in sorted(ev, key=lambda x: -x[1])[:16]:
            print(f"  {t / 1e3:7.2f} ms {c:5d}x  {k[:84]}")


if __name__ == "__main__":
    main()

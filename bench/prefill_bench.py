#!/usr/bin/env python3
"""prefill_bench.py -- prompt throughput and time-to-first-token of the Phase 3 prefill path.

TTFT = prefill of the whole prompt + sampling the first token (the decode graph is already captured,
as in a running server). Prompts are random token ids (cost does not depend on content).

  uv run python bench/prefill_bench.py [--lens 512 2048 8192 32768] [--chunk 2048] [--profile]
Targets (docs/history/baseline.md section 5 / PLAN.md Phase 3): >= 3,500 tok/s at 2k-8k (plan exit 2,500),
TTFT(2k) <= 0.6 s (plan 0.8 s), 32k prompt <= 12 s (plan 16 s).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.model.fast import load_fast_model, to_fast  # noqa: E402
from engine.model.prefill import prefill, prepare_prefill  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--lens", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--profile", action="store_true", help="per-kernel breakdown of one 2048-token prefill")
    ap.add_argument("--profile-at", type=int, default=0, help="--profile: the chunk that follows this many prompt tokens")
    a = ap.parse_args()
    m = to_fast(load_fast_model(a.ckpt))
    prepare_prefill(m)
    st = m.new_state(1, max(a.lens + [a.profile_at + a.chunk]) + 16)
    print(f"chunk {a.chunk}")
    print(f"{'prompt':>7s} {'TTFT s':>8s} {'tok/s':>8s}")
    for L in a.lens:
        ids = torch.randint(0, 200000, (1, L), device="cuda")
        best = None
        for r in range(a.repeats + 1):
            st.reset(); st.pos = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = prefill(m, ids, st, chunk=a.chunk)
            int(logits.argmax())
            dt = time.perf_counter() - t0
            if r > 0:  # first run warms up FlashInfer / Triton JIT for new shapes
                best = dt if best is None else min(best, dt)
        print(f"{L:7d} {best:8.3f} {L / best:8.0f}", flush=True)
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        st.reset(); st.pos = 0
        if a.profile_at:  # the prompt so far, then profile the next chunk
            prefill(m, torch.randint(0, 200000, (1, a.profile_at), device="cuda"), st, chunk=a.chunk)
        ids = torch.randint(0, 200000, (1, a.chunk), device="cuda")
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            prefill(m, ids, st, chunk=a.chunk)
            torch.cuda.synchronize()
        ev = [(e.key, e.self_device_time_total, e.count) for e in prof.key_averages()]
        tot = sum(t for _, t, _ in ev)
        print(f"\nGPU time for a {a.chunk}-token prefill chunk at position {a.profile_at}: {tot / 1e3:.1f} ms")
        for k, t, c in sorted(ev, key=lambda x: -x[1])[:int(os.environ.get("PROFILE_TOP", "18"))]:
            print(f"  {t / 1e3:8.2f} ms {100 * t / tot:5.1f}% {c:5d}x  {k[:80]}")


if __name__ == "__main__":
    main()

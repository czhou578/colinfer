#!/usr/bin/env python3
"""Decode MoE (csrc/moe.cu) weight-streaming rate by the number D of distinct experts in a launch, at the real layer shape
(E = 512, K = 2560, I = 640: 2.76 MB per expert). The bench cycles over 4 copies of a layer and over 8 expert sets, so
nothing runs from L2 (LPDDR5x peak ~238 GB/s). Then it times the full layer (router, routing, shared expert, experts,
combine) with the routing of random router weights, and the prefill path (KernelMoE.prefill: the grouped NVFP4 GEMMs of
csrc/moe_prefill.cu) per chunk of T tokens, with the time of each step at T = 2048.

   ./py bench/moe_bench.py
"""
import itertools

import torch

from bench.timing import timed
from engine.kernels import ops
from engine.model.moe_kernels import TOPK, KernelMoE, random_moe_weights

E, K, I = 512, 2560, 640
GU_BYTES = 2 * I * K // 2 + 2 * I * K // 16  # gate | up weights + scales of one expert
DN_BYTES = K * I // 2 + K * I // 16
CASES = [(1, 10), (8, 10), (8, 20), (8, 40), (8, 75), (16, 75), (16, 138)]  # (M, D)


def forced_ids(M, D, off):
    """M x 10 ids over exactly D distinct experts (10 <= D <= 10 M), no expert twice in a row."""
    ex = (torch.arange(D) * 7 + off) % E  # 7 is odd: D distinct experts
    return ex[torch.arange(M * TOPK) % D].view(M, TOPK).int().cuda()


def main():
    g = torch.Generator(device="cuda").manual_seed(0)
    layers = [KernelMoE(**random_moe_weights(E, K, I, generator=g), stats=True) for _ in range(4)]
    o = ops()
    print(f"E={E} K={K} I={I}: {(GU_BYTES + DN_BYTES) / 1e6:.2f} MB per expert, 4 layer copies x 8 expert sets")
    for M, D in CASES:
        x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
        h = torch.empty(M * TOPK, I, device="cuda", dtype=torch.bfloat16)
        yp = torch.empty(M * TOPK, K, device="cuda")
        sets = [forced_ids(M, D, 61 * i) for i in range(8)]
        res = []
        for name, fn, nbytes in (("gate_up", lambda m, ids: o.moe_gate_up(x, ids, m.w_gu, m.sf_gu, m.gs, h), GU_BYTES),
                                 ("down", lambda m, ids: o.moe_down(h, ids, m.w_dn, m.sf_dn, m.gs, yp), DN_BYTES),
                                 ("both", lambda m, ids: (o.moe_gate_up(x, ids, m.w_gu, m.sf_gu, m.gs, h),
                                                          o.moe_down(h, ids, m.w_dn, m.sf_dn, m.gs, yp)), GU_BYTES + DN_BYTES)):
            it = itertools.cycle([(m, ids) for ids in sets for m in layers])  # a new layer copy each call
            dt = timed(lambda: fn(*next(it)), n=96, warmup=32)
            res.append(f"{name} {dt * 1e6:7.1f} us {D * nbytes / dt / 1e9:5.0f} GB/s")
        print(f"M={M:2d} D={D:3d} {D * (GU_BYTES + DN_BYTES) / 1e6:6.1f} MB | " + " | ".join(res), flush=True)
    from engine.model.fast import capture
    for M in (1, 8, 16):
        x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
        for m in layers:
            m.stats.zero_()
        it = itertools.cycle(layers)
        dt = timed(lambda: next(it)(x), n=96, warmup=8)
        st = torch.stack([m.stats for m in layers]).sum(0).tolist()
        D = st[0] / max(st[1], 1)
        graphs = [capture(lambda m=m: m(x))[0] for m in layers]  # one graph per layer copy: no host overhead
        gi = itertools.cycle(graphs)
        dg = timed(lambda: next(gi).replay(), n=96, warmup=8)
        print(f"full layer M={M:2d}: eager {dt * 1e6:7.1f} us, graph {dg * 1e6:7.1f} us ({dg * 48e3:5.2f} ms per 48 layers), "
              f"mean distinct experts {D:5.1f}", flush=True)
        del graphs
    prefill(layers, g)


def prefill(layers, g):
    from engine.model.moe_kernels import prefill_plan
    o = ops()
    for T in (2048, 4096, 8192):
        x = torch.randn(T, K, device="cuda", generator=g).bfloat16()
        it = itertools.cycle(layers)
        dt = timed(lambda: next(it).prefill(x), n=12, warmup=4)
        print(f"prefill T={T:5d}: {dt * 1e3:6.2f} ms per layer, MoE ceiling {T / dt / 48:6.0f} tok/s over 48 layers", flush=True)
    T, m = 2048, layers[0]
    P = T * TOPK
    x = torch.randn(T, K, device="cuda", generator=g).bfloat16()
    logits = x @ m.router.T
    ids = torch.empty(T, TOPK, dtype=torch.int32, device="cuda")
    w = torch.empty(T, TOPK, device="cuda")
    sg = torch.empty(T, device="cuda")
    o.moe_route(logits, E, ids, w, sg)
    order, off, sfo, inv = prefill_plan(ids, E)
    rows = o.moe_sf_rows(P, E)
    a = torch.empty(P, K // 2, dtype=torch.uint8, device="cuda")
    sfa = torch.empty(rows * K // 16, dtype=torch.uint8, device="cuda")
    d1 = torch.empty(P, 2 * I, dtype=torch.bfloat16, device="cuda")
    a2 = torch.empty(P, I // 2, dtype=torch.uint8, device="cuda")
    sfa2 = torch.empty(rows * I // 16, dtype=torch.uint8, device="cuda")
    d2 = torch.empty(P, K, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
    shared = torch.randn(T, K, device="cuda", generator=g).bfloat16()
    lit = itertools.cycle(layers)
    steps = {
        "router (cuBLAS)": lambda: x @ m.router.T,
        "moe_route": lambda: o.moe_route(logits, E, ids, w, sg),
        "prefill_plan": lambda: prefill_plan(ids, E),
        "gather_quant": lambda: o.moe_gather_quant(x, ids, order, off, sfo, m.in_scale, a, sfa),
        "grouped gate|up": lambda: (lambda L: o.moe_grouped_gemm(0, a, sfa, L.w_gu, L.sf_gu, d1, off, sfo, L.in_scale, L.gs))(next(lit)),
        "swiglu_quant": lambda: o.moe_swiglu_quant(d1, ids, order, off, sfo, m.in_scale, a2, sfa2),
        "grouped down": lambda: (lambda L: o.moe_grouped_gemm(1, a2, sfa2, L.w_dn, L.sf_dn, d2, off, sfo, L.in_scale, L.gs))(next(lit)),
        "shared (cuBLAS)": lambda: (torch.nn.functional.silu(x @ m.sh_gate.T) * (x @ m.sh_up.T)) @ m.sh_down.T,
        "combine": lambda: o.moe_combine_prefill(d2, inv, w, sg, shared, y),
    }
    print("prefill T=2048 by step: " + ", ".join(f"{k} {timed(f, n=20, warmup=4) * 1e3:.3f} ms" for k, f in steps.items()), flush=True)


if __name__ == "__main__":
    main()

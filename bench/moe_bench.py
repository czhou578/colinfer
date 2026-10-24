#!/usr/bin/env python3
"""Decode MoE (csrc/moe.cu) weight-streaming rate by the number D of distinct experts in a launch, at the real layer shape
(E = 512, K = 2560, I = 640: 2.76 MB per expert). The bench cycles over 4 copies of a layer and over 8 expert sets, so
nothing runs from L2 (LPDDR5x peak ~238 GB/s). Then it times the full layer (router, routing, shared expert, experts,
combine) with the routing of random router weights.

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


if __name__ == "__main__":
    main()

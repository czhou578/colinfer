"""The MoE block of qwen4_exp on the decode kernels of csrc/moe.cu (docs/history/qwen4exp_plan.md section 5.2).

KernelMoE.forward(x [M, K]) -> y [M, K] runs one MoE layer: the routed experts and the gated shared expert, with no
residual. The chain of kernels is:

1. skinny_bf16: the router and the shared-expert gate in one weight pass (logits [M, E + 8], bf16).
2. moe_route: the top-10 experts of each row, the renormalized weights and the sigmoid of the shared gate.
3. On a side stream: the shared expert (skinny_swiglu_bf16, then skinny_bf16). With the INT6 decode copy of its gate | up
   (set_shared_copies, tools/requant_qwen4exp.py): skinny_int into [M, 2 Is], silu(gate) * up in fp32 with one rounding
   (elementwise), then the BF16 down (K = Is = 640 fits no block-scaled format of the skinny GEMM: K % 512).
4. moe_gate_up, then moe_down: the expert GEMMs over the (row, slot) pairs.
5. moe_combine: the weighted sum in slot order plus the gated shared expert.

Each output row is bit-identical for any set of rows in the launch (plain decode, verify and the drafter agree). The
skip flag of the draft early exit (ops().skinny_skip) gives a zero output.

KernelMoE.prefill(x [T, K]) runs a prefill chunk on csrc/moe_prefill.cu: the (token, slot) pairs sorted by expert
(prefill_plan) through two CUTLASS grouped NVFP4 GEMMs, one group per expert, with no host synchronization.

moe_reference() is the pure-torch reference (the per-expert loop of transformers, in fp32) for the tests.
skinny_bf16_selftest() and moe_selftest() are the entries of the startup self-test (engine/selftest.py).
Tests: tests/test_moe.py (decode) and tests/test_moe_prefill.py (prefill). Bench: bench/moe_bench.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from engine.kernels import ops
from engine.weights.loader import dequant_nvfp4

TOPK = 10
MAX_ROWS = 16   # rows per launch of the expert kernels: a verify of k + 1 rows must fit (the kernels take up to 48)
SKINNY_ROWS = 16  # rows per launch of the skinny GEMM (csrc/skinny.cu)
ROUTER_PAD = 8  # the stacked router: E expert rows, the shared-expert gate, 7 zero rows (N % 8 == 0 for the skinny GEMM)

_BUFS: dict = {}


def swizzle_expert_scales(sf: torch.Tensor, K: int) -> torch.Tensor:
    """Row-major e4m3 scales [E, R, K/16] of E stacked [R, K] operands -> uint8 [E, R * K/16] in the CUTLASS 128x4 layout
    of each operand (csrc/gemm_nvfp4.cu). R % 128 == 0 and K % 64 == 0, so the operands have no padding. Then the
    swizzle of the stacked [E * R, K] matrix equals the swizzles of the operands one after the other: one launch."""
    E, R, KB = sf.shape
    assert KB * 16 == K and R % 128 == 0 and K % 64 == 0, (sf.shape, K)
    out = torch.empty(E, R * KB, dtype=torch.uint8, device=sf.device)
    ops().nvfp4_swizzle_sf(sf.contiguous().view(torch.uint8).view(E * R, KB), out, K)
    return out


def _buffers(device, E, K, I, Is, rows) -> SimpleNamespace:
    """The work buffers of one launch shape. All layers of a model share them: the layers run one after the other."""
    key = (device, E, K, I, Is, rows)
    if key not in _BUFS:
        f = {"device": device}
        _BUFS[key] = SimpleNamespace(
            logits=torch.empty(rows, E + ROUTER_PAD, dtype=torch.bfloat16, **f),
            ids=torch.empty(rows, TOPK, dtype=torch.int32, **f),
            w=torch.empty(rows, TOPK, dtype=torch.float32, **f),
            sg=torch.empty(rows, dtype=torch.float32, **f),
            h=torch.empty(rows * TOPK, I, dtype=torch.bfloat16, **f),
            yp=torch.empty(rows * TOPK, K, dtype=torch.float32, **f),
            hs=torch.empty(rows, Is, dtype=torch.bfloat16, **f),
            shared=torch.empty(rows, K, dtype=torch.bfloat16, **f))
    return _BUFS[key]


class KernelMoE:
    """One MoE layer on the decode kernels.

    The constructor takes the tensors of a layer as the loader stacks them:
      w_gu      uint8 [E, 2I, K/2]   the gate rows 0..I-1, then the up rows I..2I-1 (packed e2m1)
      sf_gu     e4m3  [E, 2I, K/16]  row-major block scales (swizzled here; this object keeps only the swizzled copy)
      w_dn      uint8 [E, K, I/2]
      sf_dn     e4m3  [E, K, I/16]   row-major
      gs        f32   [E, 3]         weight_scale_2 of gate, up, down
      in_scale  f32   [E, 3]         input_scale of gate, up, down (the prefill path needs it; decode does not)
      router_w  bf16  [E, K];  shared_gate_w bf16 [1, K]
      sh_gate, sh_up bf16 [Is, K];  sh_down bf16 [K, Is]   the shared expert
    The caller drops its references to sf_gu and sf_dn after the constructor, so that the row-major copies are freed.

    stats=True: self.stats int32 [2] counts the distinct experts of each decode launch (stats[0]) and the launches
    (stats[1]).
    """

    def __init__(self, w_gu, sf_gu, w_dn, sf_dn, gs, in_scale, router_w, shared_gate_w, sh_gate, sh_up, sh_down, max_rows: int = MAX_ROWS,
                 stats: bool = False):
        E, I2, K2 = w_gu.shape
        self.E, self.K, self.I = E, K2 * 2, I2 // 2
        K, I = self.K, self.I
        assert w_dn.shape == (E, K, I // 2), w_dn.shape
        assert sf_gu.shape == (E, 2 * I, K // 16) and sf_dn.shape == (E, K, I // 16), (sf_gu.shape, sf_dn.shape)
        assert gs.shape == (E, 3) and in_scale.shape == (E, 3), (gs.shape, in_scale.shape)
        assert router_w.shape == (E, K) and shared_gate_w.shape == (1, K), (router_w.shape, shared_gate_w.shape)
        assert 1 <= max_rows <= 48
        self.Is = sh_gate.shape[0]
        assert sh_gate.shape == sh_up.shape == (self.Is, K) and sh_down.shape == (K, self.Is)
        self.w_gu, self.w_dn = w_gu.contiguous(), w_dn.contiguous()
        self.sf_gu = swizzle_expert_scales(sf_gu, K)
        self.sf_dn = swizzle_expert_scales(sf_dn, I)
        self.gs = gs.float().contiguous()
        self.in_scale = in_scale.float().contiguous()
        pad = torch.zeros(ROUTER_PAD - 1, K, dtype=torch.bfloat16, device=router_w.device)
        self.router = torch.cat([router_w.bfloat16(), shared_gate_w.bfloat16(), pad]).contiguous()
        self.sh_gate, self.sh_up, self.sh_down = (t.bfloat16().contiguous() for t in (sh_gate, sh_up, sh_down))
        self.gate_up_one_scale = bool(torch.equal(self.gs[:, 0], self.gs[:, 1]) and torch.equal(self.in_scale[:, 0], self.in_scale[:, 1]))
        self.max_rows = max_rows
        self.stats = torch.zeros(2, dtype=torch.int32, device=w_gu.device) if stats else None
        self.buf = _buffers(w_gu.device, E, K, I, self.Is, max_rows)
        self.sh_gu_dec = None  # the INT6 decode copy of the shared expert's gate | up (set_shared_copies)

    def set_shared_copies(self, gate_up) -> None:
        """Decode streams gate_up (an IntLinear [2 Is, K]: the gate rows, then the up rows) instead of the BF16 sh_gate and
        sh_up; None restores them. The prefill keeps the BF16 weights."""
        if gate_up is not None and (gate_up.out_features, gate_up.in_features) != (2 * self.Is, self.K):
            raise ValueError(f"the shared gate | up copy is [{gate_up.out_features}, {gate_up.in_features}], not [{2 * self.Is}, {self.K}]")
        self.sh_gu_dec = gate_up
        if gate_up is not None and not hasattr(self.buf, "gu"):
            self.buf.gu = torch.empty(self.buf.hs.shape[0], 2 * self.Is, dtype=torch.bfloat16, device=self.buf.hs.device)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x bf16 [..., K] -> y bf16 [..., K]: the routed experts plus the gated shared expert, no residual. Rows beyond
        max_rows go in further launches of max_rows (each row's result does not change)."""
        shp = x.shape
        x2 = x.reshape(-1, self.K).contiguous()
        y = torch.empty_like(x2)
        for i in range(0, x2.shape[0], self.max_rows):
            self._run(x2[i:i + self.max_rows], y[i:i + self.max_rows])
        return y.view(shp)

    def _run(self, x, y):
        from engine.model.fast import side_stream
        o, b, M = ops(), self.buf, x.shape[0]
        logits, ids, w, sg = b.logits[:M], b.ids[:M], b.w[:M], b.sg[:M]
        h, yp, hs, shared = b.h[:M * TOPK], b.yp[:M * TOPK], b.hs[:M], b.shared[:M]
        for i in range(0, M, SKINNY_ROWS):  # the skinny GEMMs take 16 rows per launch (each row's result does not change)
            o.skinny_bf16(x[i:i + SKINNY_ROWS], self.router, None, logits[i:i + SKINNY_ROWS])
        o.moe_route(logits, self.E, ids, w, sg, self.stats)
        main, side = torch.cuda.current_stream(x.device), side_stream(x.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):  # the shared expert beside the routed experts (a parallel branch in a graph)
            for i in range(0, M, SKINNY_ROWS):
                r = slice(i, i + SKINNY_ROWS)
                d = self.sh_gu_dec
                if d is None:
                    o.skinny_swiglu_bf16(x[r], self.sh_gate, self.sh_up, hs[r])
                else:
                    gu = b.gu[r][:hs[r].shape[0]]
                    o.skinny_int(x[r], d.wlo, d.whi, d.sf, d.gscale, None, gu)
                    hs[r].copy_(torch.nn.functional.silu(gu[:, :self.Is].float()) * gu[:, self.Is:].float())
                o.skinny_bf16(hs[r], self.sh_down, None, shared[r])
        o.moe_gate_up(x, ids, self.w_gu, self.sf_gu, self.gs, h)
        o.moe_down(h, ids, self.w_dn, self.sf_dn, self.gs, yp)
        main.wait_stream(side)
        o.moe_combine(yp, w, sg, shared, y)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x bf16 [T, K] (a prefill chunk, any T) -> y bf16 [T, K] on the grouped GEMM path of csrc/moe_prefill.cu: the
        router on cuBLAS, moe_route, then the pairs sorted by expert through two grouped NVFP4 GEMMs, and the combine in
        slot order. The shared expert runs on cuBLAS in bf16. No host synchronization."""
        assert self.gate_up_one_scale, "the grouped gate | up GEMM takes one alpha per expert: gate and up need the same scales"
        o, E, K, I = ops(), self.E, self.K, self.I
        x = x.reshape(-1, K).contiguous()
        T = x.shape[0]
        P = T * TOPK
        logits = x @ self.router.T
        ids = torch.empty(T, TOPK, dtype=torch.int32, device=x.device)
        w = torch.empty(T, TOPK, dtype=torch.float32, device=x.device)
        sg = torch.empty(T, dtype=torch.float32, device=x.device)
        o.moe_route(logits, E, ids, w, sg)  # no stats: they count the experts of decode launches
        order, off, sfo, inv = prefill_plan(ids, E)
        rows = o.moe_sf_rows(P, E)
        a = torch.empty(P, K // 2, dtype=torch.uint8, device=x.device)
        sfa = torch.empty(rows * K // 16, dtype=torch.uint8, device=x.device)
        o.moe_gather_quant(x, ids, order, off, sfo, self.in_scale, a, sfa)
        d1 = torch.empty(P, 2 * I, dtype=torch.bfloat16, device=x.device)
        o.moe_grouped_gemm(0, a, sfa, self.w_gu, self.sf_gu, d1, off, sfo, self.in_scale, self.gs)
        a2 = torch.empty(P, I // 2, dtype=torch.uint8, device=x.device)
        sfa2 = torch.empty(rows * I // 16, dtype=torch.uint8, device=x.device)
        o.moe_swiglu_quant(d1, ids, order, off, sfo, self.in_scale, a2, sfa2)
        del d1, a, sfa
        d2 = torch.empty(P, K, dtype=torch.bfloat16, device=x.device)
        o.moe_grouped_gemm(1, a2, sfa2, self.w_dn, self.sf_dn, d2, off, sfo, self.in_scale, self.gs)
        shared = (torch.nn.functional.silu(x @ self.sh_gate.T) * (x @ self.sh_up.T)) @ self.sh_down.T
        y = torch.empty(T, K, dtype=torch.bfloat16, device=x.device)
        o.moe_combine_prefill(d2, inv, w, sg, shared, y)
        return y

    def last_routing(self, M: int):
        """(logits, ids, w, sg) of the last launch of M rows (views of the shared buffers: valid until the next launch)."""
        b = self.buf
        return b.logits[:M], b.ids[:M], b.w[:M], b.sg[:M]


def prefill_plan(ids: torch.Tensor, E: int):
    """The sorted pairs of a prefill chunk, on the device without a host sync: ids int32 [T, 10] ->
    order int32 [P] (the pairs sorted by expert, stable: a token's pairs keep the token order), off int32 [E + 1] (the
    first sorted row of each expert), sfo int32 [E + 1] (the first scale row of each group, the groups padded to 128 rows),
    inv int32 [P] (the sorted row of each pair)."""
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    off = torch.searchsorted(flat[order], torch.arange(E + 1, dtype=flat.dtype, device=flat.device), out_int32=True)
    padded = (off[1:] - off[:-1] + 127) // 128 * 128
    sfo = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), padded.cumsum(0).int()])
    inv = torch.empty_like(flat)
    inv[order] = torch.arange(flat.numel(), dtype=flat.dtype, device=flat.device)
    return order.int(), off, sfo, inv


# ------------------------------------------------------------------------------------------------------------ reference
def route_reference(logits: torch.Tensor):
    """The routing of transformers: logits bf16 [M, E] -> ids int64 [M, 10], w f32 [M, 10] (bf16-rounded). The sort is
    stable, so a tie keeps the lower index first."""
    p = torch.softmax(logits.float(), dim=-1)
    v, i = torch.sort(p, dim=-1, descending=True, stable=True)
    v, i = v[:, :TOPK], i[:, :TOPK]
    return i, (v / v.sum(-1, keepdim=True)).bfloat16().float()


def moe_reference(x, w_gu, sf_gu, w_dn, sf_dn, gs, router_w, shared_gate_w, sh_gate, sh_up, sh_down, logits=None):
    """y fp32 [M, K]: the MoE layer as the per-expert loop of transformers, in fp32 on weights from the loader's
    dequant_nvfp4 (row-major scales). logits: bf16 [M, >= E + 1] to use instead of the router product (the tests pass the
    kernel's logits, so that a one-ulp difference cannot change the selection)."""
    E, I2, _ = w_gu.shape
    I = I2 // 2
    xf = x.float()
    if logits is None:
        logits = (xf @ torch.cat([router_w, shared_gate_w]).float().T).bfloat16()
    ids, w = route_reference(logits[:, :E])
    sg = torch.sigmoid(logits[:, E].float()).bfloat16().float()
    y = torch.zeros_like(xf)
    for e in ids.unique().tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        wg = dequant_nvfp4(w_gu[e, :I], sf_gu[e, :I], gs[e, 0], torch.float32)
        wu = dequant_nvfp4(w_gu[e, I:], sf_gu[e, I:], gs[e, 1], torch.float32)
        wd = dequant_nvfp4(w_dn[e], sf_dn[e], gs[e, 2], torch.float32)
        xe = xf[rows]
        ye = (torch.nn.functional.silu(xe @ wg.T) * (xe @ wu.T)) @ wd.T
        y.index_add_(0, rows, ye * w[rows, slots, None])
    shared = (torch.nn.functional.silu(xf @ sh_gate.float().T) * (xf @ sh_up.float().T)) @ sh_down.float().T
    return y + sg[:, None] * shared


def random_moe_weights(E: int, K: int, I: int, Is: int | None = None, generator=None, device="cuda") -> dict:
    """Random tensors of one MoE layer in the loader's layout (the KernelMoE constructor's arguments), for tests, benches
    and the self-test. The scales keep the outputs near unit size for x ~ N(0, 1)."""
    Is = Is or I
    g = generator

    def nvfp4(*shape):
        w = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device=device, generator=g)
        sf = (torch.rand(*shape[:-1], shape[-1] // 16, device=device, generator=g) * 3 + 0.25).to(torch.float8_e4m3fn)
        return w, sf

    def bf16(*shape, fan_in):
        return (torch.randn(*shape, device=device, generator=g) / fan_in ** 0.5).bfloat16()

    w_gu, sf_gu = nvfp4(E, 2 * I, K)
    w_dn, sf_dn = nvfp4(E, K, I)
    base = torch.tensor([0.1 / K ** 0.5, 0.1 / K ** 0.5, 0.1 / I ** 0.5], device=device)
    gs = base * (0.5 + torch.rand(E, 3, device=device, generator=g))
    gs[:, 1] = gs[:, 0]  # the checkpoint's gate and up share weight_scale_2
    in_scale = torch.tensor([0.003, 0.003, 0.004], device=device).expand(E, 3).contiguous()  # e4m3 block scales stay below 448
    return dict(w_gu=w_gu, sf_gu=sf_gu, w_dn=w_dn, sf_dn=sf_dn, gs=gs, in_scale=in_scale, router_w=bf16(E, K, fan_in=K),
                shared_gate_w=bf16(1, K, fan_in=K), sh_gate=bf16(Is, K, fan_in=K), sh_up=bf16(Is, K, fan_in=K), sh_down=bf16(K, Is, fan_in=Is))


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------------------------------------ self-test
def skinny_bf16_selftest(o, g) -> dict:
    """The BF16 skinny GEMM and its SwiGLU against fp32 references (relative errors by name)."""
    dev = "cuda"
    x = torch.randn(12, 1024, device=dev, generator=g).bfloat16()
    w = (torch.randn(264, 1024, device=dev, generator=g) * 0.05).bfloat16()
    w2 = (torch.randn(264, 1024, device=dev, generator=g) * 0.05).bfloat16()
    out = torch.empty(12, 264, device=dev)
    o.skinny_bf16(x, w, None, out)
    res = {"skinny_bf16": _rel(out, x.float() @ w.float().T)}
    out = torch.empty(12, 264, device=dev, dtype=torch.bfloat16)
    o.skinny_swiglu_bf16(x, w, w2, out)
    res["skinny_swiglu_bf16"] = _rel(out, torch.nn.functional.silu(x.float() @ w.float().T) * (x.float() @ w2.float().T))
    return res


def moe_selftest(o, g) -> dict:
    """A small MoE layer (E = 16, K = 1024, I = 256) against references (relative errors by name): the decode path (M = 3)
    against the fp32 loop, and the grouped gate | up GEMM of the prefill path (T = 6) against the emulation of its NVFP4
    operands."""
    from engine.weights.quant_emul import fake_quant_nvfp4_unscaled
    E, K, I = 16, 1024, 256
    wts = random_moe_weights(E, K, I, generator=g)
    moe = KernelMoE(**wts)
    x = torch.randn(3, K, device="cuda", generator=g).bfloat16()
    y = moe(x)
    logits = moe.last_routing(3)[0].clone()
    res = {"moe_decode": _rel(y, moe_reference(x, **{k: v for k, v in wts.items() if k != "in_scale"}, logits=logits))}
    T = 6
    x = torch.randn(T, K, device="cuda", generator=g).bfloat16()
    ids = torch.stack([torch.randperm(E, device="cuda", generator=g)[:TOPK] for _ in range(T)]).int()
    order, off, sfo, _ = prefill_plan(ids, E)
    a = torch.empty(T * TOPK, K // 2, dtype=torch.uint8, device="cuda")
    sfa = torch.empty(o.moe_sf_rows(T * TOPK, E) * K // 16, dtype=torch.uint8, device="cuda")
    o.moe_gather_quant(x, ids, order, off, sfo, moe.in_scale, a, sfa)
    d1 = torch.empty(T * TOPK, 2 * I, dtype=torch.bfloat16, device="cuda")
    o.moe_grouped_gemm(0, a, sfa, moe.w_gu, moe.sf_gu, d1, off, sfo, moe.in_scale, moe.gs)
    ref = torch.empty(T * TOPK, 2 * I, device="cuda")
    for r, p in enumerate(order.tolist()):
        e, s_in = int(ids.view(-1)[p]), float(moe.in_scale[int(ids.view(-1)[p]), 0])
        w = dequant_nvfp4(wts["w_gu"][e], wts["sf_gu"][e], moe.gs[e, 0] * s_in, torch.float32)
        ref[r] = fake_quant_nvfp4_unscaled(x[p // TOPK], s_in).float() @ w.T
    res["moe_prefill_grouped_gemm"] = _rel(d1, ref)
    return res

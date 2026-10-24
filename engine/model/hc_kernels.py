"""Hyper-connections (the gated residual of qwen4_exp) on the kernels of csrc/hc.cu and csrc/norm.cu
(docs/history/qwen4exp_plan.md section 5.4).

The residual stream of a row is x4 [4 x 2560]. Before a sublayer, a GatedResidual normalizes each stream, mixes the four
into the sublayer input y [2560], and computes the injection weights inj [4]; after the sublayer, the output o [2560] goes
back into each stream: x4_s += o * inj_s. The rounding points follow transformers' Qwen4ExpTextGatedResidual: bf16
weights, fp32 math, one rounding per op.

The down | inject product (K = 10240) runs on bf16_gemv for now: it rounds once and gives each row the same bits for any
M. (torch's bf16 matmul does neither: cuBLAS reduces split-K partials in bf16 and picks its kernel by M.) The BF16
skinny GEMM of the MoE work replaces it later."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kernels import ops

STREAMS, HIDDEN, LOWRANK = 4, 2560, 320
GEMV_ROWS = 8  # bf16_gemv takes at most 8 rows


class KernelGatedResidual(torch.nn.Module):
    """norm_w [10240] (the hc_norm weight), w_down [320, 10240], w_inj [4, 10240] or None (the final mixer), w_up
    [10240, 320], all bf16. forward(x4 [M, 10240]) -> (y [M, 2560], inj [M, 4]), or y without w_inj; post(o, inj, x4)
    adds the sublayer output into x4 in place. M <= 16."""

    def __init__(self, norm_w: torch.Tensor, w_down: torch.Tensor, w_inj: torch.Tensor | None, w_up: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm_w = norm_w.contiguous()
        self.w_up = w_up.contiguous()
        self.has_inj = w_inj is not None
        self.w_di = torch.cat([w_down, w_inj], 0).contiguous() if self.has_inj else w_down.contiguous()  # [324 | 320, 10240]

    def forward(self, x4: torch.Tensor):
        M = x4.shape[0]
        k = ops()
        n = torch.empty_like(x4)
        k.hc_norm(x4, self.norm_w, self.eps, n)
        de = torch.empty(M, self.w_di.shape[0], device=x4.device, dtype=torch.bfloat16)
        for i in range(0, M, GEMV_ROWS):
            k.bf16_gemv(n[i:i + GEMV_ROWS], self.w_di, de[i:i + GEMV_ROWS])
        y = torch.empty(M, HIDDEN, device=x4.device, dtype=torch.bfloat16)
        if not self.has_inj:
            k.hc_mix(de, self.w_up, n, y)
            return y
        inj = torch.empty(M, STREAMS, device=x4.device, dtype=torch.bfloat16)
        k.hc_mix(de[:, :LOWRANK], self.w_up, n, y, de[:, LOWRANK:], inj)
        return y, inj

    @staticmethod
    def post(o: torch.Tensor, inj: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        ops().hc_post(o.reshape(-1, HIDDEN), inj, x4)
        return x4


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def gated_residual_reference(x4, norm_w, w_down, w_inj, w_up, eps: float = 1e-6, parts: dict | None = None):
    """transformers' Qwen4ExpTextGatedResidual.forward, op for op, with fp32 math and its bf16 rounding points. The matmuls
    accumulate in fp64 and round once (the exactly rounded product). x4 bf16 [M, 10240] -> (y [M, 2560], inj [M, 4]), or y
    when w_inj is None. parts (optional): receives the intermediates n, d, a, u, mix, p."""
    M = x4.shape[0]
    g = x4.float().view(M, STREAMS, HIDDEN)
    n = _bf((g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)).view(M, -1) * (1.0 + norm_w.float()))
    d = _bf(n.double() @ w_down.double().T)
    a = _bf(F.silu(_bf(d.float() / STREAMS).float()))
    u = _bf(a.double() @ w_up.double().T)
    mix = _bf(torch.sigmoid(u.float()))
    p = _bf(mix.float() * n.float()).float().view(M, STREAMS, HIDDEN)
    y = _bf((((p[:, 0] + p[:, 1]) + p[:, 2]) + p[:, 3]) / STREAMS)
    if parts is not None:
        parts.update(n=n, d=d, a=a, u=u, mix=mix, p=p)
    if w_inj is None:
        return y
    e = _bf(n.double() @ w_inj.double().T)
    inj = _bf(2 * _bf(torch.sigmoid(_bf(e.float() / STREAMS).float())).float())
    return y, inj


def post_reference(o, inj, x4):
    """x4_s + o * inj_s with transformers' rounding: bf16(o * inj_s), then bf16(x4_s + that)."""
    M = x4.shape[0]
    injection = _bf(o.float().view(M, 1, HIDDEN) * inj.float().view(M, STREAMS, 1))
    return _bf(x4.float().view(M, STREAMS, HIDDEN) + injection.float()).view(M, -1)


# ------------------------------------------------------------------------------------------------------------ self-test
def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def hc_selftest(o, g: torch.Generator) -> dict[str, float]:
    """Startup self-test of the hyper-connection kernels (engine/selftest.py style): relative errors of hc_norm + hc_mix +
    hc_post against the torch reference at M = 3. o: ops(); g: a CUDA generator."""
    dev, M = "cuda", 3
    rnd = lambda *s, sc=1.0: (torch.randn(*s, device=dev, generator=g) * sc).bfloat16()  # noqa: E731
    gr = KernelGatedResidual(rnd(STREAMS * HIDDEN, sc=0.2), rnd(LOWRANK, STREAMS * HIDDEN, sc=0.01), rnd(STREAMS, STREAMS * HIDDEN, sc=0.01),
                             rnd(STREAMS * HIDDEN, LOWRANK, sc=0.05))
    x4 = rnd(M, STREAMS * HIDDEN)
    y, inj = gr(x4)
    y_ref, inj_ref = gated_residual_reference(x4, gr.norm_w, gr.w_di[:LOWRANK], gr.w_di[LOWRANK:], gr.w_up)
    o_ = rnd(M, HIDDEN)
    want = post_reference(o_, inj, x4)
    got = KernelGatedResidual.post(o_, inj, x4.clone())
    return {"hc_mix": _rel(y, y_ref), "hc_inj": _rel(inj, inj_ref), "hc_post": _rel(got, want)}

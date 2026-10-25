"""Hyper-connections (the gated residual of qwen4_exp) on the kernels of csrc/hc.cu and csrc/norm.cu
(docs/history/qwen4exp_plan.md section 5.4).

The residual stream of a row is x4 [4 x 2560]. Before a sublayer, a GatedResidual normalizes each stream, mixes the four
into the sublayer input y [2560], and computes the injection weights inj [4]; after the sublayer, the output o [2560] goes
back into each stream: x4_s += o * inj_s. The rounding points follow transformers' Qwen4ExpTextGatedResidual: bf16
weights, fp32 math, one rounding per op.

The down | inject product (K = 10240) runs on bf16_gemv: it rounds once and gives each row the same bits for any M.
(torch's bf16 matmul does neither: cuBLAS reduces split-K partials in bf16 and picks its kernel by M.)

Decode copies (tools/requant_qwen4exp.py, attached by fast_qwen4exp.attach_decode_copies4 through set_decode_copies):
- dec: an INT6 copy of the down | inject stack (fast.IntLinear, padded with zero rows to 328 rows, N % 8 == 0) on the INT
  skinny GEMM (ops().skinny_int). It replaces bf16_gemv, and it also gives each row the same bits for any M.
- w_up_q / w_up_s: an INT8 copy of W_up (int8 [10240, 320], one fp32 scale per row) for the INT8 mode of hc_mix.
The prefill (engine/model/prefill_qwen4exp.py) reads w_di and w_up, which stay BF16.
Tests: tests/test_hc.py. hc_selftest() is an entry of the startup self-test (engine/selftest.py)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kernels import ops

STREAMS, HIDDEN, LOWRANK = 4, 2560, 320
GEMV_ROWS = 8  # bf16_gemv takes at most 8 rows


class KernelGatedResidual(torch.nn.Module):
    """norm_w [10240] (the hc_norm weight), w_down [320, 10240], w_inj [4, 10240] or None (the final mixer), w_up
    [10240, 320], all bf16. forward(x4 [M, 10240]) -> (y [M, 2560], inj [M, 4]), or y without w_inj; post(o, inj, x4)
    adds the sublayer output into x4 in place. M <= 16. set_decode_copies attaches the decode copies (module docstring)."""

    def __init__(self, norm_w: torch.Tensor, w_down: torch.Tensor, w_inj: torch.Tensor | None, w_up: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm_w = norm_w.contiguous()
        self.w_up = w_up.contiguous()
        self.has_inj = w_inj is not None
        self.w_di = torch.cat([w_down, w_inj], 0).contiguous() if self.has_inj else w_down.contiguous()  # [324 | 320, 10240]
        self.dec = None                     # the INT6 copy of w_di (rows: 320 | 324 + 4 zero rows)
        self.w_up_q = self.w_up_s = None    # the INT8 copy of w_up and its row scales

    def set_decode_copies(self, dec=None, w_up_q: torch.Tensor | None = None, w_up_s: torch.Tensor | None = None):
        """dec: an IntLinear over the rows of w_di (plus zero rows up to a multiple of 8); w_up_q int8 [10240, 320] with
        w_up_s fp32 [10240]: W_up = w_up_q * w_up_s per row. A None leaves that copy as it is."""
        if dec is not None:
            n_di = self.w_di.shape[0]
            if dec.in_features != self.w_di.shape[1] or not n_di <= dec.out_features < n_di + 8:
                raise ValueError(f"the down | inject copy is [{dec.out_features}, {dec.in_features}], w_di is {list(self.w_di.shape)}")
            self.dec = dec
        if w_up_q is not None:
            if w_up_q.dtype != torch.int8 or w_up_q.shape != self.w_up.shape or w_up_s is None or w_up_s.shape != (self.w_up.shape[0],):
                raise ValueError(f"the INT8 W_up copy: int8 {list(self.w_up.shape)} and fp32 [{self.w_up.shape[0]}] scales")
            self.w_up_q, self.w_up_s = w_up_q.contiguous(), w_up_s.float().contiguous()

    def forward(self, x4: torch.Tensor):
        M = x4.shape[0]
        k = ops()
        n = torch.empty_like(x4)
        k.hc_norm(x4, self.norm_w, self.eps, n)
        if self.dec is not None:
            de = torch.empty(M, self.dec.out_features, device=x4.device, dtype=torch.bfloat16)
            self.dec.rows(n, None, de)
        else:
            de = torch.empty(M, self.w_di.shape[0], device=x4.device, dtype=torch.bfloat16)
            for i in range(0, M, GEMV_ROWS):
                k.bf16_gemv(n[i:i + GEMV_ROWS], self.w_di, de[i:i + GEMV_ROWS])
        w_up, w_s = (self.w_up, None) if self.w_up_q is None else (self.w_up_q, self.w_up_s)
        y = torch.empty(M, HIDDEN, device=x4.device, dtype=torch.bfloat16)
        if not self.has_inj:
            k.hc_mix(de[:, :LOWRANK], w_up, n, y, w_scale=w_s)
            return y
        inj = torch.empty(M, STREAMS, device=x4.device, dtype=torch.bfloat16)
        k.hc_mix(de[:, :LOWRANK], w_up, n, y, de[:, LOWRANK:LOWRANK + STREAMS], inj, w_s)
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
    """Startup self-test of the hyper-connection kernels (an entry of engine/selftest.py): relative errors of hc_norm +
    hc_mix + hc_post against the torch reference at M = 3, and of hc_mix's INT8 mode (W_up = q * s per row) against the
    reference on the dequantized W_up. o: ops(); g: a CUDA generator."""
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
    sc = gr.w_up.float().abs().amax(1) / 127
    q = torch.round(gr.w_up.float() / sc[:, None]).clamp(-127, 127).to(torch.int8)
    gr.set_decode_copies(w_up_q=q, w_up_s=sc)
    y8 = gr(x4)[0]
    y8_ref = gated_residual_reference(x4, gr.norm_w, gr.w_di[:LOWRANK], gr.w_di[LOWRANK:], q.float() * sc[:, None])[0]
    return {"hc_mix": _rel(y, y_ref), "hc_inj": _rel(inj, inj_ref), "hc_post": _rel(got, want), "hc_mix_int8": _rel(y8, y8_ref)}

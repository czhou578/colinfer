"""FP8-QK causal prefill attention (csrc/attn_prefill.cu) against fp32 attention with Q rounded to e4m3 the same way
(per (token, head) scale amax / 448), and with K / V the e4m3 cache values.
Run: uv run pytest tests/test_attn_prefill_fp8.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
D, Hq, Hkv = 256, 24, 4


def reference(q, k, v, pos):
    """q [1, Hq, T, D] bf16; k, v [1, Hkv, Lmax, D] e4m3 -> [T, Hq, D] fp32."""
    T = q.shape[2]
    L = pos + T
    qq = q[0].float()
    sc = qq.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448
    q8 = (qq / sc).to(torch.float8_e4m3fn).float() * sc
    kk = k[0, :, :L].float().repeat_interleave(Hq // Hkv, 0)
    vv = v[0, :, :L].float().repeat_interleave(Hq // Hkv, 0)
    s = (q8 @ kk.transpose(1, 2)) * D ** -0.5
    s = s.masked_fill(torch.ones(T, L, dtype=torch.bool, device=q.device).triu(pos + 1), float("-inf"))
    return (torch.softmax(s, -1) @ vv).transpose(0, 1)


@pytest.mark.parametrize("T,pos", [(1, 0), (1, 700), (37, 0), (64, 64), (100, 200), (130, 5000), (2048, 0)])
def test_prefill_fp8_matches_reference(T, pos):
    from engine.kernels import ops
    torch.manual_seed(T + pos)
    Lmax = pos + T + 50
    q = torch.randn(1, Hq, T, D, device="cuda").bfloat16()
    k = torch.randn(1, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(1, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
    k[0, :, pos + T:] = float("nan")  # past the context: must never be read into the result
    v[0, :, pos + T:] = float("nan")
    out = torch.empty(T, Hq * D, device="cuda", dtype=torch.bfloat16)
    ops().attn_prefill_fp8(q, k, v, out, pos, D ** -0.5)
    r = reference(q, k, v, pos)
    rel = ((out.view(T, Hq, D).float() - r).norm() / r.norm()).item()
    assert rel < 1e-2, rel

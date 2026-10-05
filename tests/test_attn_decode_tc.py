"""Tensor-core multi-row decode attention (csrc/attn_decode.cu, namespace tc): accuracy against torch SDPA, and every
row bit-identical to a one-row launch at that row's length (what makes speculation output-invariant).
Run: uv run pytest tests/test_attn_decode_tc.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
from test_attn_decode import kv4_values, rand_kv4, ref  # noqa: E402

Hq, Hkv, D = 24, 4, 256


def cache(kind, B, Lmax):
    if kind == "fp8":
        k = (torch.randn(B, Hkv, Lmax, D, device="cuda") * 2).to(torch.float8_e4m3fn)
        v = torch.randn(B, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
        return k, v, k.bfloat16(), v.bfloat16()
    k, v = rand_kv4(B, Hkv, Lmax), rand_kv4(B, Hkv, Lmax)
    return k, v, kv4_values(k), kv4_values(v)


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
@pytest.mark.parametrize("B,T,lens", [(1, 1, [1]), (1, 1, [31]), (1, 1, [1000]), (1, 1, [8192]), (3, 1, [5, 4096, 777]),
                                      (1, 4, [300]), (2, 8, [17, 2000]), (1, 3, [33]), (2, 10, [600, 64]), (1, 16, [1500])])
def test_tc_matches_sdpa(kind, B, T, lens):
    from engine.kernels import ops
    torch.manual_seed(sum(lens) + T)
    Lmax = 8448
    k, v, kr, vr = cache(kind, B, Lmax)
    q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    ops().attn_decode_tc(q, k, v, sl, out, D ** -0.5)
    r = ref(q, kr, vr, lens)
    err = (out.float() - r.float()).abs().max().item()
    assert err <= 1e-2 * max(1.0, r.float().abs().max().item()), err  # P is rounded to f16 (~5e-4 relative), plus a bf16 ulp


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
@pytest.mark.parametrize("T,L", [(4, 37), (8, 4000), (8, 33), (2, 70), (12, 900), (8, 64 * 16 * 3 + 5)])
def test_tc_rows_bit_identical_to_single_row(kind, T, L):
    """Row t of a T-row launch at seq_len L == the one-row launch at seq_len L - (T-1-t), including the gated layout."""
    from engine.kernels import ops
    torch.manual_seed(T * 1000 + L)
    B, Lmax = 2, 4096 + 64 * 16 * 3
    k, v, _, _ = cache(kind, B, Lmax)
    q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
    gate = torch.randn(B, T, Hq, 2 * D, device="cuda").bfloat16()
    sl = torch.tensor([L, L + 3], dtype=torch.int32, device="cuda")
    out = torch.empty(B, T, Hq * D, device="cuda", dtype=torch.bfloat16)
    ops().attn_decode_tc(q, k, v, sl, out, D ** -0.5, gate)
    for t in range(T):
        q1 = q[:, :, t:t + 1].contiguous()
        g1 = gate[:, t:t + 1].contiguous()
        o1 = torch.empty(B, 1, Hq * D, device="cuda", dtype=torch.bfloat16)
        ops().attn_decode_tc(q1, k, v, sl - (T - 1 - t), o1, D ** -0.5, g1)
        assert torch.equal(o1[:, 0], out[:, t]), (t, (o1[:, 0].float() - out[:, t].float()).abs().max().item())

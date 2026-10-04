"""Split-KV decode attention (csrc/attn_decode.cu) against torch SDPA. Run: uv run pytest tests/test_attn_decode.py -q"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def ref(q, k, v, seq_lens):
    B, Hq, T, D = q.shape
    out = torch.empty_like(q)
    for b in range(B):
        L = int(seq_lens[b])
        kk, vv = k[b:b + 1, :, :L], v[b:b + 1, :, :L]
        mask = torch.ones(T, L, dtype=torch.bool, device=q.device).tril(L - T)  # row t sees L - (T-1-t) positions
        out[b:b + 1] = F.scaled_dot_product_attention(q[b:b + 1].float(), kk.float(), vv.float(), attn_mask=mask, enable_gqa=True).to(q.dtype)
    return out


@pytest.mark.parametrize("B,T,lens,splits", [
    (1, 1, [1], 1), (1, 1, [7], 4), (1, 1, [1000], 16), (1, 1, [8192], 32), (1, 1, [33], 64),  # more splits than keys
    (3, 1, [5, 4096, 777], 24), (1, 2, [300], 8), (2, 4, [17, 2000], 16),
])
@pytest.mark.parametrize("kv_fp8", [False, True])
def test_attn_decode(B, T, lens, splits, kv_fp8):
    from engine.kernels import ops
    torch.manual_seed(sum(lens) + T)
    Hq, Hkv, D, Lmax = 24, 4, 256, 8448
    q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
    k = torch.randn(B, Hkv, Lmax, D, device="cuda").bfloat16()
    v = torch.randn(B, Hkv, Lmax, D, device="cuda").bfloat16()
    if kv_fp8:  # the reference sees exactly the values the kernel reads
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    ops().attn_decode(q, k, v, sl, out, splits, D ** -0.5)
    r = ref(q, k.bfloat16(), v.bfloat16(), lens)
    err = (out.float() - r.float()).abs().max().item()
    assert err < 2e-2, err

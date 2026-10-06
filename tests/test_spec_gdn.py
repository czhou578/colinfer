"""Speculative verify / commit GDN kernels must reproduce T sequential single-token decode steps bit for bit."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


@pytest.mark.parametrize("B,T", [(1, 4), (2, 3), (1, 1), (1, 8), (2, 16)])
def test_verify_commit_bit_exact(B, T):
    from engine.kernels import ops
    torch.manual_seed(B * 10 + T)
    Hk, Hv, C = 16, 48, 2 * 16 * 128 + 48 * 128
    mixed = torch.randn(B, T, C, device="cuda").bfloat16()
    z = torch.randn(B, T, Hv * 128, device="cuda").bfloat16()
    bb = torch.randn(B, T, Hv, device="cuda").bfloat16()
    aa = torch.randn(B, T, Hv, device="cuda").bfloat16()
    A_log = (torch.randn(Hv, device="cuda") * 0.5).bfloat16()
    dtb = torch.randn(Hv, device="cuda").bfloat16()
    nw = (torch.randn(128, device="cuda") * 0.2 + 1).bfloat16()
    w = (torch.randn(C, 1, 4, device="cuda") * 0.5).bfloat16()
    conv0 = torch.randn(B, C, 3, device="cuda").bfloat16()
    rec0 = torch.randn(B, Hv, 128, 128, device="cuda") * 0.1
    # reference: T single-token decode steps
    conv, rec = conv0.clone(), rec0.clone()
    ref_out = []
    for t in range(T):
        qkv = torch.empty(B, C, device="cuda", dtype=torch.bfloat16)
        ops().gdn_conv(mixed[:, t].contiguous(), conv, w, qkv)
        o = torch.empty(B, Hv * 128, device="cuda", dtype=torch.bfloat16)
        ops().gdn_delta(qkv, z[:, t].contiguous(), bb[:, t].contiguous(), aa[:, t].contiguous(), A_log, dtb, nw, rec, o, Hk, 1e-6)
        ref_out.append(o)
    ref_out = torch.stack(ref_out, 1)
    # verify: all T outputs, state untouched
    conv_v, rec_v = conv0.clone(), rec0.clone()
    qkv = torch.empty_like(mixed)
    ops().gdn_conv_multi(mixed, conv_v, w, qkv)
    out = torch.empty_like(z)
    ops().gdn_delta_multi(qkv, z, bb, aa, A_log, dtb, nw, rec_v, out, Hk, 1e-6)
    assert torch.equal(out, ref_out)
    assert torch.equal(conv_v, conv0) and torch.equal(rec_v, rec0)
    # commit n = T: state equals T sequential steps; commit n < T: equals n steps
    n = torch.full((B,), T, dtype=torch.int32, device="cuda")
    ops().gdn_conv_commit(mixed, conv_v, n)
    ops().gdn_delta_multi(qkv, z, bb, aa, A_log, dtb, nw, rec_v, out, Hk, 1e-6, n)
    assert torch.equal(conv_v, conv) and torch.equal(rec_v, rec)
    if T > 1:
        conv1, rec1 = conv0.clone(), rec0.clone()
        q1 = torch.empty(B, C, device="cuda", dtype=torch.bfloat16)
        ops().gdn_conv(mixed[:, 0].contiguous(), conv1, w, q1)
        o1 = torch.empty(B, Hv * 128, device="cuda", dtype=torch.bfloat16)
        ops().gdn_delta(q1, z[:, 0].contiguous(), bb[:, 0].contiguous(), aa[:, 0].contiguous(), A_log, dtb, nw, rec1, o1, Hk, 1e-6)
        conv_c, rec_c = conv0.clone(), rec0.clone()
        one = torch.ones(B, dtype=torch.int32, device="cuda")
        ops().gdn_conv_commit(mixed, conv_c, one)
        ops().gdn_delta_multi(qkv, z, bb, aa, A_log, dtb, nw, rec_c, out, Hk, 1e-6, one)
        assert torch.equal(conv_c, conv1) and torch.equal(rec_c, rec1)


def test_strided_views_bit_exact():
    """mixed / z as column slices of one projection output and b / a of one [.., 2 Hv] output (no copies)."""
    from engine.kernels import ops
    torch.manual_seed(7)
    B, T, Hk, Hv = 2, 5, 16, 48
    C, Z = 2 * Hk * 128 + Hv * 128, Hv * 128
    proj = torch.randn(B, T, C + Z, device="cuda").bfloat16()
    ba = torch.randn(B, T, 2 * Hv, device="cuda").bfloat16()
    mixed, z = proj[..., :C], proj[..., C:]
    b, a = ba[..., :Hv], ba[..., Hv:]
    A_log = (torch.randn(Hv, device="cuda") * 0.5).bfloat16()
    dtb = torch.randn(Hv, device="cuda").bfloat16()
    nw = (torch.randn(128, device="cuda") * 0.2 + 1).bfloat16()
    w = (torch.randn(C, 1, 4, device="cuda") * 0.5).bfloat16()
    conv0 = torch.randn(B, C, 3, device="cuda").bfloat16()
    rec0 = torch.randn(B, Hv, 128, 128, device="cuda") * 0.1
    n = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    res = []
    for m_, z_, b_, a_ in ((mixed, z, b, a), (mixed.contiguous(), z.contiguous(), b.contiguous(), a.contiguous())):
        conv, rec = conv0.clone(), rec0.clone()
        qkv = torch.empty(B, T, C, device="cuda", dtype=torch.bfloat16)
        ops().gdn_conv_multi(m_, conv, w, qkv)
        out = torch.empty(B, T, Z, device="cuda", dtype=torch.bfloat16)
        ops().gdn_delta_multi(qkv, z_, b_, a_, A_log, dtb, nw, rec, out, Hk, 1e-6)
        ops().gdn_conv_commit(m_, conv, n)
        ops().gdn_delta_multi(qkv, z_, b_, a_, A_log, dtb, nw, rec, out[:0].new_empty(out.shape), Hk, 1e-6, n)
        res.append((out, conv, rec))
    for x, y in zip(*res):
        assert torch.equal(x, y)

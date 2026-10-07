"""GDN verify / commit (csrc/gdn_step.cu over T tokens) must reproduce T sequential single-token decode steps bit for
bit. The test reads mixed / z / b / a as strided column views of the projection outputs.
Run: uv run pytest tests/test_spec_gdn.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

Hk, Hv = 16, 48
C, Z = 2 * Hk * 128 + Hv * 128, Hv * 128


def _inputs(B, T, seed):
    torch.manual_seed(seed)
    proj = torch.randn(B, T, C + Z, device="cuda").bfloat16()   # [mixed | z], as the stacked qkv / z projection writes it
    ba = torch.randn(B, T, 2 * Hv, device="cuda").bfloat16()    # [b | a]
    p = dict(A_log=(torch.randn(Hv, device="cuda") * 0.5).bfloat16(), dt_bias=torch.randn(Hv, device="cuda").bfloat16(),
             norm_w=(torch.randn(128, device="cuda") * 0.2 + 1).bfloat16(), w=(torch.randn(C, 1, 4, device="cuda") * 0.5).bfloat16())
    conv0 = torch.randn(B, C, 3, device="cuda").bfloat16()
    rec0 = torch.randn(B, Hv, 128, 128, device="cuda") * 0.1
    return proj, ba, p, conv0, rec0


def _views(proj, ba):
    return proj[..., :C], proj[..., C:], ba[..., :Hv], ba[..., Hv:]


def _step(ops, mixed, z, b, a, p, conv, rec, n, out=True):
    """one call over all T tokens of mixed: outputs (if out) and the state advanced by n"""
    B, T = mixed.shape[:2]
    qkv = torch.empty(B, T, C, device="cuda", dtype=torch.bfloat16)
    ops.gdn_conv(mixed, conv, p["w"], qkv)
    if n is not None:
        ops.gdn_conv_commit(mixed, conv, n)
    o = torch.empty(B, T, Z, device="cuda", dtype=torch.bfloat16) if out else None
    ops.gdn_delta(qkv, z, b, a, p["A_log"], p["dt_bias"], p["norm_w"], rec, o, Hk, 1e-6, n)
    return o, qkv


@pytest.mark.parametrize("B,T", [(1, 4), (2, 3), (1, 1), (1, 8), (2, 16)])
def test_verify_commit_bit_exact(B, T):
    from engine.kernels import ops
    o_ = ops()
    proj, ba, p, conv0, rec0 = _inputs(B, T, B * 10 + T)
    mixed, z, b, a = _views(proj, ba)
    one = torch.ones(B, dtype=torch.int32, device="cuda")
    # reference: T single-token decode steps
    conv, rec = conv0.clone(), rec0.clone()
    tok = lambda x, t: x[:, t:t + 1].contiguous()  # noqa: E731  (one token of every slot)
    ref = torch.cat([_step(o_, tok(mixed, t), tok(z, t), tok(b, t), tok(a, t), p, conv, rec, one)[0] for t in range(T)], 1)
    # verify: all T outputs, state untouched
    conv_v, rec_v = conv0.clone(), rec0.clone()
    out, qkv = _step(o_, mixed, z, b, a, p, conv_v, rec_v, None)
    assert torch.equal(out, ref)
    assert torch.equal(conv_v, conv0) and torch.equal(rec_v, rec0)
    # commit all T: the sequential state
    full = torch.full((B,), T, dtype=torch.int32, device="cuda")
    o_.gdn_conv_commit(mixed, conv_v, full)
    o_.gdn_delta(qkv, z, b, a, p["A_log"], p["dt_bias"], p["norm_w"], rec_v, None, Hk, 1e-6, full)
    assert torch.equal(conv_v, conv) and torch.equal(rec_v, rec)
    # commit a per-slot prefix (and 0 for the last slot when B > 1): the state after that many steps
    n = torch.tensor([max(1, T - 1 - i) for i in range(B - 1)] + [0 if B > 1 else 1], dtype=torch.int32, device="cuda")
    conv_c, rec_c = conv0.clone(), rec0.clone()
    o_.gdn_conv_commit(mixed, conv_c, n)
    o_.gdn_delta(qkv, z, b, a, p["A_log"], p["dt_bias"], p["norm_w"], rec_c, None, Hk, 1e-6, n)
    for i in range(B):
        c1, r1 = conv0[i:i + 1].clone(), rec0[i:i + 1].clone()
        for t in range(int(n[i])):
            _step(o_, mixed[i:i + 1, t:t + 1], z[i:i + 1, t:t + 1], b[i:i + 1, t:t + 1], a[i:i + 1, t:t + 1], p, c1, r1, one[:1])
        assert torch.equal(conv_c[i:i + 1], c1) and torch.equal(rec_c[i:i + 1], r1)


def test_strided_views_match_contiguous():
    from engine.kernels import ops
    o_ = ops()
    B, T = 2, 5
    proj, ba, p, conv0, rec0 = _inputs(B, T, 7)
    n = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    res = []
    for mixed, z, b, a in (_views(proj, ba), [t.contiguous() for t in _views(proj, ba)]):
        conv, rec = conv0.clone(), rec0.clone()
        out, _ = _step(o_, mixed, z, b, a, p, conv, rec, n)
        res.append((out, conv, rec))
    for x, y in zip(*res):
        assert torch.equal(x, y)


def test_plain_step_over_several_slots():
    """Plain decode of B slots (T = 1): mixed / z are [B, 1, C] views of one [B, C + Z] projection output, whose size-1 dim
    may carry any stride. Each slot must match its own single-slot step."""
    from engine.kernels import ops
    o_ = ops()
    B = 3
    proj, ba, p, conv0, rec0 = _inputs(B, 1, 11)
    flat = proj.view(B, C + Z)
    mixed, z = flat[:, :C].reshape(B, 1, C), flat[:, C:].reshape(B, 1, Z)
    b, a = ba[..., :Hv], ba[..., Hv:]
    act = torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
    conv, rec = conv0.clone(), rec0.clone()
    out, _ = _step(o_, mixed, z, b, a, p, conv, rec, act)
    for i in range(B):
        c1, r1 = conv0[i:i + 1].clone(), rec0[i:i + 1].clone()
        o1, _ = _step(o_, mixed[i:i + 1].contiguous(), z[i:i + 1].contiguous(), b[i:i + 1].contiguous(), a[i:i + 1].contiguous(), p, c1, r1,
                      act[i:i + 1])
        assert torch.equal(out[i:i + 1], o1) and torch.equal(conv[i:i + 1], c1) and torch.equal(rec[i:i + 1], r1)

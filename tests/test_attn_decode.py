"""Decode attention (csrc/attn_decode.cu) over an fp8 KV cache: accuracy against torch SDPA, every row bit-identical to a
one-row launch at that row's length (what makes speculation output-invariant), and the full KernelAttention layer
(prologue + attention + gate) against the reference qwen35.Attention. Run: uv run pytest tests/test_attn_decode.py -q"""
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


Hq, Hkv, D = 24, 4, 256


def cache(B, Lmax):
    k = (torch.randn(B, Hkv, Lmax, D, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = torch.randn(B, Hkv, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
    return k, v, k.bfloat16(), v.bfloat16()  # the reference sees exactly the values the kernel reads


@pytest.mark.parametrize("B,T,lens", [(1, 1, [1]), (1, 1, [31]), (1, 1, [1000]), (1, 1, [8192]), (3, 1, [5, 4096, 777]),
                                      (1, 4, [300]), (2, 8, [17, 2000]), (1, 3, [33]), (2, 10, [600, 64]), (1, 16, [1500])])
def test_attn_decode_matches_sdpa(B, T, lens):
    from engine.kernels import ops
    torch.manual_seed(sum(lens) + T)
    Lmax = 8448
    k, v, kr, vr = cache(B, Lmax)
    q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    ops().attn_decode(q, k, v, sl, out, D ** -0.5)
    r = ref(q, kr, vr, lens)
    err = (out.float() - r.float()).abs().max().item()
    assert err <= 1e-2 * max(1.0, r.float().abs().max().item()), err  # P is rounded to f16 (~5e-4 relative), plus a bf16 ulp


@pytest.mark.parametrize("T,L", [(4, 37), (8, 4000), (8, 33), (2, 70), (12, 900), (8, 64 * 16 * 3 + 5)])
def test_rows_bit_identical_to_single_row(T, L):
    """Row t of a T-row launch at seq_len L == the one-row launch at seq_len L - (T-1-t), including the gated layout."""
    from engine.kernels import ops
    torch.manual_seed(T * 1000 + L)
    B, Lmax = 2, 4096 + 64 * 16 * 3
    k, v, _, _ = cache(B, Lmax)
    q = torch.randn(B, Hq, T, D, device="cuda").bfloat16()
    gate = torch.randn(B, T, Hq, 2 * D, device="cuda").bfloat16()
    sl = torch.tensor([L, L + 3], dtype=torch.int32, device="cuda")
    out = torch.empty(B, T, Hq * D, device="cuda", dtype=torch.bfloat16)
    ops().attn_decode(q, k, v, sl, out, D ** -0.5, gate)
    for t in range(T):
        q1 = q[:, :, t:t + 1].contiguous()
        g1 = gate[:, t:t + 1].contiguous()
        o1 = torch.empty(B, 1, Hq * D, device="cuda", dtype=torch.bfloat16)
        ops().attn_decode(q1, k, v, sl - (T - 1 - t), o1, D ** -0.5, g1)
        assert torch.equal(o1[:, 0], out[:, t]), (t, (o1[:, 0].float() - out[:, t].float()).abs().max().item())


class _LinearR(torch.nn.Linear):
    def forward(self, x, residual=None):
        y = super().forward(x)
        return y if residual is None else y + residual


@pytest.mark.parametrize("B,T,pos", [(1, 1, 37), (2, 1, 500), (1, 3, 64)])
def test_kernel_attention_layer_matches_reference(B, T, pos):
    """Fused prologue (q/k norm, RoPE, fp8 KV write) + multi-row attention + gated combine vs qwen35.Attention."""
    from engine.model.fast import FastState, KernelAttention
    from engine.model.qwen35 import Attention, ModelState, Qwen35Config
    torch.manual_seed(B * 100 + T + pos)
    cfg = Qwen35Config(layer_types=["full_attention"], num_hidden_layers=1)
    ref = Attention(cfg).cuda().bfloat16()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        old = getattr(ref, name)
        new = _LinearR(old.in_features, old.out_features, bias=False).cuda().bfloat16()
        setattr(ref, name, new)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) * (0.02 if "proj" in n else 0.3))
    rs = ModelState(cfg, B, 1024, "cuda")
    fs = FastState(cfg, B, 1024, "cuda")
    hist_k = torch.randn_like(rs.k[0][:, :, :pos]) * 2
    hist_v = torch.randn_like(rs.v[0][:, :, :pos])
    # both sides see the same (fp8-representable) history
    hist_k, hist_v = hist_k.to(torch.float8_e4m3fn).bfloat16(), hist_v.to(torch.float8_e4m3fn).bfloat16()
    rs.k[0][:, :, :pos], rs.v[0][:, :, :pos] = hist_k, hist_v
    fs.k[0][:, :, :pos] = hist_k.to(fs.k[0].dtype)
    fs.v[0][:, :, :pos] = hist_v.to(fs.v[0].dtype)
    rs.pos = fs.pos = pos
    fs.pos_t.fill_(pos)
    x = torch.randn(B, T, cfg.hidden_size, device="cuda").bfloat16()
    res = torch.randn_like(x)
    d = cfg.rotary_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device="cuda") / d))
    positions = torch.arange(pos, pos + T, device="cuda").float()
    emb = torch.cat([positions[:, None] * inv] * 2, -1)
    cos, sin = emb.cos().bfloat16()[None].expand(B, -1, -1), emb.sin().bfloat16()[None].expand(B, -1, -1)
    with torch.inference_mode():
        want = ref(x, cos, sin, rs, 0) + res
        ka = Attention(cfg).cuda().bfloat16()
        ka.load_state_dict(ref.state_dict(), strict=False)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(ka, name, getattr(ref, name))
        ka.__class__ = KernelAttention
        ka.inv_freq = inv
        got = ka(x, None, None, fs, 0, residual=res)
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert rel < 3e-2, rel  # the new rows' K / V are rounded to e4m3 on one side only

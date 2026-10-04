"""Fused GDN decode step (csrc/gdn_step.cu) against the Phase 1 PyTorch GatedDeltaNet (T = 1, existing
state), at Qwen3.8-27B dimensions. Run: uv run pytest tests/test_gdn_step.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.model.qwen35 import GatedDeltaNet, ModelState, Qwen35Config  # noqa: E402


@pytest.mark.parametrize("B", [1, 3])
def test_gdn_step_matches_reference(B):
    from engine.kernels import ops
    torch.manual_seed(B)
    cfg = Qwen35Config(layer_types=["linear_attention"], num_hidden_layers=1)
    layer = GatedDeltaNet(cfg).cuda().bfloat16()
    with torch.no_grad():
        for n, p in layer.named_parameters():
            p.copy_(torch.randn_like(p) * (0.02 if "proj" in n else 0.5))
    st = ModelState(cfg, B, 8, "cuda")
    st.conv[0].copy_(torch.randn_like(st.conv[0]))
    st.rec[0].copy_(torch.randn_like(st.rec[0]) * 0.1)
    st.pos = 5
    x = torch.randn(B, 1, cfg.hidden_size, device="cuda").bfloat16()
    conv0, rec0 = st.conv[0].clone(), st.rec[0].clone()
    with torch.inference_mode():
        ref = layer(x, st, 0)
        ref_conv, ref_rec = st.conv[0].clone(), st.rec[0].clone()
        conv, rec = conv0.clone(), rec0.clone()
        mixed = layer.in_proj_qkv(x)[:, 0].contiguous()
        z = layer.in_proj_z(x)[:, 0].contiguous()
        b = layer.in_proj_b(x)[:, 0].contiguous()
        a = layer.in_proj_a(x)[:, 0].contiguous()
        qkv = torch.empty_like(mixed)
        ops().gdn_conv(mixed, conv, layer.conv1d.weight.contiguous(), qkv)
        o = torch.empty_like(z)
        ops().gdn_delta(qkv, z, b, a, layer.A_log.data, layer.dt_bias.data, layer.norm.weight.data, rec, o, cfg.linear_num_key_heads, cfg.rms_norm_eps)
        out = layer.out_proj(o)[:, None]
    assert torch.equal(conv, ref_conv)
    torch.testing.assert_close(rec, ref_rec, rtol=1e-4, atol=1e-4)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1e-2, rel

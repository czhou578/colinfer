"""The GDN decode step (csrc/gdn_step.cu: gdn_conv + gdn_conv_commit + gdn_delta, T = 1) against the reference PyTorch
GatedDeltaNet (engine/model/qwen35.py) on an existing state, at Qwen3.8-27B dimensions; inactive slots keep their state.
Run: uv run pytest tests/test_gdn_step.py -q"""
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
        active = torch.ones(B, dtype=torch.int32, device="cuda")
        ops().gdn_conv_commit(mixed, conv, active)
        o = torch.empty_like(z)
        ops().gdn_delta(qkv, z, b, a, layer.A_log.data, layer.dt_bias.data, layer.norm.weight.data, rec, o, cfg.linear_num_key_heads, cfg.rms_norm_eps,
                        active)
        out = layer.out_proj(o)[:, None]
        # an inactive slot: same outputs, state untouched
        conv_i, rec_i = conv0.clone(), rec0.clone()
        idle = torch.zeros(B, dtype=torch.int32, device="cuda")
        ops().gdn_conv_commit(mixed, conv_i, idle)
        o_i = torch.empty_like(z)
        ops().gdn_delta(qkv, z, b, a, layer.A_log.data, layer.dt_bias.data, layer.norm.weight.data, rec_i, o_i, cfg.linear_num_key_heads,
                        cfg.rms_norm_eps, idle)
    assert torch.equal(conv, ref_conv)
    torch.testing.assert_close(rec, ref_rec, rtol=1e-4, atol=1e-4)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1e-2, rel
    assert torch.equal(o_i, o) and torch.equal(conv_i, conv0) and torch.equal(rec_i, rec0)

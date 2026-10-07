"""Tests for engine/weights/quant_emul.py. The FlashInfer comparison needs the GPU, and pytest skips it without one.
Run: uv run pytest tests/test_quant_emul.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.weights.loader import E2M1_LUT  # noqa: E402
from engine.weights.quant_emul import (e2m1_round, fake_quant_fp8_unscaled, fake_quant_nvfp4_unscaled,  # noqa: E402
                                       parse_effects, requant_fused_fp8)


def test_e2m1_round_ties_to_even_and_saturation():
    a = torch.tensor([0.0, 0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 7.0, 100.0])
    want = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0])
    assert torch.equal(e2m1_round(a), want)


def test_nvfp4_unscaled_is_exact_in_bf16():
    torch.manual_seed(0)
    x = torch.randn(8, 256).bfloat16()
    q = fake_quant_nvfp4_unscaled(x, 0.003)
    assert torch.equal(q.float().bfloat16().float(), fake_quant_nvfp4_unscaled(x.float(), 0.003))


def test_fp8_static_clamps():
    x = torch.tensor([[1e6, -1e6, 0.0, 1.0]]).bfloat16()
    assert torch.equal(fake_quant_fp8_unscaled(x, 1.0).float(), torch.tensor([[448.0, -448.0, 0.0, 1.0]]))


def test_requant_fused_fp8_moves_to_max_scale():
    meta = {f"layers.3.self_attn.{p}_proj": dict(kind="fp8", w_scale=s, in_scale=1.0) for p, s in (("q", 1.0), ("k", 0.5), ("v", 0.25))}
    sd = {k + ".weight": torch.tensor([[2.0, 3.0, 448.0]]).bfloat16() for k in meta}
    assert requant_fused_fp8(meta, sd) == 2
    assert all(m["w_scale"] == 1.0 for m in meta.values())
    assert torch.equal(sd["layers.3.self_attn.k_proj.weight"].float(), torch.tensor([[1.0, 1.5, 224.0]]))
    assert torch.equal(sd["layers.3.self_attn.v_proj.weight"].float(), torch.tensor([[0.5, 0.75, 112.0]]))


def test_parse_effects():
    assert parse_effects("all") == {"act_nvfp4", "act_fp8", "fp8_requant"}
    assert parse_effects(None) == set()
    with pytest.raises(ValueError):
        parse_effects("act_fp16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
def test_nvfp4_matches_flashinfer_bitwise():
    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(0)
    x = torch.randn(128, 5120, device="cuda")
    x[torch.rand_like(x) < 0.001] *= 50
    x = x.bfloat16()
    s = float(x.float().abs().max()) / (6 * 448) * 0.7
    q, sf = flashinfer.fp4_quantize(x, torch.tensor([1.0 / s], device="cuda"), sf_vec_size=16, is_sf_swizzled_layout=False)
    q = q.view(torch.uint8)
    sf = sf.view(torch.float8_e4m3fn).float().reshape(x.shape[0], -1)[:, : x.shape[1] // 16]
    lut = E2M1_LUT.cuda()
    ref = torch.stack([lut[(q & 15).long()], lut[(q >> 4).long()]], -1).reshape(x.shape) * sf.repeat_interleave(16, 1)
    assert torch.equal(fake_quant_nvfp4_unscaled(x, s).float(), ref)

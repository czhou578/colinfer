"""Tests for engine/weights/quant_emul.py. The FlashInfer comparison needs the GPU, and pytest skips it without one.
Run: uv run pytest tests/test_quant_emul.py -q"""
import pytest
import torch

from engine.weights.loader import E2M1_LUT
from engine.weights.quant_emul import e2m1_round, fake_quant_nvfp4_unscaled


def test_e2m1_round_ties_to_even_and_saturation():
    a = torch.tensor([0.0, 0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 7.0, 100.0])
    want = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0])
    assert torch.equal(e2m1_round(a), want)


def test_nvfp4_unscaled_is_exact_in_bf16():
    torch.manual_seed(0)
    x = torch.randn(8, 256).bfloat16()
    q = fake_quant_nvfp4_unscaled(x, 0.003)
    assert torch.equal(q.float().bfloat16().float(), fake_quant_nvfp4_unscaled(x.float(), 0.003))


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

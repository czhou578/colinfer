"""Tests for engine/weights/quant_emul.py (CPU). Run: uv run pytest tests/test_quant_emul.py -q"""
import torch

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

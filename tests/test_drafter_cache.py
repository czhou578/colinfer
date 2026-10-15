"""The drafter's quantized parts round-trip through its cache (engine/spec/mtp.py): a DraftLinear rebuilt from saved
parts computes what the one quantized from BF16 computes, bit for bit, for draft steps and for the prompt pass."""
import pytest
import torch
from safetensors.torch import save_file

from engine.weights.direct import read_tensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def test_a_linear_from_cached_parts_matches(tmp_path):
    from engine.spec.mtp import DraftLinear
    torch.manual_seed(0)
    w = (torch.randn(256, 512, device="cuda") * 0.05).bfloat16()
    d = DraftLinear(w)
    path = str(tmp_path / "quant.safetensors")
    save_file({"x.w": d.low.w.cpu(), "x.sf": d.low.sf.cpu(), "x.gs": torch.tensor(d.low.gscale, dtype=torch.float64)}, path)
    q = read_tensors(path, device="cuda")
    e = DraftLinear.from_parts(q["x.w"], q["x.sf"], float(q["x.gs"]))
    assert (e.out_features, e.in_features) == (256, 512) and e.low.gscale == d.low.gscale
    assert torch.equal(e.low.w, d.low.w) and torch.equal(e.low.sf.view(torch.uint8), d.low.sf.view(torch.uint8))
    x = torch.randn(3, 512, device="cuda").bfloat16()
    assert torch.equal(e(x), d(x))  # the draft steps (skinny GEMM)
    xp = torch.randn(40, 512, device="cuda").bfloat16()
    assert torch.equal(e(xp), d(xp))  # the prompt pass (dequantized)

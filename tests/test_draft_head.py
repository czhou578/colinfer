"""Low-rank draft head pieces (engine/spec/mtp.py, docs/history/phase6_progress.md section 19)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def test_rescore_nvfp4_matches_dequantized_rows():
    from engine.kernels import ops
    from engine.weights.loader import dequant_nvfp4
    from engine.weights.quantize import quantize
    torch.manual_seed(0)
    V, K, B, C = 3000, 5120, 3, 64
    w = torch.randn(V, K, device="cuda") * 0.02
    gs = float(w.abs().max()) / 2688
    p, sf = quantize(w, gs)
    W = dequant_nvfp4(p, sf, torch.tensor(gs), out_dtype=torch.float32)
    x = torch.randn(B, K, device="cuda").bfloat16()
    cand = torch.randint(0, V, (B, C), device="cuda")
    out = ops().rescore_nvfp4(x, p, sf, gs, cand)
    ref = torch.stack([W[cand[b]] @ x[b].float() for b in range(B)])
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_rescore_bf16_matches_the_rows_and_is_row_invariant():
    """ops().rescore_bf16 (the rescoring of the low-rank draft head of qwen4_exp, over its BF16 lm_head) equals the fp32
    products of the candidate rows, and a row's logits do not depend on the other rows of the launch."""
    from engine.kernels import ops
    torch.manual_seed(1)
    V, K, B, C = 5000, 2560, 4, 256
    w = (torch.randn(V, K, device="cuda") * 0.02).bfloat16()
    x = torch.randn(B, K, device="cuda").bfloat16()
    cand = torch.randint(0, V, (B, C), device="cuda")
    out = ops().rescore_bf16(x, w, cand)
    ref = torch.stack([w[cand[b]].float() @ x[b].float() for b in range(B)])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert torch.equal(ops().rescore_bf16(x[2:3].contiguous(), w, cand[2:3].contiguous()), out[2:3])
    assert torch.equal(ops().rescore_bf16(x, w, cand[:, :7].contiguous()), out[:, :7])
    with pytest.raises(RuntimeError):
        ops().rescore_bf16(x, w[:, :K - 4].contiguous(), cand)  # K must match x

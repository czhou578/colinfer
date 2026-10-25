"""The calibrated decode copies of tools/gptq_qwen4exp.py: GPTQ in the engine's INT6 / INT5 layout, and the collection of
the input second moments."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def correlated(n: int, K: int, seed: int) -> torch.Tensor:
    """n input rows of K dims with a few strong directions and outlier dims, as a layer's activations have."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    basis = torch.randn(32, K, device="cuda", generator=g)
    x = torch.randn(n, 32, device="cuda", generator=g) @ basis + 0.3 * torch.randn(n, K, device="cuda", generator=g)
    x[:, :8] *= 6
    return x


@pytest.mark.parametrize("bits,act_order", [(6, False), (5, False), (6, True)])
def test_gptq_beats_round_to_nearest_in_the_engines_layout(bits, act_order):
    from engine.weights.quantize import dequant_int, int_global_scale, quantize_int
    from tools.gptq_qwen4exp import gptq, output_error
    N, K = 384, 512
    W = torch.randn(N, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)) * 0.02
    x = correlated(5120, K, 2)
    x, xt = x[:4096], x[4096:]  # calibration rows, and held-out rows of the same distribution
    H = x.T @ x / x.shape[0]
    gs = int_global_scale(float(W.abs().max()), bits)
    codes, sf = gptq(W, H, gs, bits, act_order=act_order)
    assert codes.dtype == torch.uint8 and sf.dtype == torch.float8_e4m3fn and sf.shape == (N, K // 16)
    q = codes.int() - 2 ** (bits - 1)
    assert int(q.abs().max()) <= 2 ** (bits - 1) - 1
    deq = dequant_int(codes, sf, gs, bits)
    rc, rs = quantize_int(W, gs, bits)
    e, e_rtn = output_error(W, deq, H), output_error(W, dequant_int(rc, rs, gs, bits), H)
    assert e < 0.7 * e_rtn, (e, e_rtn)  # the output error on the inputs falls; the weight error may grow
    assert ((xt @ (deq - W).T).norm() / (xt @ (dequant_int(rc, rs, gs, bits) - W).T).norm()) < 0.85


def test_quantize_group_writes_the_requant_layout():
    """The file tensors of a stacked group equal those that tools/requant_qwen4exp.py writes (names, shapes, dtypes, one
    global scale), and dequantize() of that tool reads them back to the GPTQ weights."""
    from tools.gptq_qwen4exp import quantize_group
    from tools.requant_qwen4exp import dequantize
    from tools.requant_qwen4exp import quantize_group as rtn_group
    g = torch.Generator(device="cuda").manual_seed(4)
    ws = {"a.in_proj_qkv": torch.randn(64, 512, device="cuda", generator=g) * 0.02,
          "a.in_proj_z": torch.randn(32, 512, device="cuda", generator=g) * 0.05}
    x = correlated(2048, 512, 5)
    out, e, e_rtn = quantize_group(ws, x.T @ x / x.shape[0], "gptq6")
    ref, _ = rtn_group(ws, "int6")
    assert set(out) == set(ref) and all(out[k].shape == ref[k].shape and out[k].dtype == ref[k].dtype for k in ref)
    assert float(out["a.in_proj_qkv.weight_scale_2"]) == float(out["a.in_proj_z.weight_scale_2"]) == float(ref["a.in_proj_z.weight_scale_2"])
    assert e < e_rtn
    for m, w in ws.items():
        back = dequantize({k: v.cuda() for k, v in out.items()}, m)
        assert back.shape == w.shape and float((back - w).norm() / w.norm()) < 0.1


def test_moments_see_the_prefills_matmuls():
    """The torch function mode adds x^T x for x @ w.t() with a registered w, and nothing for other weights."""
    from tools.gptq_qwen4exp import Moments
    w, other = torch.randn(16, 32, device="cuda"), torch.randn(16, 32, device="cuda")
    H, rows = {}, {}
    x = torch.randn(5, 32, device="cuda")
    with torch.inference_mode(), Moments({w.data_ptr(): "w"}, H, rows):
        y = x @ w.t()
        _ = x @ other.t()
        _ = x @ w.t()
    assert list(H) == ["w"] and rows["w"] == 10 and torch.allclose(H["w"], 2 * x.T @ x, rtol=1e-5, atol=1e-5)
    assert torch.allclose(y, x @ w.t())

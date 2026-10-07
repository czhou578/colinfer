"""Chunked Gated DeltaNet forward (csrc/gdn_prefill.cu) against the reference chunk_gated_delta_rule
(engine/model/qwen35.py, fp32): the output and the continued state, GVA (16 key heads, 48 value heads), and lengths
that end mid-chunk. Run: uv run pytest tests/test_gdn_prefill.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


@pytest.mark.parametrize("T", [1, 5, 63, 64, 65, 200, 1000])
def test_gdn_prefill_matches_reference(T):
    from engine.kernels import ops
    from engine.model.qwen35 import chunk_gated_delta_rule
    torch.manual_seed(T)
    Hk, Hv, D = 16, 48, 128
    q = torch.nn.functional.normalize(torch.randn(T, Hk, D, device="cuda"), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(T, Hk, D, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(T, Hv, D, device="cuda").bfloat16()
    g = -torch.rand(T, Hv, device="cuda") * 0.5 - 0.01
    beta = torch.rand(T, Hv, device="cuda").bfloat16()
    s0 = torch.randn(1, Hv, D, D, device="cuda") * 0.1
    rep = lambda x: x.repeat_interleave(Hv // Hk, dim=1)[None]  # noqa: E731  (GVA: each key head serves 3 value heads)
    o_ref, s_ref = chunk_gated_delta_rule(rep(q), rep(k), v[None], g[None], beta[None], s0.clone())
    st = s0[0].clone()
    o = torch.empty(T, Hv, D, device="cuda", dtype=torch.bfloat16)
    ops().gdn_prefill(q, k, v, g, beta, st, o, D ** -0.5)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    assert rel(o, o_ref[0]) < 1e-2
    assert rel(st, s_ref[0]) < 1e-2

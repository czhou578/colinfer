"""Chunked Gated DeltaNet forward (csrc/gdn_prefill.cu) against FLA's chunk_gated_delta_rule: output and the continued
state, GVA (16 key heads, 48 value heads), lengths that end mid-chunk. Run: uv run pytest tests/test_gdn_prefill.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


@pytest.mark.parametrize("T", [1, 5, 63, 64, 65, 200, 1000])
def test_gdn_prefill_matches_fla(T):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from engine.kernels import ops
    torch.manual_seed(T)
    Hk, Hv, D = 16, 48, 128
    q = torch.nn.functional.normalize(torch.randn(T, Hk, D, device="cuda"), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(T, Hk, D, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(T, Hv, D, device="cuda").bfloat16()
    g = -torch.rand(T, Hv, device="cuda") * 0.5 - 0.01
    beta = torch.rand(T, Hv, device="cuda").bfloat16()
    s0 = torch.randn(1, Hv, D, D, device="cuda") * 0.1
    o_ref, s_ref = chunk_gated_delta_rule(q[None], k[None], v[None], g=g[None], beta=beta[None], initial_state=s0.clone(),
                                          output_final_state=True, use_qk_l2norm_in_kernel=False)
    st = s0[0].clone()
    o = torch.empty(T, Hv, D, device="cuda", dtype=torch.bfloat16)
    ops().gdn_prefill(q, k, v, g, beta, st, o, D ** -0.5)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    assert rel(o, o_ref[0]) < 1e-2
    assert rel(st, s_ref[0]) < 1e-2

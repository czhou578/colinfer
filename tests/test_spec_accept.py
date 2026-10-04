"""Speculative sampling acceptance (engine/spec/accept.py) preserves the target distribution:
over many independent trials the first two emitted tokens follow p_0 and p_1(.|first) exactly."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.runtime.sampler import SamplerParams  # noqa: E402
from engine.spec.accept import draw, inverse_cdf, processed_probs  # noqa: E402


def test_inverse_cdf_marginal_is_exact():
    torch.manual_seed(0)
    V, trials = 50, 200000
    P = torch.softmax(torch.randn(1, V, device="cuda") * 1.5, -1)
    P[0, 7] = 0
    P /= P.sum()
    x = inverse_cdf(P.expand(trials, V), torch.rand(trials, device="cuda").clamp_min(1e-7))
    emp = torch.bincount(x, minlength=V).float() / trials
    assert emp[7] == 0
    assert 0.5 * (emp - P[0]).abs().sum().item() < 0.01


def test_draw_rows_match_single_row_draws():
    """Verifying k+1 rows at once draws, row for row, what k+1 single-row (plain decode) steps draw at those positions:
    speculative sampling then reproduces plain sampling exactly."""
    torch.manual_seed(1)
    V, R = 248320, 4
    logits = torch.randn(2, R, V, device="cuda") * 3
    p = SamplerParams(2, V, "cuda")
    p.set(0, temperature=0.8, top_k=20, top_p=0.95, seed=11)
    p.set(1, temperature=1.0, top_p=0.9, min_p=0.02, seed=12)
    pos = torch.tensor([100, 7], device="cuda", dtype=torch.int32)
    many = draw(logits, p, pos)
    one = torch.stack([draw(logits[:, r:r + 1], p, pos + r)[:, 0] for r in range(R)], 1)
    assert torch.equal(many, one)


def test_processed_probs_filters():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]], device="cuda"))
    one = lambda v, dt=torch.float32: torch.tensor([v], dtype=dt, device="cuda")  # noqa: E731
    p = processed_probs(logits, one(1.0), one(2, torch.int32), one(1.0), one(float("-inf")))
    torch.testing.assert_close(p[0], torch.tensor([0.625, 0.375, 0, 0], device="cuda"), atol=1e-5, rtol=0)
    p = processed_probs(logits, one(1.0), one(4, torch.int32), one(0.7), one(float("-inf")))
    torch.testing.assert_close(p[0], torch.tensor([0.625, 0.375, 0, 0], device="cuda"), atol=1e-5, rtol=0)
    import math
    p = processed_probs(logits, one(1.0), one(4, torch.int32), one(1.0), one(math.log(0.4)))
    torch.testing.assert_close(p[0], torch.tensor([0.625, 0.375, 0, 0], device="cuda"), atol=1e-5, rtol=0)


def test_philox_uniform_per_slot():
    """Seeded uniforms (csrc/sampling.cu): reproducible, in (0, 1], a slot's stream depends only on its own seed / offset."""
    from engine.kernels import ops
    seed = torch.tensor([7, 7, 123], device="cuda")
    off = torch.tensor([0, 0, 5 << 20], device="cuda")
    u = torch.empty(3, 4096, device="cuda")
    ops().philox_uniform(seed, off, u)
    assert torch.equal(u[0], u[1]) and not torch.equal(u[0], u[2])
    assert float(u.min()) > 0.0 and float(u.max()) <= 1.0 and abs(float(u.mean()) - 0.5) < 0.02
    v = torch.empty(1, 4096, device="cuda")
    ops().philox_uniform(seed[2:], off[2:], v)
    assert torch.equal(v[0], u[2])

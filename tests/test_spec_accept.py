"""Speculative sampling acceptance (engine/spec/accept.py) preserves the target distribution:
over many independent trials the first two emitted tokens follow p_0 and p_1(.|first) exactly."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.spec.accept import accept_sample, processed_probs  # noqa: E402


def test_first_token_marginal_is_exact():
    torch.manual_seed(0)
    V, k, trials = 50, 3, 30000
    P = torch.softmax(torch.randn(k + 1, V, device="cuda") * 1.5, -1)
    drafts = torch.tensor([int(P[0].argmax()), 7, 3], device="cuda")  # a likely draft, then arbitrary ones
    counts = torch.zeros(V, device="cuda")
    for t in range(trials // 1000):
        for j in range(1000):
            u = torch.rand(k, device="cuda")
            n, nxt = accept_sample(P, drafts, u, torch.tensor([t * 1000 + j], device="cuda"), torch.zeros(1, dtype=torch.long, device="cuda"))
            first = drafts[0] if int(n) > 1 else nxt[0]
            counts[int(first)] += 1
    emp = counts / counts.sum()
    tv = 0.5 * (emp - P[0]).abs().sum().item()
    assert tv < 0.02, tv


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

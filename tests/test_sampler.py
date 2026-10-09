"""In-graph sampler (engine/runtime/sampler.py): distribution checks against an exact truncated softmax, per-slot
parameters, determinism by (seed, position), and the capture inside a CUDA graph."""
import pytest
import torch

from engine.runtime.sampler import SamplerParams, sample

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

V = 1000


def expected(logits, t, k, p, min_p):
    z = logits / t
    probs = torch.softmax(z, -1)
    keep = probs >= min_p * probs.max()
    order = torch.argsort(z, descending=True)
    keep_k = torch.zeros_like(keep)
    keep_k[order[:k]] = True
    keep &= keep_k
    z2 = z.masked_fill(~keep, float("-inf"))
    pr = torch.softmax(z2, -1)
    srt, idx = torch.sort(pr, descending=True)
    cut = (srt.cumsum(0) - srt) < p
    final = torch.zeros_like(pr)
    final[idx[cut]] = srt[cut]
    return final / final.sum()


@pytest.mark.parametrize("t,k,p,min_p", [(1.0, 0, 1.0, 0.0), (0.7, 20, 1.0, 0.0), (1.0, 0, 0.8, 0.0), (1.3, 50, 0.9, 0.05)])
def test_distribution(t, k, p, min_p):
    torch.manual_seed(0)
    base = torch.randn(V, device="cuda") * 2
    B, rounds = 4, 25000
    params = SamplerParams(B, V, "cuda")
    params.temperature.fill_(t)
    params.top_k.fill_(k if k > 0 else V)
    params.top_p.fill_(p)
    params.log_min_p.fill_(float(torch.tensor(min_p).log()) if min_p > 0 else float("-inf"))
    params.seed.copy_(torch.arange(B, device="cuda") * 7919)
    counts = torch.zeros(V, device="cuda")
    logits = base.expand(B, V).contiguous()
    for r in range(rounds):
        counts += torch.bincount(sample(logits, params, torch.full((B,), r, device="cuda")), minlength=V).float()
    emp = counts / counts.sum()
    want = expected(base, t, k if k > 0 else V, p, min_p)
    assert emp[want == 0].sum() == 0, "sampled a token outside the allowed set"
    tv = 0.5 * (emp - want).abs().sum().item()
    assert tv < 0.03, tv


def test_slot_independence():
    """A slot's draws depend only on its own seed, not on its neighbours' (FlashInfer row-0 issue)."""
    torch.manual_seed(3)
    logits = torch.randn(2, V, device="cuda")
    outs = []
    for other_seed in (0, 12345):
        p = SamplerParams(2, V, "cuda")
        p.set(0, temperature=1.0, seed=other_seed)
        p.set(1, temperature=1.0, seed=9)
        outs.append([int(sample(logits, p, torch.full((2,), i, device="cuda"))[1]) for i in range(10)])
    assert outs[0] == outs[1]


def test_greedy_slots_and_seed_determinism():
    torch.manual_seed(1)
    logits = torch.randn(3, V, device="cuda")
    p = SamplerParams(3, V, "cuda")
    p.set(0, temperature=0.0)
    p.set(1, temperature=1.0, seed=7)
    p.set(2, temperature=1.0, seed=7)
    pos = lambda i: torch.full((3,), i, device="cuda")  # noqa: E731
    out = sample(logits, p, pos(0))
    assert out[0] == logits[0].argmax()
    a = [int(sample(logits, p, pos(i))[1]) for i in range(20)]
    b = [int(sample(logits, p, pos(i))[1]) for i in range(20)]
    assert a == b and len(set(a)) > 1
    assert [int(sample(logits[[1, 1, 1]], p, pos(i))[2]) for i in range(20)] == a  # a slot's draws depend on (seed, position)


def test_captured_in_cuda_graph_follows_position():
    torch.manual_seed(2)
    logits = torch.zeros(2, V, device="cuda")  # uniform: draws should differ step to step
    p = SamplerParams(2, V, "cuda")
    p.set(0, temperature=1.0, seed=3)
    p.set(1, temperature=1.0, seed=4)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    pos = torch.zeros(2, dtype=torch.int32, device="cuda")
    with torch.cuda.stream(s):
        sample(logits, p, pos)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = sample(logits, p, pos)
        pos += 1
    draws = []
    for _ in range(30):
        g.replay()
        draws.append(tuple(out.tolist()))
    assert len(set(draws)) > 20


def test_top_k_beyond_int32_means_no_top_k():
    p = SamplerParams(1, V, "cuda")
    p.set(0, temperature=1.0, top_k=3_000_000_000)  # an int32 overflow on the engine thread used to end the server
    assert int(p.top_k[0]) == V

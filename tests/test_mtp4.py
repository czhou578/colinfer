"""The MTP drafter of qwen4_exp and its speculative cycle (engine/spec/mtp4.py) on the small random model of
tests/test_fast_qwen4exp.py (hidden 2560, 4 layers, 16 NVFP4 experts) and a random drafter of the same dims with FP8
block-scaled experts. The tests check:
- the requantization of the FP8 experts to NVFP4 (KernelMoE's stacked layout) against dequant_fp8_block
- the drafter's forward against a plain-PyTorch reference of the wiring of vLLM's qwen4_exp/nvidia/mtp.py, in bf16
  (the reference modules of engine/model/qwen4exp.py for the layer and the mixer), over 40 rows
- row invariance: 8 rows in one launch equal 8 single rows, bit for bit (outputs and the drafter's state)
- Mtp4.prefill (the prefill path for a long prompt) against the decode-path rows
- exactness: greedy tokens with speculation (k = 1, 2, 3) equal plain decode over 64 tokens, with the drafter's own
  drafts and with partly correct host drafts (so that every accepted length occurs)
- the sampled cycle (position-keyed uniforms) equals plain sampling
- a graph replay of the cycle equals the eager cycle, bit for bit
Run: uv run pytest tests/test_mtp4.py -q"""
import itertools

import pytest
import torch
import torch.nn.functional as F

from tests.test_fast_qwen4exp import EOS, V, build_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

L = 256
TOL = 0.03     # max |logit difference| / std(reference logits) per row. Measured 0.0125 (x4: 0.0027 relative)
PREFILL_TOL = 0.06  # the prefill path quantizes the expert activations to NVFP4 and runs its projections on cuBLAS. Measured 0.023
bf = torch.bfloat16


# ------------------------------------------------------------------------------------------------------------ fixtures
def random_fp8_experts(E, I, H, g):
    """FP8 experts with 128 x 128 block scales, as the checkpoint's mtp.layers.0.mlp.experts.*: weights near 1 / sqrt(fan in)."""
    def fp8(*shape, scale):
        w = torch.randn(*shape, device="cuda", generator=g).to(torch.float8_e4m3fn)
        nb = (shape[0], -(-shape[1] // 128), -(-shape[2] // 128))
        si = ((torch.rand(*nb, device="cuda", generator=g) * 0.5 + 0.75) * scale).to(bf)
        return w, si
    g8, gs = fp8(E, I, H, scale=H ** -0.5)
    u8, us = fp8(E, I, H, scale=H ** -0.5)
    d8, ds = fp8(E, H, I, scale=I ** -0.5)
    return g8, gs, u8, us, d8, ds


def build_drafter_reference(cfg, seed: int = 5):
    """(tensors, experts, the reference layer, the reference mixer): a random drafter in the checkpoint's names."""
    from engine.model.qwen4exp import DecoderLayer4, GatedResidual
    from engine.spec.mtp4 import drafter_cfg, requant_experts
    H = cfg.hidden_size
    torch.manual_seed(seed)
    with torch.device("cuda"):
        layer = DecoderLayer4(drafter_cfg(cfg), 0, experts="nvfp4")
        mixer = GatedResidual(cfg, combine=False)
    layer, mixer = layer.to(bf), mixer.to(bf)  # before the experts: .to(dtype) would convert the e4m3 scales
    g = torch.Generator(device="cuda").manual_seed(seed)
    new = lambda *shape: torch.empty(*shape, device="cuda", dtype=bf)  # noqa: E731
    t = {"fc_embedding.weight": new(H, H), "fc_hidden.weight": new(H, H), "pre_fc_norm_embedding.weight": new(H),
         "pre_fc_norm_hidden.weight": new(4 * H)}
    params = dict(itertools.chain(layer.named_parameters(prefix="layers.0"), mixer.named_parameters(prefix="hyper_connection_mixer")))
    with torch.no_grad():
        for n, p in itertools.chain(params.items(), t.items()):
            if p.dim() == 1:
                p.normal_(0, 0.1, generator=g)  # zero-centered norms: (1 + w)
            else:
                p.normal_(0, p.shape[1] ** -0.5, generator=g)
        ex = requant_experts(*random_fp8_experts(cfg.num_experts, cfg.moe_intermediate_size, H, g))
        for k in ("w_gu", "sf_gu", "w_dn", "sf_dn", "gs"):
            setattr(layer.mlp.experts, k, ex[k])
    t |= {n: p.detach().clone() for n, p in params.items()}
    return t, ex, layer.eval().requires_grad_(False), mixer.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def setup():
    from engine.model.fast_qwen4exp import to_fast4
    from engine.spec.mtp4 import Mtp4
    fast = to_fast4(build_reference())
    t, ex, rlayer, rmixer = build_drafter_reference(fast.cfg)
    mtp = Mtp4(fast, None, tensors=dict(t), experts=ex, log=lambda *a: None)
    return fast, mtp, rlayer, rmixer


@pytest.fixture(scope="module")
def tokens():
    g = torch.Generator().manual_seed(7)
    t = torch.randint(4, V - 1, (64,), generator=g)
    t[[9, 10]] = EOS
    return t.tolist()


def _ctx(hist):
    return ([EOS, EOS] + list(hist))[-2:]


@torch.inference_mode()
def target_hidden(fast, st, toks):
    """Feeds toks from st.pos_t through verify + commit in chunks of 16: (the logits [N, V], hidden4 [N, 10240]) of every row."""
    lg, hs, done = [], [], []
    for c0 in range(0, len(toks), 16):
        ch = toks[c0:c0 + 16]
        rows = fast.host_rows([ch], [_ctx(done)]).cuda()
        logits, h4 = fast.verify(torch.tensor([ch], device="cuda"), st, rows)
        fast.commit(st, torch.tensor([len(ch)], dtype=torch.int32, device="cuda"))
        lg.append(logits[0])
        hs.append(h4[0])
        done += ch
    return torch.cat(lg), torch.cat(hs)


def ref_rows(cfg, rlayer, rmixer, t, rst, tok, hid):
    """The reference drafter (vLLM's qwen4_exp MTP wiring in plain PyTorch, bf16) on T rows of one slot at rst.pos:
    tok [T] (x_{i+1}), hid [T, 10240] -> (g [T, 2560], x4 [T, 10240], logits fp32 [T, V])."""
    from engine.model.qwen35 import rotary
    from engine.spec.mtp4 import drafter_cfg
    T, H, S = tok.shape[0], cfg.hidden_size, cfg.hc_count

    def gemma(x, w):  # GemmaRMSNorm over the whole last dim
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1.0 + w.float())).to(bf)
    e = F.linear(gemma(t["embed"][tok], t["pre_fc_norm_embedding.weight"]), t["fc_embedding.weight"])        # [T, H]
    h = F.linear(gemma(hid, t["pre_fc_norm_hidden.weight"]).view(T, S, H), t["fc_hidden.weight"])         # one norm over 4H, shared fc
    x4 = (h + e[:, None]).reshape(1, T, S * H)                                                            # + e to every stream
    p = torch.arange(rst.pos, rst.pos + T, device="cuda")
    rst.rope_pos[:, rst.pos:rst.pos + T] = p[:, None].expand(T, 3)
    cos, sin = rotary(drafter_cfg(cfg), p, bf)
    x4 = rlayer(x4, cos, sin, rst, None)
    g = rmixer(x4)
    rst.pos += T
    return g[0], x4[0], F.linear(g[0], t["lm_head"]).float()


# ------------------------------------------------------------------------------------------------------------ experts
@torch.inference_mode()
def test_requant_experts():
    from engine.spec.mtp4 import requant_experts
    from engine.weights.loader import dequant_fp8_block, dequant_nvfp4
    E, I, H = 6, 640, 2560
    g = torch.Generator(device="cuda").manual_seed(3)
    g8, gs, u8, us, d8, ds = random_fp8_experts(E, I, H, g)
    ex = requant_experts(g8, gs, u8, us, d8, ds, batch=4)  # batches of 4: a partial last batch
    assert ex["w_gu"].shape == (E, 2 * I, H // 2) and ex["sf_gu"].shape == (E, 2 * I, H // 16) and ex["sf_gu"].dtype == torch.float8_e4m3fn
    assert ex["w_dn"].shape == (E, H, I // 2) and ex["sf_dn"].shape == (E, H, I // 16) and ex["gs"].shape == (E, 3)
    assert torch.equal(ex["gs"][:, 0], ex["gs"][:, 1])  # gate and up share their global scale (the grouped GEMM's alpha)
    worst = 0.0
    for e in range(E):
        f32 = torch.float32
        for got, want in ((dequant_nvfp4(ex["w_gu"][e, :I], ex["sf_gu"][e, :I], ex["gs"][e, 0], f32), dequant_fp8_block(g8[e], gs[e], out_dtype=f32)),
                          (dequant_nvfp4(ex["w_gu"][e, I:], ex["sf_gu"][e, I:], ex["gs"][e, 1], f32), dequant_fp8_block(u8[e], us[e], out_dtype=f32)),
                          (dequant_nvfp4(ex["w_dn"][e], ex["sf_dn"][e], ex["gs"][e, 2], f32), dequant_fp8_block(d8[e], ds[e], out_dtype=f32))):
            worst = max(worst, ((got - want).norm() / want.norm()).item())
    print(f"\n[mtp4] NVFP4 requantization: worst relative error {worst:.4f} (the e2m1 grid's error on Gaussian weights)")
    assert worst < 0.12, worst


# ------------------------------------------------------------------------------------------------------------ forward
@torch.inference_mode()
def test_forward_matches_reference(setup, tokens):
    from engine.model.qwen4exp import Qwen4ExpState
    from engine.spec.mtp4 import Mtp4State, drafter_cfg
    fast, mtp, rlayer, rmixer = setup
    cfg, N = fast.cfg, 40
    _, hid = target_hidden(fast, fast.new_state(1, L), tokens[:N])
    t = {"embed": fast.embed_tokens.weight, "lm_head": fast.lm_head.w, "pre_fc_norm_embedding.weight": mtp.pre_e.weight,
         "pre_fc_norm_hidden.weight": mtp.pre_h.weight, "fc_embedding.weight": mtp.fc_e.w, "fc_hidden.weight": mtp.fc_h.w}
    rst = Qwen4ExpState(drafter_cfg(cfg), 1, L, "cuda", bf, kv_dtype=torch.float8_e4m3fn)  # the engine's fp8 KV
    mst = Mtp4State(cfg, L, "cuda")
    nxt = torch.tensor(tokens[1:N + 1], device="cuda")  # row i takes x_{i+1} and the target's hidden4 of position i
    worst, worst_x, flips, checked = 0.0, 0.0, 0, 0
    for i in range(N):
        g_r, x_r, want = ref_rows(cfg, rlayer, rmixer, t, rst, nxt[i:i + 1], hid[i:i + 1])
        g, x4 = mtp(nxt[i:i + 1].view(1, 1), hid[i:i + 1].view(1, 1, -1), mst)
        mst.pos_t += 1
        got = mtp.head(g[0])
        worst = max(worst, ((got - want).abs().max() / want.std()).item())
        worst_x = max(worst_x, ((x4[0].float() - x_r.float()).norm() / x_r.float().norm()).item())
        top2 = want[0].topk(2).values
        if (top2[0] - top2[1]).item() > 0.1 * want.std().item():
            checked += 1
            flips += int(got.argmax() != want.argmax())
    print(f"\n[mtp4] drafter vs reference over {N} rows: worst max|diff| / std {worst:.4f}, x4 relative error {worst_x:.4f}; "
          f"argmax checked at {checked} rows, {flips} differ")
    assert worst < TOL and worst_x < 0.01, (worst, worst_x)
    assert flips == 0 and checked >= 25


@torch.inference_mode()
def test_row_invariance(setup, tokens):
    from engine.spec.mtp4 import Mtp4State
    fast, mtp, _, _ = setup
    P, T = 13, 8  # rows 13..20 complete the blocks 3 (row 15) and 4 (row 19)
    _, hid = target_hidden(fast, fast.new_state(1, L), tokens[:P + T])
    nxt = torch.tensor(tokens[1:P + T + 1], device="cuda")
    a, b = Mtp4State(fast.cfg, L, "cuda"), Mtp4State(fast.cfg, L, "cuda")
    for st in (a, b):
        for i in range(P):
            mtp(nxt[i:i + 1].view(1, 1), hid[i:i + 1].view(1, 1, -1), st)
            st.pos_t += 1
    g8, x8 = mtp(nxt[P:P + T].view(1, T), hid[P:P + T].view(1, T, -1), a)
    for i in range(T):
        g1, x1 = mtp(nxt[P + i:P + i + 1].view(1, 1), hid[P + i:P + i + 1].view(1, 1, -1), b)
        b.pos_t += 1
        assert torch.equal(g8[0, i], g1[0, 0]) and torch.equal(x8[0, i], x1[0, 0]), i
        assert torch.equal(mtp.draft(g8[:, i])[0], mtp.draft(g1[:, 0])[0])
    for x, y, r in zip(a.kv_tensors(), b.kv_tensors(), a.kv_ratios()):
        n = (P + T) // r
        assert torch.equal(x[:, :, :n], y[:, :, :n]), x.shape
    assert a.snap_tensors() == [] and len(a.kv_tensors()) == 5


@torch.inference_mode()
def test_prefill_path(setup, tokens):
    from engine.spec.mtp4 import Mtp4State
    fast, mtp, _, _ = setup
    N = 40  # > 16 rows: the prefill path (cuBLAS projections, the sparse prefill attention, the grouped NVFP4 MoE)
    _, hid = target_hidden(fast, fast.new_state(1, L), tokens[:N])
    nxt = torch.tensor(tokens[1:N + 1], device="cuda")
    a = Mtp4State(fast.cfg, L, "cuda")
    g_p, x_p = mtp.prefill(nxt, hid, a.slot(0))
    assert int(a.pos_t[0]) == N
    b = Mtp4State(fast.cfg, L, "cuda")
    for i in range(N):
        g, x = mtp(nxt[i:i + 1].view(1, 1), hid[i:i + 1].view(1, 1, -1), b)
        b.pos_t += 1
    lp, lg = mtp.head(g_p[0]), mtp.head(g[0, 0])
    err = ((lp - lg).abs().max() / lg.std()).item()
    xe = ((x_p[0].float() - x[0, 0].float()).norm() / x[0, 0].float().norm()).item()
    ke = ((a.ik[0][0, :N].float() - b.ik[0][0, :N].float()).norm() / b.ik[0][0, :N].float().norm()).item()
    print(f"\n[mtp4] prefill path vs decode rows over {N} rows: last row max|diff| / std {err:.4f}, x4 {xe:.4f}, raw indexer keys {ke:.4f}")
    assert err < PREFILL_TOL and xe < 0.02 and ke < 0.015, (err, xe, ke)
    assert torch.equal(a.rope_pos[0, :N], b.rope_pos[0, :N])


# ------------------------------------------------------------------------------------------------------------ cycle
@torch.inference_mode()
def plain_decode(fast, prompt, n, params=None):
    """n tokens of plain decode (T = 1 steps) after the prompt: greedy, or the position-keyed samples of params."""
    from engine.spec.accept import draw
    st = fast.new_state(1, L)
    hist, out = [], []
    for t in prompt:
        logits = fast(torch.tensor([[t]], device="cuda"), st, fast.host_rows([[t]], [_ctx(hist)]).cuda())
        hist.append(t)
    while len(out) < n:
        y = int(logits[0].argmax()) if params is None else int(draw(logits[:, None], params, st.pos_t)[0, 0])
        out.append(y)
        logits = fast(torch.tensor([[y]], device="cuda"), st, fast.host_rows([[y]], [_ctx(hist)]).cuda())
        hist.append(y)
    return out


class Spec:
    """The host side of speculation for one slot (what the scheduler does): the prompt through verify + commit, the
    drafter's prompt rows and first drafts, then cycles whose n-gram rows the host writes before each replay."""

    def __init__(self, fast, mtp, k, params=None):
        from engine.spec.mtp4 import Mtp4Cycle, Mtp4State
        self.fast, self.mtp, self.k, self.params = fast, mtp, k, params
        self.st = fast.new_state(1, L)
        self.mst = Mtp4State(fast.cfg, L, "cuda", active=self.st.active, rope_delta=self.st.rope_delta)
        self.cyc = Mtp4Cycle(fast, mtp, self.st, self.mst, k, params=params)

    @torch.inference_mode()
    def start(self, prompt):
        from engine.spec.accept import draw
        self.st.reset()
        self.mst.reset()
        logits, hid = target_hidden(self.fast, self.st, prompt)
        last = logits[-1:]
        y = int(last.argmax()) if self.params is None else int(draw(last[:, None], self.params, self.st.pos_t)[0, 0])
        d = self.mtp.first_drafts(torch.tensor(prompt[1:] + [y], device="cuda"), hid, self.mst.slot(0), self.k)
        self.cyc.tok.copy_(torch.tensor([[y] + d], device="cuda"))
        self.prompt, self.out = list(prompt), [y]

    def cycle(self, drafts=None):
        """One replay; drafts (optional): host drafts that replace the drafter's. Returns the accepted count n."""
        if drafts is not None:
            self.cyc.tok[0, 1:] = torch.tensor(drafts, device="cuda")
        tok = self.cyc.tok[0].tolist()
        self.cyc.feed.copy_(self.fast.host_rows([tok], [_ctx(self.prompt + self.out[:-1])]))
        self.cyc.graph.replay()
        n = int(self.cyc.n[0])
        self.out += self.cyc.out_tok[0, :n].tolist()
        return n


@pytest.mark.parametrize("k", [1, 2, 3])
@torch.inference_mode()
def test_spec_equals_plain_greedy(setup, tokens, k):
    fast, mtp, _, _ = setup
    prompt, N = tokens[:20], 64
    want = plain_decode(fast, prompt, N + 8)
    sp = Spec(fast, mtp, k)
    sp.start(prompt)
    ns = []
    while len(sp.out) < N:
        ns.append(sp.cycle())
    assert sp.out[:N] == want[:N], next(i for i, (a, b) in enumerate(zip(sp.out, want)) if a != b)
    # host drafts: the right continuation, cut at a cycling point (j correct drafts, then a wrong one): every n in 1..k+1
    g = torch.Generator().manual_seed(k)
    sp.start(prompt)
    seen = set()
    while len(sp.out) < N:
        i, j = len(sp.out), int(torch.randint(0, k + 1, (1,), generator=g))
        d = (want + [5] * 16)[i:i + k]
        if j < k:
            d[j] = (d[j] + 1) % V
        n = sp.cycle(d)
        assert n == j + 1
        seen.add(n)
    assert sp.out[:N] == want[:N]
    assert seen == set(range(1, k + 2))
    print(f"\n[mtp4] k = {k}: 64 greedy tokens equal plain decode; the random drafter's own drafts: {sum(ns) / len(ns):.2f} tokens "
          f"per cycle over {len(ns)} cycles; with host drafts, accepted lengths {sorted(seen)}")


@torch.inference_mode()
def test_spec_equals_plain_sampled(setup, tokens):
    from engine.runtime.sampler import SamplerParams
    fast, mtp, _, _ = setup
    params = SamplerParams(1, V, "cuda")
    params.set(0, temperature=0.8, top_k=50, top_p=0.95, seed=1234)
    prompt, N, k = tokens[:20], 48, 2
    want = plain_decode(fast, prompt, N + 8, params)
    sp = Spec(fast, mtp, k, params=params)
    sp.start(prompt)
    while len(sp.out) < N:
        i = len(sp.out)
        sp.cycle((want + [5] * 16)[i:i + k] if i % 3 == 0 else None)  # some cycles with the right drafts: accepted samples
    assert sp.out[:N] == want[:N]


@torch.inference_mode()
def test_graph_equals_eager(setup, tokens):
    fast, mtp, _, _ = setup
    prompt, k = tokens[:20], 2
    a, b = Spec(fast, mtp, k), Spec(fast, mtp, k)
    a.start(prompt)
    b.start(prompt)
    want = plain_decode(fast, prompt, 32)
    for c in range(6):
        i = len(a.out)
        d = (want + [5] * 16)[i:i + k] if c % 2 else None
        if d is not None:
            for s in (a, b):
                s.cyc.tok[0, 1:] = torch.tensor(d, device="cuda")
        tok = a.cyc.tok[0].tolist()
        assert tok == b.cyc.tok[0].tolist()
        rows = fast.host_rows([tok], [_ctx(a.prompt + a.out[:-1])])
        a.cyc.feed.copy_(rows)
        a.cyc.graph.replay()
        got = [t.clone() for t in (a.cyc.out_tok, a.cyc.n, a.cyc.logits, a.cyc.H)]
        b.cyc.feed.copy_(rows)
        eager = b.cyc._body()
        for x, y in zip(got, eager):
            assert torch.equal(x, y)
        assert torch.equal(a.cyc.tok, b.cyc.tok)
        n = int(got[1][0])
        a.out += got[0][0, :n].tolist()
        b.out += got[0][0, :n].tolist()
    for x, y in zip(a.mst.kv_tensors() + a.st.kv_tensors() + a.st.snap_tensors(), b.mst.kv_tensors() + b.st.kv_tensors() + b.st.snap_tensors()):
        assert torch.equal(x, y)


def test_drafter_contract(setup):
    from engine.spec.mtp4 import Mtp4State
    fast, mtp, _, _ = setup
    st = Mtp4State(fast.cfg, 64, "meta", batch=2)
    assert [tuple(t.shape) for t in st.kv_tensors()] == [(2, 2, 64, 256), (2, 2, 64, 256), (2, 1, 64, 128), (2, 1, 16, 128), (2, 1, 64, 3)]
    assert st.snap_tensors() == [] and st.kv_ratios() == [1, 1, 1, 4, 1]
    assert mtp.vocab_ids.numel() == V and mtp.head.w.shape == (V, 2560)  # a small vocabulary is scored whole
    assert "target" not in dict(mtp.named_children()) and "lm_head" not in dict(mtp.named_children())

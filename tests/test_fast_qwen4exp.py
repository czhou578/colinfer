"""The decode model of qwen4_exp on the kernels (engine/model/fast_qwen4exp.py) on a small random model that fits the
kernels' shapes (hidden 2560, 4 layers: GDN, GDN + PLE, GDN, QSA; 16 NVFP4 experts top-10 of 128; an in-memory FP8
n-gram table over primes near 1009). The tests check:
- the decode path (T = 1) against the reference Qwen4ExpForCausalLM (bf16, fp8 KV) on the same weights over 40 tokens
- at 4096 tokens (sparse selection: each row attends to 512 of its 1023 blocks plus the tail), the decode path through a
  CUDA graph against the reference's one-shot forward, and the block keys against the reference's
- a verify of T = 8 rows equals 8 single steps bit for bit, and commit(n) leaves the state of n single steps
- a captured CUDA graph of forward equals eager, bit for bit
- the state hooks (kv_tensors, snap_tensors), view(lo, hi) sharing and reset
- the refusal of a config that does not fit the kernels
Run: uv run pytest tests/test_fast_qwen4exp.py -q"""
import copy
import dataclasses

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

EOS, V, L = 3, 1024, 64
TOL = 0.04  # max |logit difference| / std(reference logits) per step. Measured 0.017 to 0.024; the bf16 reference itself sits
            # 0.016 to 0.028 from an fp32 copy of it, and the kernels sit as close to that copy as the bf16 reference does


def small_cfg(**kw):
    from engine.model.qwen4exp import Qwen4ExpConfig
    return Qwen4ExpConfig(**({
        "hidden_size": 2560, "num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "vocab_size": V, "max_position_embeddings": 8192,
        "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128, "num_experts": 16, "num_experts_per_tok": 10,
        "hc_count": 4, "hc_lowrank": 320, "ple_layer_ids": [2], "ple_embed_dim": 2560, "ngram_vocab_size_base": 1009,
        "split_ngram_parts": 4, "eos_token_id": EOS, "image_token_id": V - 1, "indexer_budget": 2048} | kw))


class MemTable:
    """An in-memory stand-in for engine.weights.ple.NgramTable: random FP8 rows and one scale."""

    def __init__(self, n_rows: int, dim: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.data = (torch.randn(n_rows, dim, generator=g) * 2).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        self.n_rows, self.dim = n_rows, dim
        self.scale = float(torch.tensor(0.25, dtype=torch.bfloat16).float())

    def rows(self, ids):
        return self.data[np.asarray(ids, np.int64).reshape(-1)]

    def dequant(self, raw, dtype=torch.bfloat16):
        return (raw.view(torch.float8_e4m3fn).float() * self.scale).to(dtype)


def build_reference(seed: int = 0):
    """The reference model in bf16 on the GPU with random weights and NVFP4 experts."""
    from engine.model.moe_kernels import random_moe_weights
    from engine.model.qwen4exp import Qwen4ExpForCausalLM
    from engine.weights.ple import ngram_params
    cfg = small_cfg()
    table = MemTable(ngram_params(cfg, 0).padded_rows, cfg.ngram_dim, seed)
    torch.manual_seed(seed)
    with torch.device("cuda"):
        m = Qwen4ExpForCausalLM(cfg, experts="nvfp4", ngram_table=table)
    m = m.to(torch.bfloat16)  # before the experts: .to(dtype) would convert the e4m3 scales
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"):
                p.uniform_(-1, 1, generator=g)
            elif n.endswith("dt_bias"):
                p.normal_(1, 0.2, generator=g)
            elif "conv1d" in n:
                p.normal_(0, 0.3, generator=g)
            elif n.endswith("linear_attn.norm.weight"):
                p.normal_(1, 0.1, generator=g)  # the GDN output norm multiplies by w, not (1 + w)
            elif p.dim() == 1:
                p.normal_(0, 0.1, generator=g)  # zero-centered norms: (1 + w)
            elif n == "embed_tokens.weight":
                p.normal_(0, 1, generator=g)
            elif n == "lm_head.weight":
                p.normal_(0, 4 / p.shape[1] ** 0.5, generator=g)
            else:
                p.normal_(0, p.shape[1] ** -0.5, generator=g)
        for layer in m.layers:
            w = random_moe_weights(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size, generator=g)
            ex, moe = layer.mlp.experts, layer.mlp
            for k in ("w_gu", "sf_gu", "w_dn", "sf_dn", "gs", "in_scale"):
                setattr(ex, k, w[k])
            moe.gate.weight.copy_(w["router_w"])
            moe.shared_expert_gate.weight.copy_(w["shared_gate_w"])
            moe.shared_expert.gate_proj.weight.copy_(w["sh_gate"])
            moe.shared_expert.up_proj.weight.copy_(w["sh_up"])
            moe.shared_expert.down_proj.weight.copy_(w["sh_down"])
    return m.eval().requires_grad_(False)


@pytest.fixture(scope="module")
def models():
    from engine.model.fast_qwen4exp import to_fast4
    ref = build_reference()
    fast = to_fast4(copy.deepcopy(ref))
    return ref, fast


@pytest.fixture(scope="module")
def tokens():
    g = torch.Generator().manual_seed(1)
    t = torch.randint(4, V - 1, (48,), generator=g)
    t[[10, 11, 30]] = EOS  # eos twice in a row and once alone: the n-gram segment rule
    return t.tolist()


class Feeder:
    """The host side of the n-gram rows: the context of the next token, and the rows of the given tokens on the GPU."""

    def __init__(self, fast):
        self.fast, self.ctx = fast, [EOS, EOS]

    def __call__(self, toks: list[int]) -> torch.Tensor:
        rows = self.fast.host_rows([toks], [self.ctx]).cuda()
        self.ctx = ([*self.ctx, *toks])[-2:]
        return rows


def step(fast, st, feed, tok):
    return fast(torch.tensor([[tok]], device="cuda"), st, feed([tok]))[0]


# ------------------------------------------------------------------------------------------------------------ reference
def _decode_vs_reference(ref, fast, tokens) -> tuple[float, int, int]:
    """(the worst max |logit difference| / std, the argmax flips, the steps checked) of 40 decode steps."""
    rs = ref.new_state(1, L, kv_dtype=torch.float8_e4m3fn)  # the engine's fp8 KV, so that the KV rounds the same
    fs, feed = fast.new_state(1, L), Feeder(fast)
    worst, flips, checked = 0.0, 0, 0
    for t, tok in enumerate(tokens[:40]):
        want = ref(torch.tensor([[tok]], device="cuda"), rs)[0, -1]
        got = step(fast, fs, feed, tok)
        err = ((got - want).abs().max() / want.std()).item()
        worst = max(worst, err)
        top2 = want.topk(2).values
        if (top2[0] - top2[1]).item() > 0.1 * want.std().item():  # not a near-tie: the argmax must agree
            checked += 1
            flips += int(got.argmax() != want.argmax())
    return worst, flips, checked


@torch.inference_mode()
def test_decode_matches_reference(models, tokens):
    worst, flips, checked = _decode_vs_reference(*models, tokens)
    print(f"\n[fast4] worst max|diff| / std over 40 steps: {worst:.4f}; argmax checked at {checked} steps, {flips} differ")
    assert worst < TOL, worst
    assert flips == 0 and checked >= 25


@torch.inference_mode()
def test_yarn_decode_matches_reference(tokens):
    """With YaRN (qwen4exp.apply_yarn4 before to_fast4), the kernels rotate by the YaRN frequencies and read the folded
    norms: the decode path follows the YaRN reference as closely as the plain one, and the state takes the longer slot."""
    from engine.model.fast_qwen4exp import to_fast4
    from engine.model.qwen4exp import apply_yarn4
    from engine.model.qwen35 import rope_inv_freq
    ref = build_reference()
    apply_yarn4(ref, 2.0)
    fast = to_fast4(copy.deepcopy(ref))
    assert torch.equal(fast.layers[3].self_attn.inv_freq, rope_inv_freq(ref.cfg, "cuda"))
    assert not torch.equal(fast.layers[3].self_attn.inv_freq, rope_inv_freq(small_cfg(), "cuda"))
    worst, flips, checked = _decode_vs_reference(ref, fast, tokens)
    assert worst < TOL and flips == 0 and checked >= 25, (worst, flips, checked)
    assert fast.new_state(1, 2 * 8192).k[3].shape[2] == 2 * 8192  # past the native 8192 positions
    del ref, fast
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_long_context_selection(models, monkeypatch):
    """4096 tokens: the rows past 2051 select 512 of up to 1023 blocks, so a wrong block key or selection shows in the
    logits. The decode steps replay a graph (about 1 ms each)."""
    from engine.model.fast import capture
    ref, fast = models
    N, LL, K = 4096, 4096 + 16, 32
    g = torch.Generator().manual_seed(2)
    toks = torch.randint(4, V - 1, (N,), generator=g)
    want = ref(toks.view(1, -1).cuda(), ref.new_state(1, LL, kv_dtype=torch.float8_e4m3fn))[0, -K:]  # one shot
    rows = fast.host_rows([toks.tolist()], [[EOS, EOS]]).cuda()
    st = fast.new_state(1, LL)
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    buf = fast.feed_buffer(1, 1)
    graph, out = capture(lambda: fast(tok, st), None)
    st.reset()
    got = []
    for t in range(N):
        tok.copy_(toks[t:t + 1].view(1, 1))
        buf.copy_(rows[:, t:t + 1])
        graph.replay()
        if t >= N - K:
            got.append(out[0].clone())
    got = torch.stack(got)
    err = ((got - want).abs().amax(-1) / want.std(-1)).max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    print(f"\n[fast4] 4096 tokens: worst max|diff| / std over the last {K} rows {err:.4f}; argmax agrees on {agree:.0%}")
    assert err < TOL, err
    # the selection of one more row (position N) against the reference indexer on the slot's own raw keys and positions.
    # The logits alone would not show a wrong selection: dense attention moves them by only 0.033 of their std here.
    from engine.model import qsa_kernels as qk
    cap = {}

    def wrap(name, fn):
        def f(*a, **kw):
            out = fn(*a, **kw)
            cap[name] = out
            return out
        monkeypatch.setattr(qk, name, f)
    wrap("qsa_prologue", qk.qsa_prologue)
    wrap("qsa_topk", qk.qsa_topk)
    tok.fill_(5)
    buf.copy_(fast.host_rows([[5]], [toks[-2:].tolist()]).cuda())
    fast(tok, st)
    sel, nsel = cap["qsa_topk"]
    pos = torch.tensor([N], device="cuda")
    assert int(nsel[0, 0]) == 512 and (N + 1) // 4 == 1024
    attn = fast.layers[3].self_attn
    got = qk.selection_mask(sel[0], nsel[0], pos, N + 1)
    want, scores = qk.qsa_select_reference(cap["qsa_prologue"][0], st.ik[3][0, :N + 1], pos, attn.indexer.k_layernorm.weight, attn.inv_freq,
                                           attn.indexer.k_layernorm.eps, rope_pos=st.rope_pos[0, :N + 1].long(), mrope=attn.mrope)
    diff = ((got[0] ^ want[0]).nonzero().flatten() // 4).unique()
    thr = scores[0].sort(descending=True).values[511]
    assert ((scores[0][diff] - thr).abs() <= 1e-4 * max(1.0, thr.item())).all(), diff  # only near-ties may differ
    # the block keys of the 1024 complete blocks against the reference's (fp32 RoPE here, bf16 cos / sin there)
    rs = ref.new_state(1, LL)
    rs.ik[3][:, 0, :N] = st.ik[3][:, :N]
    rs.rope_pos[:, :N] = st.rope_pos[:, :N].long()
    kb = ref.layers[3].self_attn.indexer.block_keys(rs, 3, N, torch.bfloat16)[0]
    assert torch.equal(st.ik[3][0, :N], rs.ik[3][0, 0, :N])
    torch.testing.assert_close(st.ikb[3][0, :N // 4].float(), kb.float(), rtol=2 ** -6, atol=2 ** -6)


# ------------------------------------------------------------------------------------------------------------ verify
def _state_equal(a, b, upto: int):
    """The recurrent state equal, and every per-position row below `upto`."""
    for x, y in zip(a.snap_tensors(), b.snap_tensors()):
        assert torch.equal(x, y)
    for x, y, r in zip(a.kv_tensors(), b.kv_tensors(), a.kv_ratios()):
        n = upto // r
        assert torch.equal(x[:, :, :n], y[:, :, :n]), x.shape
    assert torch.equal(a.pos_t, b.pos_t)


@torch.inference_mode()
def test_verify_equals_steps_and_commit(models, tokens):
    _, fast = models
    P, T, N = 13, 8, 5  # rows 13..20 complete the blocks 3 (row 15) and 4 (row 19)
    sts = [fast.new_state(1, L) for _ in range(3)]
    feeds = [Feeder(fast) for _ in range(3)]
    for st, fd in zip(sts, feeds):
        for tok in tokens[:P]:
            step(fast, st, fd, tok)
    a, b, c = sts
    draft = tokens[P:P + T]
    logits, hidden4 = fast.verify(torch.tensor([draft], device="cuda"), a, feeds[0](draft))
    assert logits.shape == (1, T, V) and hidden4.shape == (1, T, 4 * 2560)
    assert int(a.pos_t) == P  # the positions wait for commit
    for t, tok in enumerate(draft):
        assert torch.equal(logits[0, t], step(fast, b, feeds[1], tok)), t
    fast.commit(a, torch.tensor([N], dtype=torch.int32, device="cuda"))
    for tok in draft[:N]:
        step(fast, c, feeds[2], tok)
    _state_equal(a, c, P + N)
    feeds[0].ctx = feeds[2].ctx  # the host keeps the accepted context
    nxt = tokens[P + N]
    assert torch.equal(step(fast, a, feeds[0], nxt), step(fast, c, feeds[2], nxt))


# ------------------------------------------------------------------------------------------------------------ graph
@torch.inference_mode()
def test_graph_equals_eager(models, tokens):
    from engine.model.fast import capture
    _, fast = models
    st = fast.new_state(1, L)
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    buf = fast.feed_buffer(1, 1)
    graph, out = capture(lambda: fast(tok, st), None)
    st.reset()
    feed, got = Feeder(fast), []
    for t in tokens[:12]:
        tok.fill_(t)
        buf.copy_(feed([t]))
        graph.replay()
        got.append(out.clone())
    es, feed = fast.new_state(1, L), Feeder(fast)
    for t, g in zip(tokens[:12], got):
        assert torch.equal(step(fast, es, feed, t)[None], g)


# ------------------------------------------------------------------------------------------------------------ state
@torch.inference_mode()
def test_state_hooks_view_reset(models, tokens):
    _, fast = models
    cfg = fast.cfg
    st = fast.new_state(2, L)
    kv = st.kv_tensors()
    n_attn = 1
    assert len(kv) == 4 * n_attn + 1
    assert kv[0].shape == kv[1].shape == (2, 2, L, 256) and kv[0].dtype == torch.float8_e4m3fn
    assert kv[2].shape == (2, 1, L, 128) and kv[3].shape == (2, 1, L // 4, 128) and kv[4].shape == (2, 1, L, 3)
    assert st.kv_ratios() == [1, 1, 1, 4, 1]
    snap = st.snap_tensors()
    assert len(snap) == 3 + 3 + 1
    assert snap[0].shape == (2, 10240, 3) and snap[3].shape == (2, 48, 128, 128) and snap[6].shape == (2, 4 * cfg.hidden_size, 9)
    v = st.view(1, 2)
    for x, y in zip(v.kv_tensors() + v.snap_tensors(), kv + snap):
        assert x.data_ptr() == y[1:2].data_ptr() and x.shape[0] == 1
    # decode on the view writes the parent's slot 1 only
    feed = Feeder(fast)
    for t in tokens[:6]:
        step(fast, v, feed, t)
    assert st.pos_t.tolist() == [0, 6]
    assert st.ple_conv[1][1].any() and not st.ple_conv[1][0].any() and st.ik[3][1, :6].any() and not st.ik[3][0].any()
    assert st.rope_pos[1, :6, 0].tolist() == list(range(6))
    st.reset()
    assert all(not t.view(torch.uint8).any() if t.dtype == torch.float8_e4m3fn else not t.any() for t in st.kv_tensors() + st.snap_tensors())
    assert not st.pos_t.any()
    with pytest.raises(ValueError, match="multiple of 16"):
        fast.new_state(1, 60)


def test_refuses_other_dims():
    from engine.model.fast_qwen4exp import check_dims
    check_dims(small_cfg())
    cfg = dataclasses.replace(small_cfg(), hidden_size=1024, ple_embed_dim=1024)
    with pytest.raises(ValueError, match="hidden_size = 1024"):
        check_dims(cfg)
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        check_dims(small_cfg(moe_intermediate_size=96))


# ------------------------------------------------------------------------------------------------------------ family contract
def test_state_contract_on_meta():
    """What the scheduler, the checkpoint ring and the sessions expect of FastState4 (engine/model/family.py), on the meta
    device (the family's state_bytes builds it there)."""
    from engine.model.fast_qwen4exp import FastState4
    cfg, B, Lm = small_cfg(), 2, 64
    st = FastState4(cfg, B, Lm, "meta")
    assert all(isinstance(d, dict) for d in (st.k, st.v, st.conv, st.rec)) and (len(st.k), len(st.conv), len(st.rec)) == (1, 3, 3)
    kv = st.kv_tensors()
    assert kv[0] is st.k[3] and kv[1] is st.v[3]  # the k / v pairs first, as the 27B's
    assert [tuple(t.shape) for t in kv[2:]] == [(B, 1, Lm, 128), (B, 1, Lm // 4, 128), (B, 1, Lm, 3)]
    for t in kv:
        assert t[1, 0, 4:12].is_contiguous()
    snap = st.snap_tensors()
    assert snap[:3] == [st.conv[i] for i in (0, 1, 2)] and snap[3:6] == [st.rec[i] for i in (0, 1, 2)] and snap[6] is st.ple_conv[1]
    assert all(t[1].is_contiguous() for t in snap)
    assert st.pos == 0 and st.spec is False and st.cfg is cfg
    assert all(t.dtype == torch.int32 and t.shape == (B,) for t in (st.pos_t, st.active, st.rope_delta))


@torch.inference_mode()
def test_state_views_share_and_reset_on_gpu():
    from engine.model.fast_qwen4exp import FastState4
    st = FastState4(small_cfg(), 2, 64, "cuda")
    own = list(st.k.values()) + list(st.v.values()) + list(st.ik.values()) + list(st.ikb.values()) + [st.rope_pos]
    for t in st.kv_tensors():  # views of the state's own storage
        assert t.untyped_storage().data_ptr() in {x.untyped_storage().data_ptr() for x in own}
    st.pos = 5
    st.pos_t.fill_(7)
    st.rope_delta.fill_(-2)
    for t in st.snap_tensors():
        t.fill_(1)
    st.reset(kv=False)
    assert st.pos == 0 and not st.pos_t.any() and not st.rope_delta.any() and all(not t.any() for t in st.snap_tensors())


def test_model_contract(models):
    _, fast = models
    assert fast.drafter_hidden_dim == 4 * 2560 and fast.cfg.vocab_size == V
    assert fast.feed_buffer(1, 1) is fast.feed_buffer(1, 1)
    assert fast.feed_buffer(1, 1).shape == (1, 1, 16, 160) and fast.feed_buffer(1, 1).dtype == torch.uint8


def test_load_refuses_long_context():
    import os

    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from engine.weights.loader import resolve
    try:
        path = resolve("nvidia/Qwen3.8-Flash-Next-NVFP4")
    except Exception:  # noqa: BLE001  no checkpoint on this machine
        pytest.skip("no qwen4_exp checkpoint")
    if not os.path.exists(os.path.join(path, "config.json")):
        pytest.skip("no qwen4_exp checkpoint")
    with pytest.raises(ValueError, match="native context"):
        load_fast_qwen4exp(path, None, print, None, 1 << 21)  # past YaRN x4: refused from the config, before any weight is read

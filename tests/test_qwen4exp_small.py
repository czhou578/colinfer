"""CPU tests of engine/model/qwen4exp.py (the Qwen3.8-Flash-Next reference) on a tiny random model, against
transformers' Qwen4ExpForCausalLM (eager experts, SDPA):
- the config of the checkpoint and its refusals; the n-gram parameters and ids of transformers
- stepwise decode and split prefill against one-shot prefill (splits across eos, the 64-token delta-rule chunk and the
  start of sparse selection)
- transformers without a cache, with a cache, and with an image's positions and embeddings
- sparse selection happens; the NVFP4 experts equal their dequantized dense copy
Run: ./py -m pytest tests/test_qwen4exp_small.py -q
"""
import dataclasses
import json
import os

import numpy as np
import pytest
import torch

from engine.model.qwen4exp import Experts, Qwen4ExpConfig, Qwen4ExpForCausalLM, nvfp4_experts
from engine.weights.ple import ngram_params, ngram_rows

DT = torch.float32
EOS = 3
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "qwen4exp_config.json")


def tiny_cfg(**kw):
    return Qwen4ExpConfig(**({
        "hidden_size": 64, "num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "partial_rotary_factor": 0.25, "rope_theta": 1e4,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "vocab_size": 101, "max_position_embeddings": 4096,
        "mrope_section": [1, 1, 2], "mrope_interleaved": True, "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
        "num_experts": 4, "num_experts_per_tok": 2, "hc_count": 4, "hc_lowrank": 16, "ple_layer_ids": [2], "ple_embed_dim": 64,
        "ngram_vocab_size_base": 1009, "split_ngram_parts": 4, "eos_token_id": EOS, "image_token_id": 100, "indexer_n_heads": 2,
        "indexer_head_dim": 16, "indexer_budget": 16, "indexer_compress_ratio": 4, "output_gate_type": "sigmoid"} | kw))


def init_tiny(m: Qwen4ExpForCausalLM, seed: int = 0) -> Qwen4ExpForCausalLM:
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"):
                p.uniform_(-1, 1)
            elif "ngram_embedding" in n:
                p.normal_(0, 0.5)
            elif "norm" in n:
                p.normal_(0, 0.1)
            else:
                p.normal_(0, 0.08)  # the PLE conv too (transformers zero-initializes it)
    return m.eval()


@pytest.fixture(scope="module")
def model():
    return init_tiny(Qwen4ExpForCausalLM(tiny_cfg()).to(DT))


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(1)
    x = torch.randint(0, 101, (1, 70))  # 70 crosses the 64-token delta-rule chunk; sparse selection from position 19
    x[0, [19, 20, 45]] = EOS  # eos twice in a row, at the split points 19 / 20 / 21, and once alone
    return x


def hf_config(c: Qwen4ExpConfig, **kw):
    from transformers import Qwen4ExpTextConfig
    hc = Qwen4ExpTextConfig(**({
        "hidden_size": c.hidden_size, "num_hidden_layers": c.num_hidden_layers, "layer_types": c.layer_types,
        "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_key_value_heads, "head_dim": c.head_dim,
        "linear_num_key_heads": c.linear_num_key_heads, "linear_num_value_heads": c.linear_num_value_heads,
        "linear_key_head_dim": c.linear_key_head_dim, "linear_value_head_dim": c.linear_value_head_dim,
        "linear_conv_kernel_dim": c.linear_conv_kernel_dim, "rms_norm_eps": c.rms_norm_eps, "vocab_size": c.vocab_size,
        "eos_token_id": c.eos_token_id, "num_experts": c.num_experts, "num_experts_per_tok": c.num_experts_per_tok,
        "moe_intermediate_size": c.moe_intermediate_size, "shared_expert_intermediate_size": c.shared_expert_intermediate_size,
        "hc_count": c.hc_count, "hc_lowrank": c.hc_lowrank, "ple_layer_ids": c.ple_layer_ids, "ple_embed_dim": c.ple_embed_dim,
        "ngram_vocab_size_base": c.ngram_vocab_size_base, "split_ngram_parts": c.split_ngram_parts,
        "indexer_n_heads": c.indexer_n_heads, "indexer_kv_heads": c.indexer_kv_heads, "indexer_head_dim": c.indexer_head_dim,
        "indexer_budget": c.indexer_budget, "indexer_compress_ratio": c.indexer_compress_ratio, "output_gate_type": c.output_gate_type,
        "tie_word_embeddings": False, "experts_implementation": "eager",  # the default grouped_mm sums in another order
        "rope_parameters": {"rope_type": "default", "rope_theta": c.rope_theta, "partial_rotary_factor": c.partial_rotary_factor,
                            "mrope_section": c.mrope_section, "mrope_interleaved": True}} | kw))
    hc._attn_implementation = "sdpa"
    return hc


def _topk_index_ascending(self, k, dim=-1, largest=True, sorted=True):
    v, i = torch.sort(self + 0.0, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((v.narrow(dim, 0, k), i.narrow(dim, 0, k)))


def hf_tiny(model, tie_rule: bool = True):
    """transformers' Qwen4ExpForCausalLM with the weights of the tiny reference model. The names are the same apart
    from the `model.` prefix; the dense experts already have transformers' 3D layout (gate_up_proj [E, 2 I, H],
    down_proj [E, H, I]).

    tie_rule: the indexer's Tensor.topk breaks ties by the engine's rule (score descending, index ascending). The CPU
    topk of torch breaks them in no fixed order, and the tiny indexer (2 heads) has many blocks that score 0, so ties
    at the boundary of the top 4 are common. test_hf_differs_only_at_ties checks the rows without the rule."""
    from transformers import Qwen4ExpForCausalLM as HF

    from tests.hf_fallback import force_hf_torch_fallbacks
    force_hf_torch_fallbacks()
    hf = HF(hf_config(model.cfg)).to(DT).eval()
    sd = {(k if k == "lm_head.weight" else "model." + k): v for k, v in model.state_dict().items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    if tie_rule:
        saved = []

        def pre(mod, args):
            saved.append(torch.Tensor.topk)
            torch.Tensor.topk = _topk_index_ascending

        def post(mod, args, out):
            torch.Tensor.topk = saved.pop()
        for layer in hf.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.indexer.register_forward_pre_hook(pre)
                layer.self_attn.indexer.register_forward_hook(post)
    return hf


def block_scores(model, ids, li=3):
    """The indexer's block scores [T, nb] of attention layer li over one row of ids (one-shot)."""
    from engine.model.qwen4exp import rope_first
    idx = model.layers[li].self_attn.indexer
    cap = {}
    h = idx.register_forward_pre_hook(lambda m, a: cap.update(x=a[0], cos=a[1], sin=a[2]))
    st = model.new_state(1, ids.shape[1])
    try:
        model(ids, st)
    finally:
        h.remove()
    nh, D = idx.n_heads, idx.head_dim
    qk = idx.index_qk_proj(cap["x"])
    q = rope_first(idx.q_layernorm(qk[..., :nh * D].unflatten(-1, (nh, D))), cap["cos"].unsqueeze(2), cap["sin"].unsqueeze(2))
    kb = idx.block_keys(st, li, ids.shape[1], cap["x"].dtype)
    return torch.relu(torch.einsum("thd,bd->tbh", q[0].float(), kb[0].float())).sum(-1) / D ** 0.5


# ------------------------------------------------------------------------------------------------------------ config
def test_config_from_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(open(FIXTURE).read())
    c = Qwen4ExpConfig.from_checkpoint(str(tmp_path))
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (2560, 48, 24, 2, 256)
    assert c.layer_types == Qwen4ExpConfig().layer_types and c.layer_types.count("full_attention") == 12 and c.layer_types[3] == "full_attention"
    assert (c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.shared_expert_intermediate_size) == (512, 10, 640, 640)
    assert (c.hc_count, c.hc_lowrank, c.ple_layers, c.ple_embed_dim, c.ngram_dim, c.ple_state_len) == (4, 320, {1: 0}, 2560, 160, 9)
    assert (c.eos_token_id, c.image_token_id, c.split_ngram_parts, c.output_gate_type) == (248044, 248056, 128, "sigmoid")
    assert (c.indexer_n_heads, c.indexer_head_dim, c.block_topk, c.rotary_dim) == (4, 128, 512, 64)
    assert (c.mrope_section, c.mrope_interleaved, c.rope_theta, c.max_position_embeddings) == ([11, 11, 10], True, 1e7, 262144)
    assert c == Qwen4ExpConfig()  # the defaults are the checkpoint's


@pytest.mark.parametrize("change,err", [
    (lambda t: t["rope_parameters"].update(rope_type="yarn", factor=4.0), NotImplementedError),
    (lambda t: t.update(indexer_kv_heads=2), ValueError),
    (lambda t: t.update(indexer_budget=2050), ValueError),
    (lambda t: t.pop("indexer_head_dim"), ValueError),
    (lambda t: t.update(indexer_head_dim=32), ValueError),  # the rotary dims (64) must fit the index head
    (lambda t: t.update(ple_layer_ids=[4]), ValueError),  # layer index 3 is an attention layer
    (lambda t: t.update(ple_embed_dim=2568), ValueError),
    (lambda t: t.update(output_gate_type="gelu"), ValueError),
    (lambda t: t.update(layer_types=["sliding_attention"] * 48), ValueError),
])
def test_config_refusals(tmp_path, change, err):
    c = json.load(open(FIXTURE))
    change(c["text_config"])
    (tmp_path / "config.json").write_text(json.dumps(c))
    with pytest.raises(err):
        Qwen4ExpConfig.from_checkpoint(str(tmp_path))


def test_state_refuses_more_than_the_native_context(model):
    with pytest.raises(ValueError, match="YaRN"):
        model.new_state(1, model.cfg.max_position_embeddings + 1)


# ------------------------------------------------------------------------------------------------------------ n-gram hash
def test_ngram_params_match_transformers():
    """The real config: the 16 primes, their offsets and the 3 multipliers equal transformers'; the padded table is
    exactly the checkpoint's 128 shards of 2,500,012 rows."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers, _find_nth_prime_after
    c = Qwen4ExpConfig()
    p = ngram_params(c, 0)
    sizes = [_find_nth_prime_after(c.ngram_vocab_size_base - 1, h + 1) for h in range(16)]
    assert list(p.sizes) == sizes and p.sizes[0] == 20000003 and p.sizes[-1] == 20000171
    assert list(p.offsets) == [sum(sizes[:h]) for h in range(16)]
    assert list(p.multipliers) == _build_layer_multipliers(c.vocab_size, 3, 0, c.seed).tolist() == [23703573157769, 20109073645365, 8052911324071]
    assert p.padded_rows == 320_001_536 == 128 * 2_500_012 and sum(sizes) == 320_001_446
    assert max(p.multipliers) * (c.vocab_size - 1) < 2 ** 63  # no int64 wraparound


@torch.inference_mode()
def test_ngram_ids_match_transformers(model, ids):
    """transformers' NGramEmbedding with one dim per head and the row id as its value outputs the row ids. One-shot and
    through a DynamicCache split at every position (eos at a split, and twice in a row); numpy equals torch."""
    from transformers import DynamicCache
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    c = model.cfg
    hc = hf_config(c, ple_embed_dim=16)
    emb = Qwen4ExpTextNGramEmbedding(hc, 16, layer_idx=1, ple_layer_index=0)
    emb.ngram_embedding.weight.copy_(torch.arange(emb.ngram_embedding.num_embeddings, dtype=torch.float64)[:, None])
    emb = emb.double()
    torch.manual_seed(5)
    x = torch.cat([ids, torch.randint(0, 101, (1, ids.shape[1]))])  # two rows
    x[1, [0, 1, 30, 31, 32]] = EOS  # eos at the start of the sequence and three times in a row
    hash_ = model.layers[1].ple.ple_embedding
    eos2 = torch.full((2, 2), EOS)
    ours = hash_.rows(x, eos2)
    assert torch.equal(emb(x, None).long(), ours)
    assert np.array_equal(ngram_rows(x.numpy(), eos2.numpy(), hash_.params), ours.numpy())
    assert ours.max() < hash_.params.padded_rows and ours.min() >= 0
    for s in range(1, x.shape[1]):
        cache = DynamicCache(config=hc)
        got = torch.cat([emb(x[:, :s], cache), emb(x[:, s:], cache)], 1).long()
        assert torch.equal(got, ours), s
        ctx = torch.cat([eos2, x[:, :s]], 1)[:, -2:]
        assert torch.equal(hash_.rows(x[:, s:], ctx), ours[:, s:]), s
    cache = DynamicCache(config=hc)
    steps = torch.cat([emb(x[:, t:t + 1], cache) for t in range(x.shape[1])], 1).long()
    assert torch.equal(steps, ours)


# ------------------------------------------------------------------------------------------------------------ the model alone
@torch.inference_mode()
def test_stepwise_decode_matches_prefill(model, ids):
    full = model(ids, model.new_state(1, 80))
    st = model.new_state(1, 80)
    steps = [model(ids[:, t:t + 1], st) for t in range(ids.shape[1])]
    torch.testing.assert_close(torch.cat(steps, 1), full, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
@pytest.mark.parametrize("split", [1, 3, 19, 20, 21, 40, 69])
def test_split_prefill_matches_one_shot(model, ids, split):
    full = model(ids, model.new_state(1, 80))
    st = model.new_state(1, 80)
    a = model(ids[:, :split], st)
    b = model(ids[:, split:], st)
    torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------------------------------------------------ transformers
@torch.inference_mode()
def test_matches_hf_tiny(model, ids):
    ref = hf_tiny(model)(input_ids=ids, use_cache=False).logits
    ours = model(ids, None)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_hf_differs_only_at_ties(model, ids):
    """Without the tie rule, transformers' selection differs from ours only in rows whose top-4 boundary is a tie, and
    there only among the tied blocks."""
    hf = hf_tiny(model, tie_rule=False)
    hm, om = [], []
    hf.model.layers[3].self_attn.indexer.register_forward_hook(lambda m, a, out: hm.append(out[0, 0]))
    h = model.layers[3].self_attn.indexer.register_forward_hook(lambda m, a, out: om.append(out[0, 0]))
    try:
        hf(input_ids=ids, use_cache=False)
        model(ids, None)
    finally:
        h.remove()
    s = block_scores(model, ids)
    rows = (hm[0] != om[0]).any(-1).nonzero().flatten().tolist()
    assert rows  # the case the rule exists for
    for p in rows:
        nbq = (p + 1) // 4
        sc = s[p, :nbq]
        kth = sc.sort(descending=True).values[3]
        sym = (hm[0][p, :nbq * 4] ^ om[0][p, :nbq * 4]).view(nbq, 4).any(-1)
        assert (sc[sym] == kth).all() and (sc == kth).sum() > 1, p  # only tied blocks swap
        chosen = om[0][p, :nbq * 4].view(nbq, 4).all(-1)
        tied = (sc == kth).nonzero().flatten()
        assert chosen[tied].tolist() == sorted(chosen[tied].tolist(), reverse=True), p  # ours: the lowest indices of the tie


@torch.inference_mode()
def test_matches_hf_tiny_with_cache(model, ids):
    """transformers with a DynamicCache: a prefill of 33 tokens, then 6 single steps; ours in one shot."""
    from transformers import DynamicCache
    hf = hf_tiny(model)
    cache = DynamicCache(config=hf.config)
    ref = [hf(input_ids=ids[:, :33], past_key_values=cache, use_cache=True).logits]
    ref += [hf(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits for t in range(33, 39)]
    ours = model(ids[:, :39], model.new_state(1, 39))
    torch.testing.assert_close(ours, torch.cat(ref, 1), rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_sparse_selection_happens(model, ids):
    """With a budget of 16 tokens (4 blocks), a query at position p >= 19 attends to 16 + (p + 1) % 4 tokens; below,
    to all of its causal prefix. A copy with a huge budget (dense attention) agrees below 19 and differs above."""
    masks = []
    h = model.layers[3].self_attn.indexer.register_forward_hook(lambda m, a, out: masks.append(out))
    try:
        ours = model(ids, None)
    finally:
        h.remove()
    m = masks[0][0, 0]
    assert m.shape == (70, 70)
    for p in range(70):
        assert not m[p, p + 1:].any()
        assert m[p, :p + 1].sum() == (p + 1 if p < 19 else 16 + (p + 1) % 4), p
    assert m[60].sum() == 17
    dense = Qwen4ExpForCausalLM(dataclasses.replace(model.cfg, indexer_budget=1_000_000)).to(DT).eval()
    dense.load_state_dict(model.state_dict())
    full = dense(ids, None)
    assert torch.equal(full[:, :19], ours[:, :19])
    assert not torch.allclose(full[:, 19:], ours[:, 19:], atol=1e-3)


@torch.inference_mode()
def test_matches_hf_tiny_with_an_image(model, ids):
    """Rows 10 .. 33 are an image (a 4 x 6 token grid): input embeddings of their own and the (t, h, w) positions of
    engine/model/vision.py. transformers needs ple_input_ids with input embeddings (else it inverts the embedding)."""
    from engine.model.vision import Image, rope_positions
    im = Image(torch.empty(0), (1, 8, 12), key=0, start=10)
    torch.manual_seed(2)
    emb = model.embed_tokens(ids).clone()
    emb[0, 10:10 + im.n] = torch.randn(im.n, model.cfg.hidden_size) * 0.5
    pos = torch.from_numpy(rope_positions([im], 0, ids.shape[1])).long()
    T = ids.shape[1]
    hpos = torch.cat([torch.arange(T)[None], pos.T])[:, None]  # [4, 1, T]: the text positions, then (t, h, w)
    ref = hf_tiny(model)(inputs_embeds=emb, position_ids=hpos, ple_input_ids=ids, use_cache=False).logits
    ours = model(ids, None, positions=pos, embeds=emb)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(ours, model(ids, None, embeds=emb), atol=1e-3)  # the positions matter


# ------------------------------------------------------------------------------------------------------------ NVFP4 experts
@torch.inference_mode()
def test_nvfp4_experts_equal_dense_dequant(model, ids):
    """NVFP4 experts (engine.weights.quantize, one global scale for gate and up) equal a dense copy of their dequantized
    weights, bit for bit, in the module and in the model."""
    dense = model.layers[0].mlp.experts
    q = nvfp4_experts(dense)
    assert q.w_gu.shape == (4, 64, 32) and q.sf_gu.shape == (4, 64, 4) and q.w_dn.shape == (4, 64, 16) and q.sf_dn.shape == (4, 64, 2)
    assert torch.equal(q.gs[:, 0], q.gs[:, 1])
    deq = Experts(4, 64, 32, "dense")
    for e in range(4):
        gu, dn = q.get(e, DT)
        deq.gate_up_proj.data[e], deq.down_proj.data[e] = gu, dn
        assert (gu - dense.gate_up_proj[e]).abs().max() < 0.25 * dense.gate_up_proj[e].abs().max()  # a sane quantization
    torch.manual_seed(3)
    x = torch.randn(9, 64)
    _, w, idx = model.layers[0].mlp.gate(x)
    assert torch.equal(q(x, idx, w), deq(x, idx, w))
    mq = Qwen4ExpForCausalLM(model.cfg, experts="nvfp4").to(DT).eval()
    md = Qwen4ExpForCausalLM(model.cfg).to(DT).eval()
    sd = model.state_dict()
    mq.load_state_dict({k: v for k, v in sd.items() if ".experts." not in k}, strict=False)
    md.load_state_dict(sd)
    for lq, ld, lr in zip(mq.layers, md.layers, model.layers):
        qe = nvfp4_experts(lr.mlp.experts)
        lq.mlp.experts.load_state_dict(qe.state_dict())
        for e in range(4):
            ld.mlp.experts.gate_up_proj.data[e], ld.mlp.experts.down_proj.data[e] = qe.get(e, DT)
    assert torch.equal(mq(ids, None), md(ids, None))

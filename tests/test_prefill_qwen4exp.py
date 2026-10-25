"""engine/model/prefill_qwen4exp.py: each layer function of the qwen4_exp prefill against the reference modules of
engine/model/qwen4exp.py in bf16 on the GPU, at the real dims with random weights, over two chunks (the state carries
from the first to the second):
- hc_pre / hc_post against GatedResidual and inject
- gdn_layer against GatedDeltaNet4 (the sigmoid output gate), the conv and recurrent state included
- qsa_layer against IndexedAttention (the QSA indexer and SDPA over its mask), through the 2051 boundary
- moe_layer against moe_reference
- ple_layer against PLELayer (the n-gram rows as FP8 bytes), the conv state included
The tolerances are relative L2 errors (rel) and are stated at each assert.
Run: uv run pytest tests/test_prefill_qwen4exp.py -q"""
import math
from types import SimpleNamespace

import pytest
import torch

from engine.model import prefill_qwen4exp as pq
from engine.model.hc_kernels import KernelGatedResidual
from engine.model.qwen4exp import GatedDeltaNet4, GatedResidual, IndexedAttention, PLELayer, Qwen4ExpConfig, Qwen4ExpState, inject
from engine.model.qwen35 import rope_inv_freq, rotary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
DEV = "cuda"


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def randomize(module, seed):
    """Random weights: linear weights ~ N(0, 1 / fan_in), norm weights ~ 0.2 N(0, 1), the rest by name."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if name.endswith("A_log"):
                r = torch.rand(p.shape, generator=g) * 2
            elif name.endswith("dt_bias"):
                r = r * 0.5
            elif "norm" in name or "layernorm" in name:
                r = r * 0.2
            elif "conv1d" in name:
                r = r * 0.3
            elif p.dim() == 2:
                r = r / math.sqrt(p.shape[1])
            p.copy_(r)
    return module


@pytest.fixture(scope="module")
def cfg():
    return Qwen4ExpConfig()


def stream(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(T, 10240, device=DEV, generator=g).bfloat16()


# ------------------------------------------------------------------------------------------------------------ HC
def test_hc_pre_post(cfg):
    for combine in (True, False):
        ref = randomize(GatedResidual(cfg, combine), 1 + combine).to(DEV, torch.bfloat16)
        inj_w = ref.block_inject_weight.weight if combine else None
        hc = KernelGatedResidual(ref.hc_norm.weight, ref.input_mix_weight_down.weight, inj_w, ref.input_mix_weight_up.weight, cfg.rms_norm_eps)
        x4 = stream(700, 3)
        got = pq.hc_pre(x4, hc)
        want = ref(x4)
        if combine:
            y, inj = got
            y_ref, inj_ref = want
            assert rel(y, y_ref) < 4e-3  # bf16 rounding points of the reference; cuBLAS may reduce in another order
            assert rel(inj, inj_ref) < 4e-3
            # the decode kernels on the same rows
            yk, injk = hc(x4[:16])
            assert rel(y[:16], yk) < 4e-3 and rel(inj[:16], injk) < 4e-3
            o = stream(700, 4)[:, :2560]
            assert torch.equal(pq.hc_post(x4.clone(), o, inj), inject(x4, o, inj))  # the injection is bit-exact
        else:
            assert rel(got, want) < 4e-3


# ------------------------------------------------------------------------------------------------------------ GDN
def test_gdn_layer(cfg):
    li = 0
    ref = randomize(GatedDeltaNet4(cfg), 11).to(DEV, torch.bfloat16)
    ref.A_log.data = ref.A_log.data.float()
    ref.dt_bias.data = ref.dt_bias.data.float()
    C = ref.conv_dim
    g = SimpleNamespace(qkvz=SimpleNamespace(w=torch.cat([ref.in_proj_qkv.weight, ref.in_proj_z.weight]), sizes=[C, ref.value_dim]),
                        w_ba=torch.cat([ref.in_proj_b.weight, ref.in_proj_a.weight]), conv1d=ref.conv1d, A_log=ref.A_log, dt_bias=ref.dt_bias,
                        norm=ref.norm, out_proj=SimpleNamespace(w=ref.out_proj.weight))
    st_ref = SimpleNamespace(pos=0, conv={li: torch.zeros(1, C, 3, device=DEV, dtype=torch.bfloat16)},
                             rec={li: torch.zeros(1, 48, 128, 128, device=DEV)})
    st = SimpleNamespace(conv={li: torch.zeros(1, C, 3, device=DEV, dtype=torch.bfloat16)}, rec={li: torch.zeros(1, 48, 128, 128, device=DEV)})
    gen = torch.Generator(device=DEV).manual_seed(12)
    for T in (300, 2, 129):  # a chunk shorter than the conv window too
        x = torch.randn(T, 2560, device=DEV, generator=gen).bfloat16()
        want = ref(x[None], st_ref, li)[0]
        st_ref.pos += T
        got = pq.gdn_layer(x, g, st, li, cfg.rms_norm_eps)
        assert rel(got, want) < 2e-2, T  # gdn_prefill vs the torch chunked rule (fp32 vs bf16 operand roundings)
        assert torch.equal(st.conv[li], st_ref.conv[li])  # the conv window: the last 3 inputs, bit for bit
        assert rel(st.rec[li], st_ref.rec[li]) < 2e-2


# ------------------------------------------------------------------------------------------------------------ QSA
def test_qsa_layer(cfg):
    li, L = 3, 4096
    ref = randomize(IndexedAttention(cfg), 21).to(DEV, torch.bfloat16)
    a = SimpleNamespace(qkvi=SimpleNamespace(w=torch.cat([ref.q_proj.weight, ref.k_proj.weight, ref.v_proj.weight, ref.indexer.index_qk_proj.weight]),
                                             sizes=[12288, 512, 512, 640]),
                        q_norm=ref.q_norm, k_norm=ref.k_norm, indexer=ref.indexer, o_proj=SimpleNamespace(w=ref.o_proj.weight),
                        inv_freq=rope_inv_freq(cfg, DEV), mrope=tuple(cfg.mrope_section[1:]))
    st_ref = Qwen4ExpState(cfg, 1, L, DEV, torch.bfloat16, kv_dtype=torch.float8_e4m3fn)
    f8 = torch.float8_e4m3fn
    st = SimpleNamespace(k={li: torch.zeros(1, 2, L, 256, device=DEV, dtype=f8)}, v={li: torch.zeros(1, 2, L, 256, device=DEV, dtype=f8)},
                         ik={li: torch.zeros(1, L, 128, device=DEV, dtype=torch.bfloat16)},
                         ikb={li: torch.zeros(1, L // 4, 128, device=DEV, dtype=torch.bfloat16)},
                         rope_pos=torch.zeros(1, L, 3, device=DEV, dtype=torch.int32), pos_t=torch.zeros(1, device=DEV, dtype=torch.int32),
                         rope_delta=torch.zeros(1, device=DEV, dtype=torch.int32))
    gen = torch.Generator(device=DEV).manual_seed(22)
    pos = 0
    for T in (1500, 1000, 7):  # all rows dense; through the 2051 boundary; sparse rows
        x = (torch.randn(T, 2560, device=DEV, generator=gen) * 2).bfloat16()
        p = torch.arange(pos, pos + T, device=DEV)
        st_ref.rope_pos[0, pos:pos + T] = p[:, None]
        cos, sin = rotary(cfg, p, torch.bfloat16)
        want = ref(x[None], cos, sin, st_ref, li)[0]
        st_ref.pos += T
        st.pos_t.fill_(pos)
        got = pq.qsa_layer(x, a, st, li, pos, None, cfg.rms_norm_eps)
        # dense rows: fp32 RoPE (the reference rounds cos / sin to bf16), f16 P (measured 0.6%); sparse rows: the RoPE rounding
        # also moves the scores near the 512th, and random keys put many blocks there (measured 1.7% and 2.1%)
        assert rel(got, want) < (1e-2 if pos + T <= 2051 else 3e-2), (pos, T)
        assert rel(st.ik[li][0, pos:pos + T], st_ref.ik[li][0, 0, pos:pos + T]) < 1e-2  # the raw keys (stacked vs separate projection)
        assert rel(st.k[li][:, :, pos:pos + T].float(), st_ref.k[li][:, :, pos:pos + T].float()) < 3e-2
        assert torch.equal(st.rope_pos[0, pos:pos + T].long(), st_ref.rope_pos[0, pos:pos + T])
        pos += T


# ------------------------------------------------------------------------------------------------------------ MoE
def test_moe_layer():
    from engine.model.moe_kernels import KernelMoE, moe_reference, random_moe_weights
    w = random_moe_weights(64, 2560, 640, generator=torch.Generator(device=DEV).manual_seed(31))
    m = KernelMoE(**w)
    x = torch.randn(300, 2560, device=DEV, generator=torch.Generator(device=DEV).manual_seed(32)).bfloat16()
    got = pq.moe_layer(x, m)
    want = moe_reference(x, **{k: v for k, v in w.items() if k != "in_scale"})
    assert rel(got, want) < 5e-2  # the prefill quantizes the activations to NVFP4 (about 2% from the decode path)
    assert torch.equal(got, m.prefill(x))


# ------------------------------------------------------------------------------------------------------------ PLE
def test_ple_layer():
    cfg = Qwen4ExpConfig(ngram_vocab_size_base=300)  # a small table: the real dims, a few thousand rows
    li = 1
    ref = randomize(PLELayer(cfg, 0), 41).to(DEV, torch.bfloat16)
    p = ref.ple_embedding.params
    g = torch.Generator().manual_seed(42)
    table8 = (torch.randn(p.padded_rows, cfg.ngram_dim, generator=g) * 64).to(torch.float8_e4m3fn)
    scale = 0.0078125
    with torch.no_grad():
        ref.ple_embedding.ngram_embedding.weight.copy_((table8.float() * scale).bfloat16())  # the reference reads the dequantized rows
    table_u8 = table8.view(torch.uint8).to(DEV)
    ple = SimpleNamespace(kv_proj=SimpleNamespace(w=torch.cat([ref.key_proj.weight, ref.value_proj.weight]), sizes=[10240, 2560]),
                          norm_key=ref.norm_key, norm_query=ref.norm_query, norm_conv=ref.norm_conv, conv1d=ref.conv1d, scale=scale)
    eos = cfg.eos_token_id
    st_ref = SimpleNamespace(ple_ctx={li: torch.full((1, 2), eos, device=DEV, dtype=torch.long)},
                             ple_conv={li: torch.zeros(1, 10240, 9, device=DEV, dtype=torch.bfloat16)})
    st = SimpleNamespace(ple_conv={li: torch.zeros(1, 10240, 9, device=DEV, dtype=torch.bfloat16)})
    gt = torch.Generator(device=DEV).manual_seed(43)
    for T in (500, 5, 64):  # a chunk shorter than the conv window too
        tok = torch.randint(0, 200000, (1, T), device=DEV, generator=gt)
        tok[0, ::17] = eos
        ids = ref.ple_embedding.rows(tok, st_ref.ple_ctx[li])
        rows = table_u8[ids[0]]  # [T, 16, 160]
        h = stream(T, T)
        want = h + ref(h[None], tok, st_ref, li)[0]
        got = pq.ple_layer(h.clone(), rows, ple, st, li)
        assert rel(got, want) < 1e-2, T  # the conv in fp32 taps (the reference: bf16 conv1d), fixed-order sums
        assert rel(st.ple_conv[li], st_ref.ple_conv[li]) < 1e-2


# ------------------------------------------------------------------------------------------------------------ the chunk loop
class _Lin:
    def __init__(self, w):
        self.w = w

    def __call__(self, x):
        return x @ self.w.t()


def _hc(ref):
    inj = ref.block_inject_weight.weight if ref.block_inject_weight is not None else None
    return KernelGatedResidual(ref.hc_norm.weight, ref.input_mix_weight_down.weight, inj, ref.input_mix_weight_up.weight, ref.hc_norm.eps)


def _mini(cfg, seed):
    """A 4-layer model at the real dims (GDN, GDN with PLE, GDN, attention; 64 experts): the reference, and a stand-in
    for the decode model with its attribute names (FastQwen4Exp) over the same weights."""
    from engine.model.moe_kernels import KernelMoE, random_moe_weights
    from engine.model.qwen4exp import Qwen4ExpForCausalLM
    ref = randomize(Qwen4ExpForCausalLM(cfg, experts="nvfp4"), seed).to(DEV)
    for p in ref.parameters():
        p.data = p.data.bfloat16()
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = ref.layers[1].ple.ple_embedding.params
    table8 = (torch.randn(p.padded_rows, cfg.ngram_dim, device=DEV, generator=g) * 64).to(torch.float8_e4m3fn)
    scale = 0.0078125
    ref.layers[1].ple.ple_embedding.ngram_embedding.weight.data = (table8.float() * scale).bfloat16()
    layers = []
    for i, layer in enumerate(ref.layers):
        w = random_moe_weights(cfg.num_experts, 2560, 640, generator=g)
        ex, moe = layer.mlp.experts, layer.mlp
        for k in ("w_gu", "sf_gu", "w_dn", "sf_dn", "gs", "in_scale"):
            getattr(ex, k).copy_(w[k])
        moe.gate.weight.data = w["router_w"]
        moe.shared_expert_gate.weight.data = w["shared_gate_w"]
        moe.shared_expert.gate_proj.weight.data, moe.shared_expert.up_proj.weight.data = w["sh_gate"], w["sh_up"]
        moe.shared_expert.down_proj.weight.data = w["sh_down"]
        ns = SimpleNamespace(block_type=layer.block_type, attn_hc=_hc(layer.attn_hyper_connection), mlp_hc=_hc(layer.mlp_hyper_connection),
                             mlp=KernelMoE(**w), ple=None)
        if layer.block_type == "linear_attention":
            r = layer.linear_attn
            qkvz = SimpleNamespace(w=torch.cat([r.in_proj_qkv.weight, r.in_proj_z.weight]), sizes=[r.conv_dim, r.value_dim])
            ns.linear_attn = SimpleNamespace(qkvz=qkvz,
                                             w_ba=torch.cat([r.in_proj_b.weight, r.in_proj_a.weight]), conv1d=r.conv1d, A_log=r.A_log,
                                             dt_bias=r.dt_bias, norm=r.norm, out_proj=_Lin(r.out_proj.weight))
        else:
            r = layer.self_attn
            ns.self_attn = SimpleNamespace(qkvi=SimpleNamespace(w=torch.cat([r.q_proj.weight, r.k_proj.weight, r.v_proj.weight,
                                                                             r.indexer.index_qk_proj.weight]), sizes=[12288, 512, 512, 640]),
                                           q_norm=r.q_norm, k_norm=r.k_norm, indexer=r.indexer, o_proj=_Lin(r.o_proj.weight),
                                           inv_freq=rope_inv_freq(cfg, DEV), mrope=tuple(cfg.mrope_section[1:]))
        if layer.ple is not None:
            r = layer.ple
            ns.ple = SimpleNamespace(kv_proj=SimpleNamespace(w=torch.cat([r.key_proj.weight, r.value_proj.weight]), sizes=[10240, 2560]),
                                     norm_key=r.norm_key, norm_query=r.norm_query, norm_conv=r.norm_conv, conv1d=r.conv1d, scale=scale)
        layers.append(ns)
    fast = SimpleNamespace(cfg=cfg, embed_tokens=ref.embed_tokens, layers=layers, mixer=_hc(ref.hyper_connection_mixer),
                           lm_head=_Lin(ref.lm_head.weight))
    return ref, fast, table8.view(torch.uint8)


def _fast_state(cfg, L):
    f8, bf = torch.float8_e4m3fn, torch.bfloat16
    att = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    gdn = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    z = lambda *s, dt=bf: torch.zeros(*s, device=DEV, dtype=dt)  # noqa: E731
    return SimpleNamespace(k={i: z(1, 2, L, 256, dt=f8) for i in att}, v={i: z(1, 2, L, 256, dt=f8) for i in att}, ik={i: z(1, L, 128) for i in att},
                           ikb={i: z(1, L // 4, 128) for i in att}, rope_pos=z(1, L, 3, dt=torch.int32), conv={i: z(1, 10240, 3) for i in gdn},
                           rec={i: z(1, 48, 128, 128, dt=torch.float32) for i in gdn}, ple_conv={i: z(1, 10240, 9) for i in cfg.ple_layers},
                           pos=0, pos_t=z(1, dt=torch.int32), rope_delta=z(1, dt=torch.int32), max_seq_len=L)


def test_prefill_mini_model(tmp_path):
    """prefill() on a 4-layer model against the reference forward: every token's logits and the 4-stream hidden over
    three chunks (through the 2051 boundary), the last-token logits of one chunk, and the state it leaves."""
    cfg = Qwen4ExpConfig(num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"], num_experts=64, vocab_size=4096,
                         eos_token_id=4000, image_token_id=4001, ngram_vocab_size_base=300)
    ref, fast, table = _mini(cfg, 51)
    T, L = 2100, 2304
    ids = torch.randint(0, 4000, (1, T), device=DEV, generator=torch.Generator(device=DEV).manual_seed(52))
    ids[0, 500] = cfg.eos_token_id
    st_ref = ref.new_state(1, L, kv_dtype=torch.float8_e4m3fn)
    rows = table[ref.layers[1].ple.ple_embedding.rows(ids, st_ref.ple_ctx[1].clone())[0]]  # [T, 16, 160]
    want, h_ref = ref(ids, st_ref, return_hyper=True)
    st = _fast_state(cfg, L)
    logits, hid = pq.prefill(fast, ids, st, chunk=1024, all_logits=True, return_hidden=True, rows=rows)
    assert logits.shape == (T, cfg.vocab_size) and logits.dtype == torch.bfloat16 and hid.shape == (T, 10240)
    assert rel(hid, h_ref[0]) < 3e-2  # NVFP4 activations in the MoE prefill (about 2% a layer) dominate
    assert rel(logits, want[0]) < 5e-2
    assert st.pos == T and int(st.pos_t) == T
    assert rel(st.rec[2], st_ref.rec[2]) < 5e-2 and rel(st.conv[0], st_ref.conv[0]) < 1e-2
    assert rel(st.ple_conv[1], st_ref.ple_conv[1]) < 1e-2
    assert rel(st.ik[3][0, :T], st_ref.ik[3][0, 0, :T]) < 3e-2
    # one chunk, the last token's logits through the mixer and lm_head of the decode model
    st2 = _fast_state(cfg, L)
    last = pq.prefill(fast, ids[:, :1000], st2, chunk=2048, rows=rows[:1000])
    assert last.shape == (1, cfg.vocab_size) and last.dtype == torch.float32
    assert rel(last, logits[999:1000]) < 3e-2

    # the rows from the model's feeder (rows=None) over the same table on disk, in one call and in two calls (prev2)
    from engine.weights.ple import NgramTable, PleFeeder
    from tests.qwen4exp_ckpt import ngram_shards, write_safetensors
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    path = str(tmp_path / "ple.safetensors")
    write_safetensors(path, [(prefix + ".weight_scale", torch.tensor([0.0078125], dtype=torch.bfloat16))] +
                      ngram_shards(table.view(torch.float8_e4m3fn).cpu(), 4, [2, 0, 3, 1], prefix))
    tab = NgramTable(path, prefix)
    fast.layers[1].ple.table, fast.layers[1].ple.params = tab, ref.layers[1].ple.ple_embedding.params
    pq.prepare_prefill(fast, cache_rows=100000)  # attaches the feeder (the family's prepare_prefill), once
    feeder = fast.feeder
    assert isinstance(feeder, PleFeeder) and feeder.stats()[3] >= 100000 and pq.prepare_prefill(fast).feeder is feeder
    st3 = _fast_state(cfg, L)
    via_feeder = pq.prefill(fast, ids, st3, chunk=1024, all_logits=True)
    assert torch.equal(via_feeder, logits)
    st4 = _fast_state(cfg, L)
    a = pq.prefill(fast, ids[:, :1024], st4, chunk=1024, all_logits=True)
    with pytest.raises(ValueError):
        pq.prefill(fast, ids[:, 1024:], st4, chunk=1024, all_logits=True)  # a continued prompt needs prev2
    b = pq.prefill(fast, ids[:, 1024:], st4, chunk=1024, all_logits=True, prev2=ids[0, 1022:1024].tolist())
    assert torch.equal(torch.cat([a, b]), logits)
    # the next call's rows gathered ahead (next_ids), then a call with other tokens than announced: the same logits
    for nxt in (ids[0, 1024:2048].tolist(), ids[0, 1000:2024].tolist()):
        st5 = _fast_state(cfg, L)
        a = pq.prefill(fast, ids[:, :1024], st5, chunk=1024, all_logits=True, next_ids=nxt)
        assert fast.feeder._prefill_pending is not None
        b = pq.prefill(fast, ids[:, 1024:], st5, chunk=1024, all_logits=True, prev2=ids[0, 1022:1024].tolist())
        assert torch.equal(torch.cat([a, b]), logits)
    fast.feeder.close()
    tab.close()

"""The requantized decode copies of qwen4_exp (tools/requant_qwen4exp.py, engine/model/fast_qwen4exp.py
attach_decode_copies4) on the small random model of tests/test_fast_qwen4exp.py (hidden 2560, 4 layers: GDN, GDN + PLE,
GDN, QSA). The tool's functions quantize the model's BF16 matrices from its state dict under the checkpoint's names, write
the files into a temporary prepared directory, and the attach reads them. The tests check:
- the plan of the groups and the count of linears attached (every class)
- the decode path with the copies against the BF16 path over 40 decode steps (relative error, argmax at non-near-ties)
- row invariance with the copies: a verify of 8 rows equals 8 single steps bit for bit, and commit
- a captured CUDA graph with the copies equals eager
- the prefill keeps the BF16 weights: its 4-stream hidden is the same bits with and without the copies
- each class alone (the files are independent), and a stacked group with a part missing is refused
- the tool on the synthetic nvidia-layout checkpoint of tests/qwen4exp_ckpt.py (names from the shards)
Run: uv run pytest tests/test_requant_qwen4exp.py -q"""
import copy
import os

import pytest
import torch

from tests.test_fast_qwen4exp import EOS, Feeder, L, V, build_reference, step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

P = "model.language_model."
# a copy of every class, each format once at least (not the default formats: tools/requant_qwen4exp.py records those)
EVERY = {"gdn_in": "int5", "gdn_out": "int6", "attn": "int6", "lm_head": "nvfp4", "hc": "int6", "hc_up": "int8", "shared": "int6"}
TOL, REL = 0.6, 0.15  # all copies against the BF16 path: max |logit difference| / std and the relative L2 per step (measured
                      # 0.38 and 0.10 on these random weights: the NVFP4 lm_head 0.34 / 0.09, the INT5 GDN 0.16 / 0.04 alone)
SIM_TOL, SIM_REL = 0.06, 0.02  # the copies against the exact path on their dequantized weights (measured 0.028 and 0.0074)


def checkpoint_view(ref):
    """(the tensor names, load) of the reference model as the nvidia checkpoint names them (tests/qwen4exp_ckpt.py)."""
    sd = ref.state_dict()
    names = {("lm_head.weight" if k == "lm_head.weight" else P + k): k for k in sd}
    return list(names), lambda n: sd[names[n]]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """(the BF16-path model, the model with all copies attached, the copy files, the attach count)."""
    from engine.model.fast_qwen4exp import attach_decode_copies4, to_fast4
    from tools.requant_qwen4exp import make_copies, write_copies
    ref = build_reference()
    names, load = checkpoint_view(ref)
    out = str(tmp_path_factory.mktemp("prepared"))
    files = [os.path.join(out, f) for f in write_copies(out, make_copies(names, load, EVERY))]
    exact = to_fast4(copy.deepcopy(ref))
    fast = to_fast4(copy.deepcopy(ref))
    n = attach_decode_copies4(fast, files)
    return exact, fast, files, n, ref


def test_plan_and_attach_count(models):
    from engine.model.fast import IntLinear, Nvfp4Linear
    from tools.requant_qwen4exp import plan
    exact, fast, files, n, ref = models
    groups = plan(checkpoint_view(ref)[0])
    assert {c: len(g) for c, g in groups.items()} == {"gdn_in": 3, "gdn_out": 3, "attn": 2, "lm_head": 1, "hc": 9, "hc_up": 9, "shared": 4}
    assert groups["attn"][0][-1] == P + "layers.3.self_attn.indexer.index_qk_proj"
    assert groups["hc"][-1] == (P + "hyper_connection_mixer.input_mix_weight_down",)
    # 3 GDN layers x 3, attention 4 + 1, 8 gated residuals x 2 + the mixer, 9 W_up, 4 shared gate | up x 2, the lm_head
    assert n == 9 + 5 + 17 + 9 + 8 + 1
    assert sorted(os.path.basename(f) for f in files) == sorted(f"decode4_{c}_{f}.safetensors" for c, f in EVERY.items())
    g, a = fast.layers[0].linear_attn, fast.layers[3].self_attn
    assert isinstance(g.qkvz.dec, IntLinear) and g.qkvz.dec.whi.shape[1] * 8 == 2560  # INT5: one high bit per code
    assert isinstance(g.out_proj.dec, IntLinear) and g.out_proj.dec.whi.shape[1] * 4 == 6144  # INT6: two
    assert isinstance(a.qkvi.dec, IntLinear) and a.qkvi.dec.whi.shape[1] * 4 == 2560 and a.qkvi.dec.out_features == 13952
    assert isinstance(fast.lm_head.dec, Nvfp4Linear) and fast.lm_head.dec.out_fp32
    hc = fast.layers[0].attn_hc
    assert hc.dec.out_features == 328 and hc.w_up_q.dtype == torch.int8 and fast.mixer.dec.out_features == 320
    assert fast.layers[0].mlp.sh_gu_dec.out_features == 256
    # the prefill's BF16 weights are the exact path's
    assert torch.equal(g.qkvz.w, exact.layers[0].linear_attn.qkvz.w) and torch.equal(hc.w_di, exact.layers[0].attn_hc.w_di)
    assert exact.lm_head.dec is None and exact.layers[0].mlp.sh_gu_dec is None and exact.layers[0].attn_hc.dec is None


def _compare(exact, fast, toks, near=0.1):
    """(the worst max|diff| / std, the worst relative L2, the argmax flips, the steps checked: a margin above near x std)."""
    se, sf = exact.new_state(1, L), fast.new_state(1, L)
    fe, ff = Feeder(exact), Feeder(fast)
    worst = worst_rel = 0.0
    flips = checked = 0
    for tok in toks:
        want, got = step(exact, se, fe, tok), step(fast, sf, ff, tok)
        worst = max(worst, ((got - want).abs().max() / want.std()).item())
        worst_rel = max(worst_rel, ((got - want).norm() / want.norm()).item())
        top2 = want.topk(2).values
        if (top2[0] - top2[1]).item() > near * want.std().item():
            checked += 1
            flips += int(got.argmax() != want.argmax())
    return worst, worst_rel, flips, checked


@pytest.fixture(scope="module")
def tokens():
    g = torch.Generator().manual_seed(1)
    t = torch.randint(4, V - 1, (48,), generator=g)
    t[[10, 11, 30]] = EOS
    return t.tolist()


@torch.inference_mode()
def test_copies_match_their_dequantized_weights(models, tokens):
    """The kernels and the plumbing: the decode path with the copies against the exact path of a model whose BF16
    weights are the copies' dequantized weights (rounded to bf16), over 40 steps. Only the rounding of the dequantized
    weights to bf16 and the order of the fp32 sums differ."""
    from engine.model.fast_qwen4exp import to_fast4
    from engine.weights.direct import read_tensors
    from tools.requant_qwen4exp import dequantize, plan
    exact, fast, files, _, ref = models
    t = {}
    for f in files:
        t.update(read_tensors(f, device="cuda"))
    sim = copy.deepcopy(ref)
    names, _ = checkpoint_view(sim)
    sd = sim.state_dict()
    for groups in plan(names).values():
        for mods in groups:
            for m in mods:
                k = "lm_head.weight" if m == "lm_head" else m[len(P):] + ".weight"
                sd[k].copy_(dequantize(t, m).to(sd[k].dtype))
    sim = to_fast4(sim)
    worst, worst_rel, flips, checked = _compare(sim, fast, tokens[:40])
    print(f"\n[requant4] the copies vs the exact path on their dequantized weights, 40 steps: max|diff| / std {worst:.4f}, "
          f"relative L2 {worst_rel:.4f}; argmax checked at {checked} steps, {flips} differ")
    assert worst < SIM_TOL and worst_rel < SIM_REL, (worst, worst_rel)
    assert flips == 0 and checked >= 25


@torch.inference_mode()
def test_copies_against_bf16_path(models, tokens):
    """The format error on random weights (the perplexity gate on the real checkpoint decides the formats): all copies
    against the BF16 path over 40 steps. The NVFP4 lm_head dominates: its weight error is 8.5% on Gaussian weights."""
    exact, fast, *_ = models
    worst, worst_rel, flips, checked = _compare(exact, fast, tokens[:40], near=0.5)
    print(f"\n[requant4] all copies vs the BF16 path over 40 steps: max|diff| / std {worst:.4f}, relative L2 {worst_rel:.4f}; "
          f"argmax checked at {checked} steps (margin above 0.5 std), {flips} differ")
    assert worst < TOL and worst_rel < REL, (worst, worst_rel)
    assert flips == 0 and checked >= 5  # random logits have small margins: 8 of 40 steps here


@torch.inference_mode()
def test_verify_equals_steps_with_copies(models, tokens):
    """Every copied path is row-invariant: a verify of 8 rows equals 8 single steps bit for bit; commit(5) leaves the
    state of 5 single steps."""
    from tests.test_fast_qwen4exp import _state_equal
    _, fast, *_ = models
    P0, T, N = 13, 8, 5
    sts = [fast.new_state(1, L) for _ in range(3)]
    feeds = [Feeder(fast) for _ in range(3)]
    for st, fd in zip(sts, feeds):
        for tok in tokens[:P0]:
            step(fast, st, fd, tok)
    a, b, c = sts
    draft = tokens[P0:P0 + T]
    logits, _ = fast.verify(torch.tensor([draft], device="cuda"), a, feeds[0](draft))
    for t, tok in enumerate(draft):
        assert torch.equal(logits[0, t], step(fast, b, feeds[1], tok)), t
    fast.commit(a, torch.tensor([N], dtype=torch.int32, device="cuda"))
    for tok in draft[:N]:
        step(fast, c, feeds[2], tok)
    _state_equal(a, c, P0 + N)


@torch.inference_mode()
def test_graph_equals_eager_with_copies(models, tokens):
    from engine.model.fast import capture
    _, fast, *_ = models
    st = fast.new_state(1, L)
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    buf = fast.feed_buffer(1, 1)
    graph, out = capture(lambda: fast(tok, st), None)
    st.reset()
    feed, got = Feeder(fast), []
    for t in tokens[:8]:
        tok.fill_(t)
        buf.copy_(feed([t]))
        graph.replay()
        got.append(out.clone())
    es, feed = fast.new_state(1, L), Feeder(fast)
    for t, g in zip(tokens[:8], got):
        assert torch.equal(step(fast, es, feed, t)[None], g)


@torch.inference_mode()
def test_prefill_keeps_bf16(models, tokens):
    """The prefill reads the BF16 weights: the 4-stream hidden of every prompt token is the same bits with the copies."""
    from engine.model import prefill_qwen4exp as pq
    exact, fast, *_ = models
    ids = torch.tensor([tokens[:40]], device="cuda")
    rows = exact.host_rows(ids.cpu(), [[EOS, EOS]])[0].cuda()
    hid = [pq.prefill(m, ids, m.new_state(1, L), logits=False, return_hidden=True, rows=rows)[1] for m in (exact, fast)]
    assert torch.equal(hid[0], hid[1])


@torch.inference_mode()
def test_each_class_alone(models, tokens):
    """Each file attaches on its own (the gate can drop a class), and moves the logits less than all of them."""
    from engine.model.fast_qwen4exp import attach_decode_copies4, to_fast4
    exact, _, files, n_all, ref = models
    total, res = 0, {}
    for f in files:
        m = to_fast4(copy.deepcopy(ref))
        total += attach_decode_copies4(m, [f])
        res[os.path.basename(f)] = _compare(exact, m, tokens[:12])[:2]
        del m
        torch.cuda.empty_cache()
    print("\n[requant4] one class at a time over 12 steps (max|diff| / std, relative L2): "
          + ", ".join(f"{k[8:-12]} {a:.4f} / {b:.4f}" for k, (a, b) in res.items()))
    assert total == n_all
    assert all(a < TOL and b < REL for a, b in res.values()), res


def test_partial_stack_is_refused(models):
    from engine.model.fast_qwen4exp import _copy_of
    from tools.requant_qwen4exp import quantize_group
    w = torch.randn(64, 512, device="cuda")
    t, _ = quantize_group({"a": w, "b": w * 2}, "int6")
    t = {k: v.cuda() for k, v in t.items()}
    assert _copy_of(t, ["a", "b"]).out_features == 128
    assert _copy_of(t, ["c"]) is None
    with pytest.raises(RuntimeError, match="only some"):
        _copy_of(t, ["a", "c"])
    t["b.weight_scale_2"] = t["b.weight_scale_2"] * 2
    with pytest.raises(RuntimeError, match="one global scale"):
        _copy_of(t, ["a", "b"])


def test_quantize_group_formats():
    """The weight error of each format on Gaussian weights (the stacked parts of different scale share the global scale)."""
    from engine.weights.quantize import dequant_int
    from tools.requant_qwen4exp import quantize_group
    g = torch.Generator(device="cuda").manual_seed(3)
    ws = {"a": torch.randn(256, 1024, device="cuda", generator=g), "b": torch.randn(64, 1024, device="cuda", generator=g) * 0.1}
    errs = {f: quantize_group(ws, f)[1] for f in ("int6", "int5", "nvfp4", "int8")}
    print(f"\n[requant4] relative weight error: {errs}")
    assert errs["int6"] < 0.025 and errs["int5"] < 0.045 and errs["nvfp4"] < 0.1 and errs["int8"] < 0.01  # 0.022 / 0.041 / 0.085 / 0.008
    t, err = quantize_group(ws, "int6")
    assert float(t["a.weight_scale_2"]) == float(t["b.weight_scale_2"])
    from engine.weights.quantize import int_global_scale
    assert float(t["a.weight_scale_2"]) == pytest.approx(int_global_scale(float(ws["a"].abs().max()), 6))
    codes = t["a.qweight_lo"].cuda()  # the low nibbles: unpack and rebuild the codes with the high plane
    lo = torch.stack([codes & 15, codes >> 4], -1).view(256, 1024).int()
    hi_b = t["a.qweight_hi"].cuda().int()
    hi = torch.stack([(hi_b >> (2 * j)) & 3 for j in range(4)], -1).view(256, 1024)
    deq = dequant_int((lo | (hi << 4)).to(torch.uint8), t["a.weight_scale"].cuda(), float(t["a.weight_scale_2"]), 6)
    assert ((deq - ws["a"]).norm() / ws["a"].norm()).item() < 0.025


def test_tool_on_synthetic_checkpoint(tmp_path):
    """tools/requant_qwen4exp.py's reader and main on the tiny nvidia-layout checkpoint (names and BF16 tensors from the
    shards): the plan finds every class, and the files hold every module of the plan."""
    from safetensors import safe_open

    from engine.model.qwen4exp import Qwen4ExpForCausalLM
    from tests.qwen4exp_ckpt import write_tiny_checkpoint
    from tests.test_qwen4exp_small import init_tiny, tiny_cfg
    from tools.requant_qwen4exp import checkpoint_loader, main, plan
    ref = init_tiny(Qwen4ExpForCausalLM(tiny_cfg()).float())
    ck = str(tmp_path / "ckpt")
    write_tiny_checkpoint(ck, ref)
    names, load = checkpoint_loader(ck)
    groups = plan(names)
    assert all(groups[c] for c in groups), {c: len(g) for c, g in groups.items()}
    assert load(groups["gdn_in"][0][0] + ".weight").dtype == torch.bfloat16
    out = str(tmp_path / "prepared")
    assert main(["--model", ck, "--out", out, "--formats", "lm_head=nvfp4,gdn_in=int5,gdn_out=int6,hc=int6,shared=int6"]) == 0
    files = sorted(os.listdir(out))
    assert "decode4_lm_head_nvfp4.safetensors" in files and len(files) == 7
    with safe_open(os.path.join(out, "decode4_hc_int6.safetensors"), "pt") as f:
        keys = set(f.keys())
    assert all(m + ".qweight_lo" in keys for g in groups["hc"] for m in g)


@torch.inference_mode()
def test_lm_head_int6_fp32_logits(models, tokens):
    """An INT6 lm_head copy writes fp32 logits on skinny_int (its bf16 output is their rounding), and stays row-invariant."""
    from engine.model.fast import IntLinear
    from engine.model.fast_qwen4exp import _copy_of, attach_decode_copies4
    from tools.requant_qwen4exp import make_copies, write_copies
    exact, _, _, _, ref = models
    names, load = checkpoint_view(ref)
    out = os.path.join(os.path.dirname(models[2][0]), "lm6")
    f = os.path.join(out, write_copies(out, make_copies(names, load, {**EVERY, "lm_head": "int6"}, classes=("lm_head",)))[0])
    m = copy.deepcopy(exact)
    assert attach_decode_copies4(m, [f]) == 1 and isinstance(m.lm_head.dec, IntLinear)
    x = torch.randn(5, 2560, device="cuda").bfloat16()
    got = m.lm_head(x)
    want = torch.empty(5, V, device="cuda", dtype=torch.bfloat16)
    m.lm_head.dec.rows(x, None, want)
    assert got.dtype == torch.float32 and torch.equal(got.bfloat16(), want) and not torch.equal(got, want.float())  # not rounded to bf16
    assert torch.equal(m.lm_head(x[2:3]), got[2:3])
    from engine.weights.direct import read_tensors
    t = read_tensors(f, device="cuda")
    assert _copy_of(t, ["lm_head"]).out_features == V
    worst, worst_rel, flips, checked = _compare(exact, m, tokens[:12])
    assert worst_rel < 0.03 and flips == 0, (worst, worst_rel)

"""engine/weights/qwen4exp_loader.py on a synthetic tiny checkpoint in the nvidia layout (tests/qwen4exp_ckpt.py):
the stacked expert tensors equal the per-expert ones of the files, the loaded model's logits equal those of the tiny
reference with the same (dequantized) weights, the decoys are never read, and a corrupted hash buffer is refused.
Run: ./py -m pytest tests/test_qwen4exp_loader.py -q
"""
import os
import shutil

import pytest
import torch
from safetensors import safe_open

import engine.weights.qwen4exp_loader as QL
from engine.model.qwen4exp import Qwen4ExpForCausalLM
from tests.qwen4exp_ckpt import write_safetensors, write_tiny_checkpoint
from tests.test_qwen4exp_small import DT, EOS, init_tiny, tiny_cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU (direct reads)")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    ref = init_tiny(Qwen4ExpForCausalLM(tiny_cfg()).to(DT))
    path = str(tmp_path_factory.mktemp("qwen4exp_ckpt"))
    info = write_tiny_checkpoint(path, ref)
    return path, ref, info


@pytest.fixture(scope="module")
def loaded(ckpt):
    path, _, _ = ckpt
    read = []
    orig = QL.read_tensors

    def spy(f, names, device, into=None):
        read.extend(names)
        return orig(f, names, device, into=into)
    QL.read_tensors = spy
    try:
        m = QL.load_qwen4exp(path, "cuda", log=lambda s: None)
    finally:
        QL.read_tensors = orig
    return m, read


def test_stacked_experts_equal_the_files(ckpt, loaded):
    path, ref, info = ckpt
    m, _ = loaded
    I = ref.cfg.moe_intermediate_size
    for li, q in info["experts"].items():
        ex = m.layers[li].mlp.experts
        for a in ("w_gu", "w_dn", "gs", "in_scale"):
            assert torch.equal(getattr(ex, a).cpu(), getattr(q, a)), (li, a)
        for a in ("sf_gu", "sf_dn"):
            assert torch.equal(getattr(ex, a).cpu().view(torch.uint8), getattr(q, a).view(torch.uint8)), (li, a)
        assert ex.w_gu.is_cuda and ex.sf_gu.is_cuda and ex.gs.is_cuda
    li = info["split_layer"]  # its experts span two files: check one from each against the files themselves
    wm = QL.weight_map(path)
    ex = m.layers[li].mlp.experts
    for e in (0, ref.cfg.num_experts - 1):
        b = f"model.language_model.layers.{li}.mlp.experts.{e}."
        with safe_open(os.path.join(path, wm[b + "up_proj.weight"]), "pt") as f:
            assert torch.equal(ex.w_gu[e, I:].cpu(), f.get_tensor(b + "up_proj.weight"))
            assert torch.equal(ex.sf_dn[e].cpu().view(torch.uint8), f.get_tensor(b + "down_proj.weight_scale").view(torch.uint8))
            assert ex.gs[e, 2].item() == f.get_tensor(b + "down_proj.weight_scale_2").item()
    b = f"model.language_model.layers.{li}.mlp.experts."
    assert wm[b + "0.gate_proj.weight"] != wm[b + "3.gate_proj.weight"]


def test_decoys_are_not_read(loaded):
    m, read = loaded
    assert read and not [n for n in read if n.startswith(("mtp.", "model.visual.")) or ".ngram_embedding." in n]
    assert not [n for n in m.state_dict() if "mtp" in n or "visual" in n]
    emb = m.layers[1].ple.ple_embedding
    assert emb.ngram_embedding is None and emb.table.n_rows == emb.params.padded_rows


@torch.inference_mode()
def test_logits_equal_the_dequantized_reference(ckpt, loaded):
    """The reference: the tiny model's weights in BF16, the experts and the n-gram table dequantized, in the same
    module tree with dense experts and an in-memory table. Same ops, same device: bit-identical logits."""
    _, ref, info = ckpt
    m, _ = loaded
    cfg = ref.cfg
    dq = Qwen4ExpForCausalLM(cfg).to(torch.bfloat16)
    sd = {k: v for k, v in ref.state_dict().items() if ".experts." not in k}
    sd["layers.1.ple.ple_embedding.ngram_embedding.weight"] = (info["table"].float() * info["scale"].float()).to(torch.bfloat16)
    dq.load_state_dict(sd, strict=False)
    for li, q in info["experts"].items():
        for e in range(cfg.num_experts):
            gu, dn = q.get(e, torch.bfloat16)
            dq.layers[li].mlp.experts.gate_up_proj.data[e], dq.layers[li].mlp.experts.down_proj.data[e] = gu, dn
    dq = dq.cuda().eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 101, (1, 40), device="cuda")
    ids[0, [7, 8, 30]] = EOS
    a = m(ids, m.new_state(1, 40))
    b = dq(ids, dq.new_state(1, 40))
    assert a.dtype == torch.float32 and torch.equal(a, b)
    st = m.new_state(1, 40)  # and stepwise, through the disk table
    m(ids[:, :20], st)
    c = torch.cat([m(ids[:, t:t + 1], st) for t in range(20, 40)], 1)
    torch.testing.assert_close(c, b[:, 20:], rtol=1e-2, atol=1e-2)  # bf16: the chunked and the recurrent delta rule may round apart (0.0 measured)


def test_a_corrupted_multiplier_is_refused(ckpt, tmp_path):
    path, _, _ = ckpt
    bad = str(tmp_path / "bad")
    shutil.copytree(path, bad)
    name = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
    f = os.path.join(bad, QL.weight_map(bad)[name])
    with safe_open(f, "pt") as s:
        items = [(k, s.get_tensor(k)) for k in s.keys()]
    items = [(k, t + 2 if k == name else t) for k, t in items]
    write_safetensors(f, items)
    with pytest.raises(ValueError, match="layer_multipliers"):
        QL.load_qwen4exp(bad, "cuda", log=lambda s: None)


def test_verify_tool(ckpt, tmp_path):
    """tools/verify_qwen4exp.py passes the synthetic checkpoint, and reports a corrupted multiplier and a missing expert."""
    from tools.verify_qwen4exp import verify
    path, _, _ = ckpt
    logs = []
    assert verify(path, log=logs.append) == []
    assert any("gate/up global scales equal: True" in s for s in logs)
    bad = str(tmp_path / "bad")
    shutil.copytree(path, bad)
    wm = QL.weight_map(bad)
    mult = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
    gone = "model.language_model.layers.2.mlp.experts.1.up_proj.weight_scale"
    for f in {wm[mult], wm[gone]}:
        f = os.path.join(bad, f)
        with safe_open(f, "pt") as s:
            items = [(k, s.get_tensor(k)) for k in s.keys() if k != gone]
        write_safetensors(f, [(k, t * 3 if k == mult else t) for k, t in items])
    errors = verify(bad, log=lambda s: None)
    assert any(mult in e for e in errors) and any(gone in e for e in errors), errors

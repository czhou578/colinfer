"""The model families (engine/model/family.py): the dense family passes, the in-progress MoE family is refused with a
message that names the plan until its modules exist; the defaults and the memory figures of each family. Run:
./py -m pytest tests/test_family.py -q"""
import json
import os
import shutil
import sys
import types

import pytest
import torch

from engine.model import family as F
from engine.model.family import check_served, family_of, model_type
from engine.runtime import build

DATA = os.path.join(os.path.dirname(__file__), "data")
# the text config of Qwen3.8-27B: 48 GDN layers and 16 attention layers (one in four)
DENSE = dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=64, layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
             num_attention_heads=24, num_key_value_heads=4, head_dim=256, linear_num_key_heads=16, linear_num_value_heads=48,
             linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=248320,
             max_position_embeddings=262144)


def write_config(tmp_path, mt):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": mt}))
    return str(tmp_path)


def test_dense_family_passes(tmp_path):
    assert check_served(write_config(tmp_path, "qwen3_5")) == "qwen3_5"


def test_unknown_type_is_not_refused_here(tmp_path):
    assert model_type(write_config(tmp_path, "")) == ""
    assert check_served(str(tmp_path)) == ""


def test_moe_family_is_refused_with_the_plan(tmp_path):
    path = write_config(tmp_path, "qwen4_exp")
    with pytest.raises(NotImplementedError, match="qwen4_exp.*qwen4exp_plan.md"):
        check_served(path)
    with pytest.raises(NotImplementedError, match="not served yet"):
        build.build_engine(path, selftest=False)
    with pytest.raises(NotImplementedError, match="not served yet"):
        build.load_model(path)


def test_server_main_refuses_the_moe_family_before_the_memory_check(tmp_path):
    """engine.server's main checks the family right after it resolves the model, before its memory check (which reads the
    dense config and would fail with a KeyError on this family)."""
    from engine.server import api
    path = write_config(tmp_path, "qwen4_exp")
    with pytest.raises(SystemExit) as e:
        api.main(["--model", path])
    assert e.value.code == 2  # not NO_MEMORY_EXIT: the family check ran first


def dense_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": DENSE}))
    return str(tmp_path)


def moe_config(tmp_path):
    shutil.copy(os.path.join(DATA, "qwen4exp_config.json"), tmp_path / "config.json")
    return str(tmp_path)


def old_startup_bytes(path, slots, max_seq_len, spec):
    """build.startup_bytes before the families: the state of the slots (and of the drafter) from the dicts of FastState."""
    from engine.model.fast import FastState
    from engine.model.qwen35 import Qwen35Config
    from engine.spec.mtp import MtpState
    cfg = Qwen35Config.from_checkpoint(path)
    states = [FastState(cfg, slots, max_seq_len, "meta")] + ([MtpState(cfg, max_seq_len, "meta", batch=slots)] if spec == "mtp" else [])
    return build.ENGINE_FIXED_BYTES + sum(t.nbytes for s in states for d in (s.k, s.v, s.conv, s.rec) for t in d.values())


def test_the_dense_family_keeps_the_servers_defaults(tmp_path):
    fam = family_of(dense_config(tmp_path))
    assert fam.name == "qwen3_5" and fam.fixed_bytes == build.ENGINE_FIXED_BYTES
    assert fam.defaults == F.Defaults(slots=3, max_seq_len=524288, checkpoints=32, spec="mtp", vision=True)
    assert fam.supports == F.Supports(mtp=True, vision=True, yarn=True, int_copies=True)
    from engine.model.prefill import prefill, prepare_prefill
    assert fam.prefill is prefill and fam.prepare_prefill is prepare_prefill and fam.before_replay(None) is None and fam.feed_buffer(None) is None
    assert family_of(write_config(tmp_path, "")).name == "qwen3_5"  # no model_type: the dense family, as before the dispatch


@pytest.mark.parametrize("args", [(3, 524288, "mtp"), (3, 524288, "none"), (1, 8192, "none"), (2, 262144, "mtp")])
def test_the_startup_bytes_of_the_dense_family_are_unchanged(tmp_path, args):
    """The family's state_bytes plus its fixed bytes equal the figure of the memory check before the families, for the
    27B's shapes; None takes the defaults (3 x 524288, MTP)."""
    path = dense_config(tmp_path)
    assert build.startup_bytes(path, *args) == old_startup_bytes(path, *args)
    assert build.startup_bytes(path) == old_startup_bytes(path, 3, 524288, "mtp")
    try:
        from engine.weights.loader import MODEL, resolve
        real = resolve(MODEL)
    except Exception:
        return
    assert build.startup_bytes(real, *args) == old_startup_bytes(real, *args)


def stub_modules(monkeypatch, fast_names=("load_fast_qwen4exp", "FastState4")):
    """Stand-ins for engine/model/fast_qwen4exp.py and prefill_qwen4exp.py, with the names that family_of reads."""
    class FastState4:
        def __init__(self, cfg, batch, max_seq_len, device):
            self.k = torch.zeros(batch, 2, max_seq_len, 256, dtype=torch.float8_e4m3fn, device=device)
            self.ikb = torch.zeros(batch, 1, max_seq_len // 4, 128, dtype=torch.bfloat16, device=device)
            self.rec = torch.zeros(batch, 48, 128, 128, dtype=torch.float32, device=device)

        def kv_tensors(self):
            return [self.k, self.ikb]

        def snap_tensors(self):
            return [self.rec]
    names = dict(load_fast_qwen4exp=lambda *a: ("path", "model"), FastState4=FastState4)
    fast = types.ModuleType("engine.model.fast_qwen4exp")
    for n in fast_names:
        setattr(fast, n, names[n])
    pf = types.ModuleType("engine.model.prefill_qwen4exp")
    pf.prefill = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "engine.model.fast_qwen4exp", fast)
    monkeypatch.setitem(sys.modules, "engine.model.prefill_qwen4exp", pf)


def test_the_moe_family_is_refused_until_its_modules_have_the_names(tmp_path, monkeypatch):
    path = moe_config(tmp_path)
    monkeypatch.setitem(sys.modules, "engine.model.fast_qwen4exp", None)  # the module does not exist (as in this tree)
    with pytest.raises(NotImplementedError, match="qwen4_exp.*not served yet.*qwen4exp_plan.md"):
        family_of(path)
    stub_modules(monkeypatch, fast_names=("load_fast_qwen4exp",))  # the module exists, without FastState4
    with pytest.raises(NotImplementedError, match="not served yet"):
        check_served(path)


def test_the_moe_family_has_its_own_defaults(tmp_path, monkeypatch):
    """With the modules of the decode model and the prefill, the qwen4_exp family serves: 2 slots of 262144 tokens, 16
    checkpoints, no speculation, no images; its fixed bytes are the plan's (section 4); the build refuses MTP and a slot
    length past the native context before the memory check."""
    path = moe_config(tmp_path)
    stub_modules(monkeypatch)
    assert check_served(path) == "qwen4_exp"
    fam = family_of(path)
    assert fam.name == "qwen4_exp" and fam.load("p", "int", print, None, 262144) == ("path", "model")
    assert fam.defaults == F.Defaults(slots=2, max_seq_len=262144, checkpoints=16, spec="none", vision=False)
    assert fam.supports == F.Supports(mtp=False, vision=False, yarn=False, int_copies=False)
    assert fam.fixed_bytes == F.QWEN4EXP_FIXED_BYTES and 85.4e9 < fam.fixed_bytes < 85.5e9
    state = 2 * 262144 * (2 * 256 + 128 // 4 * 2) + 2 * 48 * 128 * 128 * 4  # k fp8, ikb bf16 at a row per 4 positions, rec fp32
    assert fam.state_bytes(path, 2, 262144, "none") == state and build.startup_bytes(path) == F.QWEN4EXP_FIXED_BYTES + state
    with pytest.raises(ValueError, match="no MTP drafter"):
        build.build_engine(path, spec="mtp", selftest=False)
    with pytest.raises(ValueError, match="native context 262144.*no YaRN"):
        build.build_engine(path, max_seq_len=524288, selftest=False)
    with pytest.raises(ValueError, match="no YaRN"):
        build.load_model(path, max_seq_len=524288)

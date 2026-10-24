"""The model families (engine/model/family.py): both families pass; the MoE family is refused with a message that names
the plan when its modules are missing, or while it is in NOT_SERVED_YET; the defaults, the memory figures and the n-gram
hooks of each family. Run: ./py -m pytest tests/test_family.py -q"""
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


def test_the_moe_family_is_served(tmp_path):
    assert check_served(moe_config(tmp_path)) == "qwen4_exp"


def test_moe_family_without_its_modules_is_refused_with_the_plan(tmp_path, monkeypatch):
    path = write_config(tmp_path, "qwen4_exp")
    monkeypatch.setitem(sys.modules, "engine.model.fast_qwen4exp", None)  # the module does not exist
    with pytest.raises(NotImplementedError, match="qwen4_exp.*qwen4exp_plan.md"):
        check_served(path)
    with pytest.raises(NotImplementedError, match="not served yet"):
        build.build_engine(path, selftest=False)
    with pytest.raises(NotImplementedError, match="not served yet"):
        build.load_model(path)


def test_a_family_in_not_served_yet_is_refused_with_its_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "NOT_SERVED_YET", {"qwen4_exp": "a reason"})
    with pytest.raises(NotImplementedError, match="not served yet: a reason"):
        check_served(moe_config(tmp_path))


def test_server_main_refuses_the_moe_family_before_the_memory_check(tmp_path, monkeypatch):
    """engine.server's main checks the family right after it resolves the model, before its memory check (which reads the
    dense config and would fail with a KeyError on this family)."""
    from engine.server import api
    path = write_config(tmp_path, "qwen4_exp")
    monkeypatch.setitem(sys.modules, "engine.model.fast_qwen4exp", None)
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


def moe_state_bytes(path, slots, max_seq_len):
    """The state of FastState4 from the config alone: per position, the fp8 k and v [2, 256] of each attention layer, its
    raw indexer key (bf16 128) and a quarter of its block key, and the rotary position (int32 x 3); per slot, the GDN
    conv windows (bf16 [10240, 3]) and recurrent states (fp32 [48, 128, 128]) and the PLE conv state (bf16 [10240, 9])."""
    from engine.model.qwen4exp import Qwen4ExpConfig
    cfg = Qwen4ExpConfig.from_checkpoint(path)
    n_attn = cfg.layer_types.count("full_attention")
    n_gdn = len(cfg.layer_types) - n_attn
    per_pos = n_attn * (2 * 2 * 256 + 128 * 2 + 128 * 2 // 4) + 3 * 4
    per_slot = n_gdn * (10240 * 3 * 2 + 48 * 128 * 128 * 4) + len(cfg.ple_layers) * 10240 * 9 * 2
    return slots * (max_seq_len * per_pos + per_slot)


def test_the_moe_family_has_its_own_defaults(tmp_path):
    """The qwen4_exp family serves: 2 slots of 262144 tokens, 16 checkpoints, no speculation, no images, prefill chunks
    of 4096 tokens with prev2; the memory check counts its fixed bytes, the state and the n-gram row cache; the build
    refuses MTP and a slot length past the native context before the memory check."""
    path = moe_config(tmp_path)
    fam = family_of(path)
    assert fam.name == "qwen4_exp"
    assert fam.defaults == F.Defaults(slots=2, max_seq_len=262144, checkpoints=16, spec="none", vision=False)
    assert fam.supports == F.Supports(mtp=False, vision=False, yarn=False, int_copies=False)
    assert fam.prefill_chunk == 4096 and fam.prefill_prev2
    assert fam.fixed_bytes == F.QWEN4EXP_FIXED_BYTES
    state = moe_state_bytes(path, 2, 262144)
    assert fam.state_bytes(path, 2, 262144, "none") == state
    assert fam.host_bytes(1.0) == 6_250_000 * 176 and fam.host_bytes(0) == 0
    assert build.startup_bytes(path) == F.QWEN4EXP_FIXED_BYTES + state + 6_250_000 * 176
    assert build.startup_bytes(path, ngram_cache_gb=0) == F.QWEN4EXP_FIXED_BYTES + state
    with pytest.raises(ValueError, match="no MTP drafter"):
        build.build_engine(path, spec="mtp", selftest=False)
    with pytest.raises(ValueError, match="native context 262144.*no YaRN"):
        build.build_engine(path, max_seq_len=524288, selftest=False)
    with pytest.raises(ValueError, match="no YaRN"):
        build.load_model(path, max_seq_len=524288)


def test_the_dense_family_has_no_host_cache(tmp_path):
    path = dense_config(tmp_path)
    assert family_of(path).host_bytes(1.0) == 0 and build.startup_bytes(path, ngram_cache_gb=50) == build.startup_bytes(path)
    assert not family_of(path).prefill_prev2 and family_of(path).prefill_chunk == 2048


def test_the_ngram_cache_flag_reaches_the_memory_check(tmp_path):
    """--ngram-cache-gb counts in the server's memory check, before the CUDA context: a cache that does not fit stops the
    start with the no-memory exit."""
    from engine.server import api
    with pytest.raises(SystemExit) as e:
        api.main(["--model", moe_config(tmp_path), "--ngram-cache-gb", "1000"])
    assert e.value.code == api.NO_MEMORY_EXIT


def test_prev2_of_pads_a_short_history_with_eos():
    assert F.prev2_of([], 7) == (7, 7) and F.prev2_of([5], 7) == (7, 5) and F.prev2_of([1, 2, 3], 7) == (2, 3)
    assert F.prev2_of(torch.tensor([4, 9]), 7) == (4, 9)


def test_the_moe_prefill_passes_the_chunk_and_prev2(tmp_path, monkeypatch):
    """The family's prefill is prefill_qwen4exp.prefill with chunks of 4096 and prev2 from the slot's history (None at
    a caller that passes none: the start of a sequence)."""
    import engine.model.prefill_qwen4exp as pq
    calls = []
    monkeypatch.setattr(pq, "prefill", lambda model, ids, state, **kw: calls.append(kw) or "logits")
    fam = family_of(moe_config(tmp_path))
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(eos_token_id=7))
    assert fam.prefill(model, "ids", "state", logits=False, prev2=[]) == "logits"
    fam.prefill(model, "ids", "state", prev2=[1, 2, 3])
    fam.prefill(model, "ids", "state", chunk=512)
    assert calls == [dict(chunk=4096, prev2=(7, 7), logits=False), dict(chunk=4096, prev2=(2, 3)), dict(chunk=512, prev2=None)]


class FakeFeeder:
    """PleFeeder.rows_for: the rows of token t are bytes t + 16 * head, after the context; records each call."""

    def __init__(self):
        self.calls = []

    def rows_for(self, tokens, prev2):
        self.calls.append((list(tokens), tuple(prev2)))
        T = len(tokens)
        rows = torch.zeros(T, 16, 160, dtype=torch.uint8)
        for i, t in enumerate(tokens):
            rows[i] = (torch.arange(16, dtype=torch.uint8) * 16 + t).view(16, 1)
        return rows


def test_the_moe_before_replay_fills_the_feed_buffer_of_its_row_count():
    """before_replay(slot, tokens) reads the rows of the T tokens after the slot's last two tokens, and writes them into
    the model's feed buffer (1, T): a plain step T = 1, a verify of k drafts T = k + 1."""
    bufs = {}
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(eos_token_id=7), feeder=FakeFeeder(),
                                  feed_buffer=lambda B, T: bufs.setdefault((B, T), torch.full((B, T, 16, 160), 255, dtype=torch.uint8)))
    hook = F.ngram_before_replay(model)
    slot = types.SimpleNamespace(tokens=[3])
    hook(slot, [5])
    slot.tokens = [3, 5, 9, 11]
    hook(slot, [12, 13, 14, 15])
    assert model.feeder.calls == [([5], (7, 3)), ([12, 13, 14, 15], (9, 11))]
    assert set(bufs) == {(1, 1), (1, 4)}
    assert torch.equal(bufs[(1, 1)][0], FakeFeeder().rows_for([5], ()))
    assert torch.equal(bufs[(1, 4)][0], FakeFeeder().rows_for([12, 13, 14, 15], ()))

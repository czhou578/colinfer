"""tools/train_drafter.py: the training-time-test unroll must compute exactly what chained drafting computes.

On a tiny random MTP head (CPU), depth 1 is the teacher-forced pass and depth s at row i must equal running the
reference decoder layer incrementally: catch-up rows 0..p with the target hidden states into a KV cache, then s-1
chain steps that feed the head its own normed output (p = i - s + 1).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))


def tiny():
    from engine.model.qwen35 import Qwen35Config
    cfg = Qwen35Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32, intermediate_size=96,
                       layer_types=["full_attention"], num_hidden_layers=1)
    g = torch.Generator().manual_seed(0)
    r = lambda *s, sc=0.2: (torch.randn(*s, generator=g) * sc)  # noqa: E731
    H, D, Hq, Hkv, I = 64, 32, 4, 2, 96
    L = "mtp.layers.0."
    t = {"mtp.fc.weight": r(H, 2 * H, sc=0.1), "mtp.pre_fc_norm_embedding.weight": r(H), "mtp.pre_fc_norm_hidden.weight": r(H),
         "mtp.norm.weight": r(H), L + "input_layernorm.weight": r(H), L + "post_attention_layernorm.weight": r(H),
         L + "self_attn.q_proj.weight": r(2 * Hq * D, H), L + "self_attn.k_proj.weight": r(Hkv * D, H),
         L + "self_attn.v_proj.weight": r(Hkv * D, H), L + "self_attn.o_proj.weight": r(H, Hq * D),
         L + "self_attn.q_norm.weight": r(D), L + "self_attn.k_norm.weight": r(D),
         L + "mlp.gate_proj.weight": r(I, H), L + "mlp.up_proj.weight": r(I, H), L + "mlp.down_proj.weight": r(H, I)}
    return cfg, t


def reference(cfg, t):
    """The MTP head built from the reference model's modules (engine/model/qwen35.py)."""
    from engine.model.qwen35 import DecoderLayer, RMSNorm
    layer = DecoderLayer(cfg, 0)
    sd = {k[len("mtp.layers.0."):]: v for k, v in t.items() if k.startswith("mtp.layers.0.")}
    layer.load_state_dict(sd)

    def norm(name):
        n = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        n.weight.data.copy_(t[name])
        return n
    return layer, norm("mtp.pre_fc_norm_embedding.weight"), norm("mtp.pre_fc_norm_hidden.weight"), norm("mtp.norm.weight")


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_unroll_matches_incremental_chain(depth):
    from engine.model.qwen35 import ModelState
    from train_drafter import Head
    cfg, t = tiny()
    head = Head(t, cfg)
    layer, pre_e, pre_h, fnorm = reference(cfg, t)
    T, Hd = 12, cfg.hidden_size
    g = torch.Generator().manual_seed(1)
    e, h = torch.randn(T, Hd, generator=g), torch.randn(T, Hd, generator=g)
    with torch.no_grad():
        outs = head.unroll(e, h, depth)
        d = cfg.rotary_dim
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))

        def cs(p0, n):
            pos = torch.arange(p0, p0 + n).float()
            emb = torch.cat([pos[:, None] * inv] * 2, -1)
            return emb.cos()[None], emb.sin()[None]

        def step(tok_e, hid, st):  # rows [n, H] at positions st.pos.. -> normed outputs
            x = torch.cat([pre_e(tok_e), pre_h(hid)], -1) @ t["mtp.fc.weight"].t()
            c, s = cs(st.pos, tok_e.shape[0])
            out = fnorm(layer(x[None], c, s, st)[0])
            st.pos += tok_e.shape[0]
            return out
        s_ = depth
        for i in range(s_ - 1, T):
            p = i - s_ + 1
            st = ModelState(cfg, 1, T + 8, "cpu", dtype=torch.float32)
            o = step(e[:p + 1], h[:p + 1], st)[-1:]  # catch-up rows 0..p; output of row p (depth 1)
            for k in range(1, s_):                    # chain: position p + k, token x_{p+k+1}, hidden = previous output
                o = step(e[p + k:p + k + 1], o, st)
            torch.testing.assert_close(outs[s_ - 1][i], o[0], atol=2e-4, rtol=2e-4)

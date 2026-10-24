"""Fast CPU tests for engine/model/qwen35.py and the dequant helpers, on a tiny random model:
- prefill continuation (T > 1 with existing state)
- the 64-token chunk boundary of the delta rule
- stepwise decode vs one-shot prefill
- agreement with Qwen3_5ForCausalLM of HF transformers at tiny scale, also with an image's multimodal positions and
  input embeddings
Run: uv run pytest tests/test_model_small.py -q
"""
import dataclasses

import pytest
import torch

from engine.model.qwen35 import Qwen35Config, Qwen35ForCausalLM
from engine.weights.loader import E2M1_LUT, dequant_nvfp4

DT = torch.float32


def tiny_cfg():
    return Qwen35Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                        num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
                        rope_theta=1e4, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                        linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=101,
                        mrope_section=[1, 1, 2], mrope_interleaved=True)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = Qwen35ForCausalLM(tiny_cfg()).to(DT)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"):
                p.uniform_(-1, 1)
            elif "norm" in n:
                p.normal_(0, 0.1)
            else:
                p.normal_(0, 0.08)
    return m.eval()


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(1)
    return torch.randint(0, 101, (1, 70))  # 70 crosses the 64-token delta-rule chunk boundary


@torch.inference_mode()
def test_stepwise_decode_matches_prefill(model, ids):
    full = model(ids, model.new_state(1, 80))
    st = model.new_state(1, 80)
    steps = [model(ids[:, :1], st)]
    for t in range(1, ids.shape[1]):
        steps.append(model(ids[:, t:t + 1], st))
    torch.testing.assert_close(torch.cat(steps, 1), full, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
@pytest.mark.parametrize("split", [1, 3, 40, 64, 69])
def test_split_prefill_matches_one_shot(model, ids, split):
    full = model(ids, model.new_state(1, 80))
    st = model.new_state(1, 80)
    a = model(ids[:, :split], st)
    b = model(ids[:, split:], st)
    torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-4, atol=1e-4)


def hf_tiny(model, sd=None, rope=None):
    """transformers' Qwen3_5ForCausalLM with the weights of the tiny reference model (sd: other weights for it; rope:
    further rope_parameters, e.g. YaRN)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from tests.hf_fallback import force_hf_torch_fallbacks
    force_hf_torch_fallbacks()
    c = model.cfg
    hcfg = Qwen3_5TextConfig(hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
                             num_hidden_layers=c.num_hidden_layers, layer_types=c.layer_types,
                             num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
                             head_dim=c.head_dim, linear_num_key_heads=c.linear_num_key_heads,
                             linear_num_value_heads=c.linear_num_value_heads, linear_key_head_dim=c.linear_key_head_dim,
                             linear_value_head_dim=c.linear_value_head_dim, linear_conv_kernel_dim=c.linear_conv_kernel_dim,
                             rms_norm_eps=c.rms_norm_eps, vocab_size=c.vocab_size, tie_word_embeddings=False,
                             rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta,
                                              "partial_rotary_factor": c.partial_rotary_factor,
                                              "mrope_section": c.mrope_section, "mrope_interleaved": True} | (rope or {}))
    hcfg._attn_implementation = "sdpa"
    hf = Qwen3_5ForCausalLM(hcfg).to(DT).eval()
    sd = {("lm_head.weight" if k == "lm_head.weight" else "model." + k): v for k, v in (sd or model.state_dict()).items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    return hf


@torch.inference_mode()
def test_matches_hf_tiny(model, ids):
    ref = hf_tiny(model)(input_ids=ids, use_cache=False).logits
    ours = model(ids, None)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_matches_hf_tiny_with_an_image(model, ids):
    """Rows 10 .. 33 are an image (a 4 x 6 token grid): input embeddings of their own, and the (t, h, w) positions of
    engine/model/vision.py, as transformers' Qwen3.5 takes them."""
    from engine.model.vision import Image, rope_positions
    im = Image(torch.empty(0), (1, 8, 12), key=0, start=10)
    torch.manual_seed(2)
    emb = model.embed_tokens(ids).clone()
    emb[0, 10:10 + im.n] = torch.randn(im.n, model.cfg.hidden_size) * 0.5
    pos = torch.from_numpy(rope_positions([im], 0, ids.shape[1])).long()
    assert pos[40, 0] == 40 + max(im.hw) - im.n  # the text after the image runs behind its index
    ref = hf_tiny(model)(inputs_embeds=emb, position_ids=pos.T[:, None, :], use_cache=False).logits
    ours = model(ids, None, positions=pos, embeds=emb)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(ours, model(ids, None, embeds=emb), atol=1e-3)  # the positions matter


# ------------------------------------------------------------------------------------------------------------ YaRN
def yarn_cfg(**kw):
    """A config with YaRN x2 over a short native context, so that the tiny rotary dims (8) span keep, ramp and divide."""
    return dataclasses.replace(tiny_cfg(), **({"max_position_embeddings": 1024, "yarn_factor": 2.0} | kw))


@pytest.mark.parametrize("cfg", [yarn_cfg(), yarn_cfg(yarn_factor=4.0),
                                 Qwen35Config(head_dim=256, partial_rotary_factor=0.25, rope_theta=1e7, yarn_factor=2.0)])  # Qwen3.8's dims
def test_yarn_frequencies_match_transformers(cfg):
    import types

    from transformers.modeling_rope_utils import _compute_yarn_parameters

    from engine.model.qwen35 import rope_inv_freq, yarn_attention_factor
    rp = {"rope_type": "yarn", "rope_theta": cfg.rope_theta, "partial_rotary_factor": cfg.partial_rotary_factor, "factor": cfg.yarn_factor,
          "original_max_position_embeddings": cfg.max_position_embeddings}
    hc = types.SimpleNamespace(rope_parameters=rp, head_dim=cfg.head_dim, hidden_size=cfg.hidden_size, num_attention_heads=cfg.num_attention_heads,
                               max_position_embeddings=cfg.max_position_embeddings * cfg.yarn_factor, standardize_rope_params=lambda: None)
    ref, scale = _compute_yarn_parameters(hc)
    ours = rope_inv_freq(cfg)
    plain = rope_inv_freq(dataclasses.replace(cfg, yarn_factor=1.0))
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=0)
    assert yarn_attention_factor(cfg) == pytest.approx(scale)
    assert ours.shape == (cfg.rotary_dim // 2,) and torch.all(ours <= plain) and ours[0] == plain[0]  # the fastest frequency is kept


@torch.inference_mode()
def test_yarn_model_matches_hf_yarn(ids):
    """apply_yarn (the YaRN frequencies, the attention factor folded into the q / k norms) against transformers' yarn
    rope type (the factor on cos and sin) with the same raw weights."""
    from engine.model.qwen35 import apply_yarn, yarn_attention_factor
    torch.manual_seed(0)
    m = Qwen35ForCausalLM(yarn_cfg(yarn_factor=1.0)).to(DT)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.normal_(0, 0.1 if "norm" in n else 0.08)
    m.eval()
    raw = {k: v.clone() for k, v in m.state_dict().items()}
    plain = m(ids, None).clone()
    apply_yarn(m, 2.0)
    assert m.cfg.yarn_factor == 2.0 and yarn_attention_factor(m.cfg) == pytest.approx(0.1 * torch.log(torch.tensor(2.0)).item() + 1)
    a = m.layers[3].self_attn
    r = m.cfg.rotary_dim
    torch.testing.assert_close(1 + a.q_norm.weight[:r], yarn_attention_factor(m.cfg) * (1 + raw["layers.3.self_attn.q_norm.weight"][:r]))
    torch.testing.assert_close(a.k_norm.weight[r:], raw["layers.3.self_attn.k_norm.weight"][r:])  # the pass-through dims stay
    hf = hf_tiny(m, sd=raw, rope={"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 1024})
    ref = hf(input_ids=ids, use_cache=False).logits
    ours = m(ids, None)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(ours, plain, atol=1e-3)  # the scaling changes the outputs
    with pytest.raises(ValueError):
        apply_yarn(m, 2.0)  # not twice


@pytest.mark.parametrize("rope", [{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024},
                                  {"rope_type": "linear", "factor": 2.0}])
def test_config_refuses_a_checkpoint_with_scaled_rope(tmp_path, rope):
    """The engine adds YaRN itself past the native context (apply_yarn). A checkpoint with scaled rope is refused."""
    import json
    c = tiny_cfg()
    t = {"hidden_size": c.hidden_size, "intermediate_size": c.intermediate_size, "num_hidden_layers": c.num_hidden_layers,
         "layer_types": c.layer_types, "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_key_value_heads,
         "head_dim": c.head_dim, "linear_num_key_heads": c.linear_num_key_heads, "linear_num_value_heads": c.linear_num_value_heads,
         "linear_key_head_dim": c.linear_key_head_dim, "linear_value_head_dim": c.linear_value_head_dim,
         "linear_conv_kernel_dim": c.linear_conv_kernel_dim, "rms_norm_eps": c.rms_norm_eps, "vocab_size": c.vocab_size,
         "max_position_embeddings": 4096, "rope_parameters": {"rope_type": "default", "rope_theta": 1e4, "partial_rotary_factor": 0.25,
                                                               "mrope_section": [1, 1, 2]}}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": t}))
    got = Qwen35Config.from_checkpoint(str(tmp_path))
    assert (got.yarn_factor, got.max_position_embeddings, got.rope_theta) == (1.0, 4096, 1e4)
    t["rope_parameters"] |= rope
    (tmp_path / "config.json").write_text(json.dumps({"text_config": t}))
    with pytest.raises(NotImplementedError, match=rope["rope_type"]):
        Qwen35Config.from_checkpoint(str(tmp_path))


def test_nvfp4_dequant_nibble_order_and_scales():
    codes = torch.arange(16, dtype=torch.uint8)                    # one row, K = 32
    packed = (codes | (codes.flip(0) << 4)).view(1, 16)          # low nibble = 0..15, high = 15..0
    sf = torch.tensor([[2.0, 0.5]]).to(torch.float8_e4m3fn)      # block 0 (k 0..15) x2, block 1 (k 16..31) x0.5
    out = dequant_nvfp4(packed, sf, torch.tensor(3.0), torch.float32)
    expect = torch.stack([E2M1_LUT[codes.long()], E2M1_LUT[codes.flip(0).long()]], -1).reshape(1, 32)
    expect = expect * torch.tensor([2.0] * 16 + [0.5] * 16) * 3.0
    torch.testing.assert_close(out, expect)

"""Fast CPU tests for engine/model/qwen35.py and the dequant helpers, on a tiny random model.

Covers paths the 27B parity run does not: prefill continuation (T > 1 with existing state), the
64-token chunk boundary of the delta rule, stepwise decode vs one-shot prefill, and agreement with
HF transformers' Qwen3_5ForCausalLM at tiny scale. Run: uv run pytest tests/test_model_small.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.model.qwen35 import Qwen35Config, Qwen35ForCausalLM  # noqa: E402
from engine.weights.loader import E2M1_LUT, dequant_fp8_block, dequant_nvfp4  # noqa: E402

DT = torch.float32


def tiny_cfg():
    return Qwen35Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                        num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
                        rope_theta=1e4, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                        linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=101)


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


@torch.inference_mode()
def test_matches_hf_tiny(model, ids):
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
                                              "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    hcfg._attn_implementation = "sdpa"
    hf = Qwen3_5ForCausalLM(hcfg).to(DT).eval()
    sd = {("lm_head.weight" if k == "lm_head.weight" else "model." + k): v for k, v in model.state_dict().items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    ref = hf(input_ids=ids, use_cache=False).logits
    ours = model(ids, None)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)


def test_nvfp4_dequant_nibble_order_and_scales():
    codes = torch.arange(16, dtype=torch.uint8)                    # one row, K = 32
    packed = (codes | (codes.flip(0) << 4)).view(1, 16)          # low nibble = 0..15, high = 15..0
    sf = torch.tensor([[2.0, 0.5]]).to(torch.float8_e4m3fn)      # block 0 (k 0..15) x2, block 1 (k 16..31) x0.5
    out = dequant_nvfp4(packed, sf, torch.tensor(3.0), torch.float32)
    expect = torch.stack([E2M1_LUT[codes.long()], E2M1_LUT[codes.flip(0).long()]], -1).reshape(1, 32)
    expect = expect * torch.tensor([2.0] * 16 + [0.5] * 16) * 3.0
    torch.testing.assert_close(out, expect)


def test_fp8_block_dequant_ragged_dims():
    w = torch.randn(200, 300)
    s = torch.rand(2, 3) + 0.5
    w8 = (w / s.repeat_interleave(128, 0)[:200].repeat_interleave(128, 1)[:, :300]).to(torch.float8_e4m3fn)
    out = dequant_fp8_block(w8, s, 128, torch.float32)
    rel = ((out - w).norm() / w.norm()).item()
    assert out.shape == (200, 300) and rel < 0.05, rel


# ---- sampling (engine/runtime/generate.py) ----
from engine.runtime.generate import generate, sample  # noqa: E402


def test_sample_greedy_and_topk1_agree():
    torch.manual_seed(2)
    logits = torch.randn(4, 101)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample(logits, 0.0, 0, 1.0, None), logits.argmax(-1))
    assert torch.equal(sample(logits, 0.7, 1, 1.0, g), logits.argmax(-1))


def test_sample_top_p_support():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    g = torch.Generator().manual_seed(0)
    draws = {int(sample(logits, 1.0, 0, 0.7, g)) for _ in range(400)}
    assert draws == {0, 1}, draws          # 0.5 + 0.3 covers 0.7; tokens 2 and 3 are cut
    draws = {int(sample(logits, 1.0, 0, 0.85, g)) for _ in range(400)}
    assert draws == {0, 1, 2}, draws


def test_sample_temperature_distribution():
    logits = torch.log(torch.tensor([[0.6, 0.4]]))
    g = torch.Generator().manual_seed(0)
    n = 4000
    hits = sum(int(sample(logits, 1.0, 0, 1.0, g)) == 0 for _ in range(n))
    assert abs(hits / n - 0.6) < 0.03, hits / n


@torch.inference_mode()
def test_generate_seeded_sampling_reproducible_and_eos(model, ids):
    a, _ = generate(model, ids[:, :10], 20, temperature=0.8, top_k=20, top_p=0.95, seed=123)
    b, _ = generate(model, ids[:, :10], 20, temperature=0.8, top_k=20, top_p=0.95, seed=123)
    assert a == b and len(a) == 20
    greedy, _ = generate(model, ids[:, :10], 20)
    stop = greedy[5]
    cut, _ = generate(model, ids[:, :10], 20, eos_ids=[stop])
    assert cut == greedy[: greedy.index(stop) + 1]

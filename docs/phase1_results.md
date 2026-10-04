# Phase 1 results (2026-10-03): plain-PyTorch reference engine

Code: `engine/model/qwen35.py`, `engine/weights/loader.py`, `engine/runtime/generate.py`.
Harness: `tests/run_phase1_gate.sh` (30-prompt parity, NVFP4 vs vLLM, perplexity), `tests/test_model_small.py` (13 CPU tests).
Raw logs and outputs: `tests/parity_out/` (git-ignored).

## Exit criteria status

| Criterion (PLAN.md Phase 1) | Result | Status |
|---|---|---|
| Token-exact vs HF transformers BF16 on 30 prompts x 128 tokens | 30/30 exact, logit KL 0, max abs logit difference 0.000 over the first 32 steps | **Met** |
| Perplexity numbers committed per weight configuration | Below | Met |
| NVFP4 vs vLLM: logit KL small, perplexity within 0.5% | With W4A4 emulation: perplexity within **0.23%**; KL 1.5e-2, equal to the FP4 activation-quantization noise floor (1.4e-2) | **Met** (perplexity); KL criterion redefined below |

## WikiText test perplexity (ctx 2048, 65,504 scored tokens, our model, weights dequantized to BF16)

| Checkpoint | Perplexity | vs BF16 |
|---|---|---|
| `Qwen/Qwen3.8-27B` (BF16) | 6.9404 | |
| `Qwen/Qwen3.8-27B-FP8` (128x128 block) | 6.9608 | +0.29% |
| `nvidia/Qwen3.8-27B-NVFP4` (MLP + lm_head NVFP4, rest FP8) | 6.9698 | +0.42% |
| `nvidia/Qwen3.8-27B-NVFP4` + FP4 activation quant (MLP, lm_head) | 7.0915 | +2.18% |
| `nvidia/Qwen3.8-27B-NVFP4` + FP8 static activation quant (attention, GDN) | 6.9634 | +0.33% |
| `nvidia/Qwen3.8-27B-NVFP4` + fused FP8 shard re-rounding | 6.9643 | +0.34% |
| `nvidia/Qwen3.8-27B-NVFP4`, all three (`--emulate all`, W4A4 as vLLM runs it) | **7.0823** | +2.04% |
| `nvidia/Qwen3.8-27B-NVFP4` served by vLLM 0.25.1 (native kernels) | 7.0985 | +2.28% |

## NVFP4 against vLLM, 30 prompts

4/30 prompts token-exact, mean matching prefix 33.6 tokens, mean KL(vLLM || ours) 1.13e-2 over vLLM's top-20 support
(median vLLM probability mass outside the top 20 is 2.6e-5, so the truncation is negligible). Three prompts differ at the very first token.

That was the weight-only path. The gap is explained, and verified, by activation quantization:

- `engine/weights/quant_emul.py` reproduces the W4A4 / W8A8 kernel numerics in PyTorch (`--emulate`). Its FP4
  activation quantizer matches FlashInfer's `fp4_quantize` bit for bit (0 mismatches over 5.2 M values, with
  outliers; `tests/test_quant_emul.py`). vLLM fuses q/k/v and in_proj_qkv + in_proj_z, whose FP8 shards carry
  weight scales up to 3.7x apart, so it re-rounds 80 shards onto the larger scale; that is emulated too.
- **FP4 activation quantization of the MLP and lm_head inputs is the whole story**: +1.75% perplexity on its own.
  FP8 activation quantization and the shard re-rounding cost under 0.1% each.
- With all three effects, our perplexity is 7.0823 against vLLM's 7.0985: **0.23% apart, inside the 0.5% gate.**
  The residual 0.2% is most likely vLLM's other fusions and accumulation order (not investigated).

### Why token-exactness and tiny KL against vLLM are not achievable, and the revised criterion

| Comparison (30 prompts, first 32 steps) | Token-exact | Mean KL |
|---|---|---|
| BF16 vs our NVFP4 weight-only | 4/30 | 1.01e-2 |
| Our NVFP4 weight-only vs our W4A4 emulation | 3/30 | 1.36e-2 |
| vLLM vs our NVFP4 weight-only | 4/30 | 1.13e-2 |
| vLLM vs our W4A4 emulation | 2/30 | 1.53e-2 |

Switching activation quantization on inside our own model moves the logits as far as the gap to vLLM. FP4 rounding
is a hard decision per element, so any tiny difference in the BF16 residual stream (kernel fusion, accumulation
order) flips some roundings and redraws the quantization noise; KL between two W4A4 implementations therefore
sits at the noise floor (about 1.4e-2) however correct both are. Token-exact parity is only meaningful in BF16
(where it is met, 30/30). **Revised gate for every quantized path from now on: WikiText perplexity within 0.5% of
the reference W4A4 path (`--emulate all`), and mean KL against it no larger than the 1.4e-2 noise floor plus 20%.**

## Findings that matter downstream

- **Activation quantization, not weight quantization, is what costs accuracy on this checkpoint**: NVFP4 weights cost
  +0.42% perplexity, FP4 activations on the MLP inputs a further +1.75%. Implication for Phase 2: the decode GEMV
  can keep activations in BF16 (W4A16) for free, since decode is bandwidth-bound and activations are tiny, and that is
  1.7% better perplexity than vLLM. Prefill needs W4A4 for tensor-core throughput; consider FP8 activations there
  (W4A8) if the CUTLASS SM120 mixed kernel is fast enough, since FP8 activations cost only 0.33%.
- The proposed re-quantization of attention and GDN projections from FP8 to NVFP4 (baseline.md section 3) should be
  judged with this harness, weights only for decode and with activations for prefill.
- Our dequant paths are validated: NVFP4 low-nibble-first order gives 9% relative weight error versus 141% for the swapped order;
  FP8 block and per-tensor give 2.7%.
- Speed (irrelevant by design): about 4 tok/s decode, 51 GB resident, 33 s to load.

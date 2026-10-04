# Phase 1 results (2026-10-03): plain-PyTorch reference engine

Code: `engine/model/qwen35.py`, `engine/weights/loader.py`, `engine/runtime/generate.py`.
Harness: `tests/run_phase1_gate.sh` (30-prompt parity, NVFP4 vs vLLM, perplexity), `tests/test_model_small.py` (13 CPU tests).
Raw logs and outputs: `tests/parity_out/` (git-ignored).

## Exit criteria status

| Criterion (PLAN.md Phase 1) | Result | Status |
|---|---|---|
| Token-exact vs HF transformers BF16 on 30 prompts x 128 tokens | 30/30 exact, logit KL 0, max abs logit difference 0.000 over the first 32 steps | **Met** |
| Perplexity numbers committed per weight configuration | Below | Met |
| NVFP4 vs vLLM: logit KL small, perplexity within 0.5% | KL 1.1e-2 (top-20 support); perplexity differs by 1.85% | **Not met** |

## WikiText test perplexity (ctx 2048, 65,504 scored tokens, our model, weights dequantized to BF16)

| Checkpoint | Perplexity | vs BF16 |
|---|---|---|
| `Qwen/Qwen3.8-27B` (BF16) | 6.9404 | |
| `Qwen/Qwen3.8-27B-FP8` (128x128 block) | 6.9608 | +0.29% |
| `nvidia/Qwen3.8-27B-NVFP4` (MLP + lm_head NVFP4, rest FP8) | 6.9698 | +0.42% |
| `nvidia/Qwen3.8-27B-NVFP4` served by vLLM 0.25.1 (native kernels) | 7.0985 | +2.28% |

## NVFP4 against vLLM, 30 prompts

4/30 prompts token-exact, mean matching prefix 33.6 tokens, mean KL(vLLM || ours) 1.13e-2 over vLLM's top-20 support
(median vLLM probability mass outside the top 20 is 2.6e-5, so the truncation is negligible). Three prompts differ at the very first token.

Likely cause, **not yet verified**: our loader dequantizes weights to BF16 and runs BF16 activations (weight-only), while the
checkpoint is W4A4 and vLLM's kernels also quantize activations (FP4 with block-16 scales for the MLPs, FP8 with the static
`input_scale` for attention and GDN). That extra rounding would raise vLLM's perplexity and explain why ours sits near BF16
(+0.42%) while vLLM sits +2.3% above it. Test: add activation fake-quantization to the NVFP4 path and see whether
perplexity moves to about 7.10 and KL collapses. This is needed anyway, because the Phase 2 kernels are W4A4.

## Findings that matter downstream

- Weight-only quantization cost on this model is small: FP8 +0.29%, mixed NVFP4/FP8 +0.42% perplexity. The proposed
  re-quantization of attention and GDN projections from FP8 to NVFP4 (baseline.md section 3) should be judged against
  these numbers, but with activation quantization included, since that is what the fast path will do.
- Our dequant paths are validated: NVFP4 low-nibble-first order gives 9% relative weight error versus 141% for the swapped order;
  FP8 block and per-tensor give 2.7%.
- Speed (irrelevant by design): about 4 tok/s decode, 51 GB resident, 33 s to load.

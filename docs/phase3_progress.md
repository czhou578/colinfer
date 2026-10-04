# Phase 3: prefill at the compute roofline (2026-10-03)

Checkpoint `nvidia/Qwen3.8-27B-NVFP4`, FP8 KV cache, chunk 2048, `bench/prefill_bench.py --kv-fp8`.
TTFT = whole-prompt prefill + first token.

## Results

| Prompt | TTFT | tok/s | vLLM 0.25.1 (same checkpoint) |
|---|---|---|---|
| 512 | 0.18 s | 2,925 | 2,150 |
| 2k | **0.62 s** | **3,303** (3,275 with FP8 KV) | 2,533 (0.81 s) |
| 8k | 2.59 s | 3,167 | 2,040 (3.9 s) |
| 32k | **12.5 s** | 2,616 | 1,802 (18.0 s) |
| 64k | 30.7 s | 2,136 | 1,515 (43.3 s) |
| 128k | 84.5 s | 1,550 | |
| 250k | < 284 s (measured while the GPU was shared with the test suite) | | |

| Exit criterion | Target | Result |
|---|---|---|
| Plan: tok/s at 2k-8k | >= 2,500 | 3,167-3,303: **met** |
| Plan: TTFT(2k) | <= 0.8 s | 0.62 s: **met** |
| Plan: 32k prompt | <= 16 s | 12.5 s: **met** |
| Frozen (baseline.md 5): tok/s at 2k-8k | >= 3,500 | 3,167-3,303: 6-10% short |
| Frozen: TTFT(2k) | <= 0.6 s | 0.62 s: 3% short |
| Frozen: 32k | <= 12 s (2,700 tok/s) | 12.5 s: 4% short |
| Quality: perplexity vs the matching W4A4 reference | within 0.5% | 7.0852 vs 7.0875: **0.03%** |
| Long context: FP8 KV retrieval at depth | | passkey 9/9 at 16k / 63k / 126k (depths 0.1 / 0.5 / 0.9), 1/1 at 250k (depth 0.5) |

## Design (engine/model/prefill.py)

Per 2048-token chunk and layer:
- input RMSNorm fused with FP8 quantization (and BF16 output for the GDN b/a projection) -> one cuBLASLt
  FP8 GEMM for q/k/v or GDN qkv/z with per-row weight scales (checkpoint scales kept exactly).
- attention: fused q/k norm + RoPE + KV write prologue (reads the stacked GEMM output in place),
  FlashInfer FA2 causal prefill over the slot's cache (an FP8 cache prefix is cast to BF16 first: FlashInfer's
  FP8-KV path runs ~48 vs ~80 TFLOPS), output gate fused with FP8 quantization, o_proj FP8 GEMM.
- GDN: causal conv (token-major, continues the conv state) writing q / k / v contiguous, FLA
  `chunk_gated_delta_rule` (16 key heads x 48 value heads directly, continuing the recurrent state),
  gated RMSNorm, FP8 out_proj.
- MLP: residual add + RMSNorm + NVFP4 quantization in one kernel, CUTLASS SM120 NVFP4 GEMM for gate|up
  (one stacked GEMM, ~330 TFLOPS), fused SiLU*up -> NVFP4, down GEMM with the residual in its epilogue.

Time for a 2048-token chunk: 630 ms, of which 62% is GEMMs (NVFP4 225 ms, FP8 165 ms), 7% SiLU*up quant
(at bandwidth roofline), ~14% FLA, the rest bandwidth-bound fused kernels.

## Quality trade-off: W4A4 prefill

Prefill uses FP4 activations on the MLPs (as vLLM and SGLang do). That costs +1.65% WikiText perplexity
against weight-only NVFP4 (6.970 -> 7.085); decode keeps BF16 activations. A W4A16 prefill (dequantized
BF16 GEMMs) would avoid it at roughly 2x the MLP time; W4A8 needs a mixed-input block-scaled kernel and is
a Phase 6 candidate.

## Multi-slot engine and prefix checkpoints (engine/runtime/engine.py)

- Up to 3 slots in one batched state; one decode CUDA graph per batch width; idle / prefilling slots are
  masked (`state.active`) so a decode step leaves their KV and GDN state untouched.
- Each engine step: at most one 2048-token prefill chunk, then one decode step for all decoding slots.
- Checkpoint ring (32 x ~154 MB of GDN state): at chunk boundaries, end of prompt, end of generation.
  New requests restore the longest valid checkpoint that prefixes their prompt.
- `tests/engine_check.py`: a greedy request is token-identical alone and alongside a 5000-token chunked
  prefill plus a sampled request; a 7,980-token second chat turn reuses 7,958 tokens: **TTFT 0.115 s vs
  2.70 s from scratch**, identical output.

## Remaining for Phase 3

- The frozen 3,500 tok/s target: candidates are a CUTLASS epilogue producing SiLU*up NVFP4 directly
  (removes the 142 MB / layer intermediate round trip, ~45 ms / chunk) and tuning or porting the FLA
  chunk kernels (~90 ms / chunk, 14%).
- Tool-calling quality gate: needs the OpenAI-compatible server (Phase 5).

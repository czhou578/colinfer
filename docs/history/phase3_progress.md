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
| 250k | < 284 s (measured while the test suite shared the GPU) | | |

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

For each 2048-token chunk and each layer:

- **Projections:** one kernel fuses the input RMSNorm with the FP8 quantization (and a BF16 output for the GDN b/a
  projection).
  Then one cuBLASLt FP8 GEMM runs for q/k/v or GDN qkv/z, with per-row weight scales. The checkpoint scales stay exact.
- **Attention:** a fused prologue does the q/k norm, the RoPE and the KV write, and reads the stacked GEMM output in
  place. Then FlashInfer FA2 runs a causal prefill over the cache of the slot. An FP8 cache prefix is cast to BF16 first,
  because the FP8-KV path of FlashInfer runs at ~48 TFLOPS against ~80. One kernel applies the output gate and the FP8
  quantization, and the o_proj FP8 GEMM follows.
- **GDN:** a causal conv (token-major, continues the conv state) writes q / k / v contiguously. Then FLA
  `chunk_gated_delta_rule` runs directly on 16 key heads x 48 value heads and continues the recurrent state. A gated
  RMSNorm and the FP8 out_proj follow.
- **MLP:** one kernel does the residual add, the RMSNorm and the NVFP4 quantization. A CUTLASS SM120 NVFP4 GEMM runs
  gate|up as one stacked GEMM (~330 TFLOPS). A fused SiLU*up kernel gives NVFP4, and the down GEMM adds the residual in
  its epilogue.

A 2048-token chunk takes 630 ms:

- 62% GEMMs (NVFP4 225 ms, FP8 165 ms)
- 7% SiLU*up quant (at the bandwidth roofline)
- ~14% FLA
- the rest in bandwidth-bound fused kernels

## Quality trade-off: W4A4 prefill

Prefill uses FP4 activations on the MLPs, as vLLM and SGLang do. This costs +1.65% WikiText perplexity against
weight-only NVFP4 (6.970 -> 7.085). Decode keeps BF16 activations. A W4A16 prefill (dequantized BF16 GEMMs) would avoid
this cost, at roughly 2x the MLP time. W4A8 needs a mixed-input block-scaled kernel. It is a Phase 6 candidate.

## Multi-slot engine and prefix checkpoints (engine/runtime/engine.py)

- One batched state holds up to 3 slots. Each batch width has one decode CUDA graph. The engine masks idle and
  prefilling slots (`state.active`), so a decode step does not change their KV and GDN state.
- Each engine step runs at most one 2048-token prefill chunk, then one decode step for all decoding slots.
- A checkpoint ring holds 32 x ~154 MB of GDN state. The engine takes a checkpoint at chunk boundaries, at the end of a
  prompt and at the end of a generation. A new request restores the longest valid checkpoint that is a prefix of its
  prompt.
- `tests/engine_check.py`: a greedy request is token-identical alone, and next to a 5000-token chunked prefill plus a
  sampled request. A 7,980-token second chat turn reuses 7,958 tokens: **TTFT 0.115 s vs 2.70 s from scratch**, with
  identical output.

## Startup self-test (engine/selftest.py)

At engine start, each matmul path runs a small random problem against an fp32 reference (~0.6 s). The paths are:

- the NVFP4 GEMV and SwiGLU GEMV
- the FP8 row-scaled GEMV and the BF16 GEMV
- the CUTLASS NVFP4 GEMM (both tiles) and cuBLASLt FP8
- the FP8-KV decode attention

On a mismatch, the engine does not start. This guards against wrong answers on sm_121
that give no error.

## Remaining for Phase 3

- The frozen 3,500 tok/s target. One candidate is a CUTLASS epilogue that writes SiLU*up as NVFP4 directly. This
  removes the round trip of the 142 MB / layer intermediate (~45 ms / chunk). Another candidate is to tune or port the
  FLA chunk kernels (~90 ms / chunk, 14%).
- The tool-calling quality gate needs the OpenAI-compatible server (Phase 5).

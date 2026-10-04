# Phase 2 progress: decode at the bandwidth roofline (2026-10-03)

Checkpoint `nvidia/Qwen3.8-27B-NVFP4` (stock, mixed NVFP4 / FP8), batch 1, one CUDA graph per decode
step, host reads every token back. `bench/decode_bench.py --kv-fp8`.

## Where decode stands

| Context | ms / token | tok/s | Ceiling at 233 GB/s | % of ceiling | vLLM / SGLang |
|---|---|---|---|---|---|
| 64 | 79.7 | **12.54** | 13.1 | 96% | 12.3 / 12.3 |
| 8k | 80.9 | **12.36** | 12.95 | 95% | 12.3 / 12.3 |
| 32k | 84.6 | 11.82 | 12.3 | 96% | |
| 128k | 98.9 | 10.11 | 10.5 | 96% | |

Ceiling = (17.56 GB of weights + FP8 KV + 0.30 GB GDN state read and written) / 233 GB/s.
**Frozen exit target (baseline.md section 5): >= 12.5 tok/s at 8k on the stock checkpoint. At 12.36, 1.1% short.**
The plan's original ">= 13.5 at 8k / >= 11 at 128k" assumed 15 GB per token and is not reachable on
this checkpoint (ceilings 12.95 / 10.5); it applies only after re-quantizing attention and GDN to NVFP4.

## Built this phase

| Piece | File | Result |
|---|---|---|
| NVFP4 W4A16 GEMV, fused SiLU*up, residual epilogue, M <= 4 | `csrc/gemv.cu` | 98% of 233 GB/s over all linears of a step (`bench/gemv_bench.py`) |
| FP8 W8A16 GEMV (per-tensor scale), BF16 GEMV | `csrc/gemv.cu` | 97% / tiny |
| Split-KV GQA decode attention, BF16 or FP8 KV, seq_len read on device | `csrc/attn_decode.cu` | 221-227 GB/s on the KV stream |
| Fused GDN step: conv + SiLU, then delta rule + gated RMSNorm, one block per head | `csrc/gdn_step.cu` | 300 MB of fp32 state per step in 1.2 ms |
| Zero-centered RMSNorm | `csrc/norm.cu` | 0.3 ms for 129 per step |
| Decode model on the kernels, device-side positions, FP8 KV | `engine/model/fast.py` | |
| Whole-step CUDA graph (`DecodeGraph`), generator | `engine/model/fast.py`, `engine/runtime/fast_generate.py` | graph == eager bit-exact |

Tests: `tests/test_gemv.py`, `test_attn_decode.py`, `test_gdn_step.py` (73 tests in the suite).

## Correctness (30 prompts, greedy, vs the Phase 1 weight-only NVFP4 reference)

| Path | Token-exact | Mean KL |
|---|---|---|
| Kernel linears, eager | 17/30 | 2.6e-4 |
| Full CUDA-graph path (kernel attention + GDN, BF16 KV) | 15/30 | 2.4e-4 |
| Final path: fused norms + residuals, FP8 KV (the 12.36 tok/s configuration) | 16/30 | 4.4e-4 |

All are ~40x or more below the quantized-path gate (KL <= 1.7e-2, docs/phase1_results.md). The kernels apply
the NVFP4 global scale in fp32 rather than rounding dequantized weights to BF16, so they are
slightly more accurate than the reference, not bit-identical to it.

## Step breakdown at 8k context (80.9 ms)

76.4 ms weight GEMVs (17.56 GB at 230 GB/s), ~1.2 ms attention, 1.2 ms GDN delta rule, 0.5 ms the
in_proj_b/a BF16 GEMV (only 12 blocks: poorly parallel), 0.3 ms norms, ~1 ms small torch ops
(q/k norm, rotary, gate, KV write) and host round trip.

## Remaining Phase 2 work

1. Close the last 1.1% at 8k: split-K for the 96-row BF16 GEMV (-0.3 ms); one GEMV launch with
   per-row scales for q/k/v and for in_proj_qkv + in_proj_z, removing the small-matrix tails
   (k/v at 82%) and 112 launches; fuse q/k-norm + rotary + gate + KV write into one attention prologue.
2. Multi-slot decode (B = 2, 3) in the graph and the 3-slot >= 0.8x-per-slot check.
3. Fused sampler (temperature / top-k / top-p) inside the graph; today the graph does argmax.
4. nsys trace of a step checked into `bench/traces/`.
5. Re-quantize attention + GDN projections to NVFP4 (gated on perplexity) to unlock the 14 tok/s target.

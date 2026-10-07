# Phase 2: decode at the bandwidth roofline (2026-10-03, complete)

Checkpoint `nvidia/Qwen3.8-27B-NVFP4` (stock, mixed NVFP4 / FP8), batch 1, one CUDA graph per decode step. The host
reads each token back. `bench/decode_bench.py --kv-fp8`.

## Where decode stands

| Context | ms / token | tok/s | Ceiling at 233 GB/s | % of ceiling | vLLM / SGLang |
|---|---|---|---|---|---|
| 64 | 78.4 | **12.75** | 13.1 | 97% | 12.3 / 12.3 |
| 8k | 79.6 | **12.56** (12.52 to 12.58 over 8 runs since the b/a overlap) | 12.95 | 97% | 12.3 / 12.3 |
| 32k | 83.3 | 12.01 | 12.3 | 98% | |
| 128k | 97.6 | 10.24 | 10.5 | 97.5% | |

Ceiling = (17.56 GB of weights + FP8 KV + 0.30 GB GDN state read and written) / 233 GB/s.

**The frozen exit target (baseline.md section 5) was >= 12.5 tok/s at 8k on the stock checkpoint. We met it: 12.56 to
12.58 over the last three runs, and 12.49 to 12.57 before the b/a overlap.**

The original plan target was ">= 13.5 at 8k / >= 11 at 128k". It assumed 15 GB per token, and this checkpoint cannot
reach it (ceilings 12.95 / 10.5). It applies only after a re-quantization of attention and GDN to NVFP4.

### Multiple slots (one graph step decodes all slots, FP8 KV)

| Slots | 8k: ms / step | tok/s per slot | aggregate | per-slot vs 1 slot | 32k aggregate |
|---|---|---|---|---|---|
| 1 | 80.1 | 12.49 | 12.5 | 1.00x | 12.0 |
| 2 | 83.2 | 12.01 | 24.0 | 0.96x | 22.1 |
| 3 | 87.4 | 11.44 | **34.3** | **0.92x** | 30.7 |

The plan criterion "3-slot decode >= 0.8x per slot": met (0.92x). The frozen target ">= 34 aggregate at 3 streams":
met (34.3, against 30.7 for vLLM and 30.6 for SGLang).

## Built in this phase

| Piece | File | Result |
|---|---|---|
| NVFP4 W4A16 GEMV, fused SiLU*up, residual epilogue, M <= 4 | `csrc/gemv.cu` | 98% of 233 GB/s over all linears of a step (`bench/gemv_bench.py`) |
| FP8 W8A16 GEMV (per-tensor scale), BF16 GEMV | `csrc/gemv.cu` | 97% / small |
| Split-KV GQA decode attention, BF16 or FP8 KV, seq_len read on device | `csrc/attn_decode.cu` | 221-227 GB/s on the KV stream |
| Fused GDN step: conv + SiLU, then delta rule + gated RMSNorm, one block per head | `csrc/gdn_step.cu` | 300 MB of fp32 state per step in 1.2 ms |
| Zero-centered RMSNorm | `csrc/norm.cu` | 0.3 ms for 129 per step |
| Fused attention prologue (q/k norm, partial RoPE, KV write) and gated combine | `csrc/attn_decode.cu` | 0.12 ms for 16 layers (was ~1 ms of torch ops) |
| q/k/v and GDN qkv/z stacked into one FP8 launch each, per-row scales (checkpoint scales kept exactly) | `csrc/gemv.cu`, `engine/model/fast.py` | |
| Decode model on the kernels, device-side positions, FP8 KV | `engine/model/fast.py` | |
| Whole-step CUDA graph (`DecodeGraph`), generator | `engine/model/fast.py`, `engine/runtime/fast_generate.py` | graph == eager bit-exact |

Tests: `tests/test_gemv.py`, `test_attn_decode.py`, `test_gdn_step.py` (73 tests in the suite).

## Correctness (30 prompts, greedy, vs the Phase 1 weight-only NVFP4 reference)

| Path | Token-exact | Mean KL |
|---|---|---|
| Kernel linears, eager | 17/30 | 2.6e-4 |
| Full CUDA-graph path (kernel attention + GDN, BF16 KV) | 15/30 | 2.4e-4 |
| Fused norms + residuals, FP8 KV | 16/30 | 4.4e-4 |
| Final path: + fused attention prologue / gate, stacked projections (the 12.53 tok/s configuration) | 14/30 | 4.9e-4 |

All paths are ~40x or more below the gate for quantized paths (KL <= 1.7e-2, docs/history/phase1_results.md). The
kernels apply the NVFP4 global scale in fp32. They do not round the dequantized weights to BF16. Thus they are slightly
more accurate than the reference, and not bit-identical to it.

## Step breakdown at 8k context (79.8 ms)

- 76.3 ms weight GEMVs (17.56 GB at 230 GB/s)
- ~1.2 ms attention
- 1.2 ms GDN delta rule
- 0.4 ms the in_proj_b/a BF16 GEMV
- 0.3 ms norms
- 0.2 ms prologue / combine / conv

The host round trip uses the rest. Almost no PyTorch elementwise kernels remain in the step.

## Sampling (in the graph)

`engine/runtime/sampler.py` does greedy, temperature, min-p, top-k and top-p sampling per slot from device buffers. The
GPU advances the Philox offsets. It takes ~0.15 ms per step. It uses the rejection sampler of FlashInfer, with one call
per slot. **FlashInfer 0.7.0.post1 accepts per-row seed/offset arrays, but the values of row 0 change the draw of each
row.** Thus batched calls would make the output of a request depend on its neighbours
(`tests/test_sampler.py::test_slot_independence`).

## Trace (`bench/traces/decode_8k_2026-10-03.nsys-rep`, summary `..._summary.txt`, `bench/trace_summary.py`)

The median step under nsys was 80.3 ms, with the **GPU idle for 0.36 ms (0.4%)** across 610 kernels. No launch bubbles
remain. GEMVs use 95.6% of the busy time. The in_proj_b/a BF16 GEMV runs on a side stream, in parallel with the qkv/z
GEMV. Its 3 ms in the trace is time that it shares the SMs with that GEMV. It does not add latency.

## Carried forward

- Re-quantize the attention + GDN projections to NVFP4 (gated on perplexity) to reach the 14 tok/s target.
- The sampler does not implement a repetition penalty. It needs the token history of each slot.

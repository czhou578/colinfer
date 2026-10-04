# Phase 2: decode at the bandwidth roofline (2026-10-03, complete)

Checkpoint `nvidia/Qwen3.8-27B-NVFP4` (stock, mixed NVFP4 / FP8), batch 1, one CUDA graph per decode
step, host reads every token back. `bench/decode_bench.py --kv-fp8`.

## Where decode stands

| Context | ms / token | tok/s | Ceiling at 233 GB/s | % of ceiling | vLLM / SGLang |
|---|---|---|---|---|---|
| 64 | 78.4 | **12.75** | 13.1 | 97% | 12.3 / 12.3 |
| 8k | 79.6 | **12.56** (12.52 to 12.58 over 8 runs since the b/a overlap) | 12.95 | 97% | 12.3 / 12.3 |
| 32k | 83.3 | 12.01 | 12.3 | 98% | |
| 128k | 97.6 | 10.24 | 10.5 | 97.5% | |

Ceiling = (17.56 GB of weights + FP8 KV + 0.30 GB GDN state read and written) / 233 GB/s.

**Frozen exit target (baseline.md section 5): >= 12.5 tok/s at 8k on the stock checkpoint: met (12.56 to 12.58 over the last three runs; before the b/a
overlap, 12.49 to 12.57).** The plan's original ">= 13.5 at 8k / >= 11 at
128k" assumed 15 GB per token and is not reachable on this checkpoint (ceilings 12.95 / 10.5); it applies
only after re-quantizing attention and GDN to NVFP4.

### Multiple slots (one graph step decodes all slots; FP8 KV)

| Slots | 8k: ms / step | tok/s per slot | aggregate | per-slot vs 1 slot | 32k aggregate |
|---|---|---|---|---|---|
| 1 | 80.1 | 12.49 | 12.5 | 1.00x | 12.0 |
| 2 | 83.2 | 12.01 | 24.0 | 0.96x | 22.1 |
| 3 | 87.4 | 11.44 | **34.3** | **0.92x** | 30.7 |

Plan criterion "3-slot decode >= 0.8x per slot": met (0.92x). Frozen target ">= 34 aggregate at 3 streams":
met (34.3; vLLM 30.7, SGLang 30.6).

## Built this phase

| Piece | File | Result |
|---|---|---|
| NVFP4 W4A16 GEMV, fused SiLU*up, residual epilogue, M <= 4 | `csrc/gemv.cu` | 98% of 233 GB/s over all linears of a step (`bench/gemv_bench.py`) |
| FP8 W8A16 GEMV (per-tensor scale), BF16 GEMV | `csrc/gemv.cu` | 97% / tiny |
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

All are ~40x or more below the quantized-path gate (KL <= 1.7e-2, docs/phase1_results.md). The kernels apply
the NVFP4 global scale in fp32 rather than rounding dequantized weights to BF16, so they are
slightly more accurate than the reference, not bit-identical to it.

## Step breakdown at 8k context (79.8 ms)

76.3 ms weight GEMVs (17.56 GB at 230 GB/s), ~1.2 ms attention, 1.2 ms GDN delta rule, 0.4 ms the
in_proj_b/a BF16 GEMV, 0.3 ms norms, 0.2 ms prologue / combine / conv; the rest is the host round trip.
Virtually no PyTorch elementwise kernels remain in the step.

## Sampling (in the graph)

`engine/runtime/sampler.py`: greedy, temperature, min-p, top-k, top-p per slot from device buffers, Philox
offsets advanced on the GPU; ~0.15 ms per step. Uses FlashInfer's rejection sampler, called once per slot:
**FlashInfer 0.7.0.post1 accepts per-row seed/offset arrays, but row 0's values perturb every row's draw**,
so batched calls would make a request's output depend on its neighbours (`tests/test_sampler.py::test_slot_independence`).

## Trace (`bench/traces/decode_8k_2026-10-03.nsys-rep`, summary `..._summary.txt`, `bench/trace_summary.py`)

Median step 80.3 ms under nsys, **GPU idle 0.36 ms (0.4%)** across 610 kernels: no launch bubbles remain.
GEMVs are 95.6% of busy time. The in_proj_b/a BF16 GEMV runs on a side stream overlapped with the qkv/z GEMV
(its 3 ms in the trace is time-sharing SMs with that GEMV, not added latency).

## Carried forward

- Re-quantize attention + GDN projections to NVFP4 (gated on perplexity) to unlock the 14 tok/s target.
- Repetition penalty is not implemented in the sampler (needs per-slot token history).

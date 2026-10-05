# Phase 6: pushing past the public numbers (progress, 2026-10-04)

Three changes so far. All of them keep greedy and seeded-sampled output token-identical across batch widths, draft
lengths and speculation on/off (`tests/scheduler_check.py`, `tests/spec_check.py`).

1. **Tensor-core skinny GEMM** (`csrc/skinny.cu`) for every decode-path linear.
2. **Adaptive draft length:** k = 3 or 7, chosen each cycle.
3. **NVFP4 re-quantization** of the checkpoint's FP8 attention and GDN projections, used for decode.

## Results

Single request, greedy, 200-token replies, `tests/scheduler_check.py`:

| Prompt | Phase 5 (k=3, GEMV) | + skinny GEMM, adaptive k | + NVFP4 attention / GDN |
|---|---|---|---|
| code edit | 42.1 | 66.4 | **74.7** |
| JSON | 42.4 | 71.0 | **76.9** |
| code generation | 37.4 | 50.2 | **51.0** |
| prose | 24.8 | 24.0 | **28.3** |
| mean | 36.7 | 52.9 | **57.7** tok/s |
| four requests at once (3 slots + 1 queued), aggregate | 42.4 | 59.3 | **68.7** |

Plain decode without speculation, 8k context (`bench/decode_bench.py`): 12.8 → **15.1 tok/s** with the
re-quantized weights. That would meet `docs/baseline.md`'s frozen target of ≥ 14.0 tok/s and its 14.5 stretch.

**The re-quantization fails its quality gate on code, so it is opt-in** (`--decode-weights requant`, section 3).
With the default FP8 projections:

- Single requests run at 66 / 71 / 50 / 24 tok/s on code edit / JSON / code generation / prose.
- Plain decode runs at 12.8 tok/s.

**Harness.** `~/Projects/model-benchmarks` core_runner, same settings as the Phase 0 baselines; 256-token greedy
outputs, 8 requests per concurrency level.

**Concurrency, aggregate tok/s:**

| Streams | Phase 6 (default: FP8 projections) | Phase 6, `--decode-weights requant` | Phase 5 | SGLang + MTP | vLLM |
|---|---|---|---|---|---|
| 1 | **68.5** | 75.6 | 41.3 | 34.1 | 12.2 |
| 2 | **105.4** | 116.5 | 49.1 | 44.1 | 23.3 |
| 3 | **97.0** | 108.0 | 46.6 | 70.5 | 30.7 |
| 4 (3 slots + 1 queued) | **112.1** | 124.5 | 53.2 | 76.3 | 45.0 |

**Single stream, prose with thinking on, tok/s:**

| Phase 6 (default) | Phase 6, re-quantized | Phase 5 | SGLang + MTP |
|---|---|---|---|
| 25.8-30.9 | 29.7-35.1 | 25.4-27.1 | 22.2-24.8 |

This is the harness's decode test. Without speculation, the default server decodes at 12.6-12.7 tok/s, and
15.3-15.4 with the re-quantized weights.

## 1. Skinny GEMM on tensor cores

**Problem.** The Phase 2 GEMV does M fused multiply-adds per weight on CUDA cores. It streams at 98% of bandwidth
for 1-4 rows but becomes compute-bound beyond that: +45% time at 8 rows, 2-4× at 12-16. That capped speculation
at k=3 for one request and forced k=1 at three.

**Kernel.**

- `mma.sync.m16n8k16` with BF16 × BF16 → fp32. Weights are dequantized exactly into BF16; e2m1 × e4m3 has at most
  5 significant bits.
- Weights stay in the checkpoint's layout, the same tensors the CUTLASS prefill reads, so there is no second copy.
- A warp owns 16 weight rows and walks K in 256-byte chunks per row.

**Read pattern.** What I learned on this LPDDR5x:

| Contiguous run per load instruction | Read rate on a 600 MB tensor |
|---|---|
| 512 B | 238 GB/s |
| 256 B | 235 GB/s |
| 128 B | ~225 GB/s |
| 64 B | 190-210 GB/s |

That ranges over 1 row × 512 B, 2 × 256, 4 × 128 and 8 × 64 B per instruction. Loading mma fragments straight
from memory gives the 64-byte pattern, because each lane needs its own row. The kernel therefore loads 2 rows ×
256 B per instruction into registers, one chunk ahead. It then transposes into fragment order through a per-warp
shared-memory scratch, with only `__syncwarp`, so warps never wait on each other.

Things that did not work:

- **Block-wide activation staging with barriers:** 10% slower.
- **A `cp.async` shared-memory pipeline:** it takes L1 away from the activations, and the long-K down projection
  fell to 100 GB/s at M=16 as activation reads thrashed to L2.

**Details.**

- **Fragment order:** a per-lane K-permutation makes a lane's B fragments contiguous scratch bytes and its A
  fragments 32 contiguous activation bytes.
- **Block scales:** loaded four chunks at a time (128 B per row).
- **Deterministic split-K:**
  - The split count depends on the matrix shape only.
  - The last item to finish a tile adds the partials in a fixed order.
  - This removes the tail of grids that are only 1.3 waves.

**Bit identity.** Every output row is bit-identical for any M from 1 to 16: rows beyond M are zeros. Every decode
path uses the same kernel, so speculation never changes outputs.

| Shape (M=1 / M=16, GB/s, `bench/skinny_bench.py`) | GEMV | Skinny |
|---|---|---|
| MLP gate+up (SwiGLU) 17408×5120 | 231 / 63 | 223 / 221 |
| MLP down 5120×17408 | 227 / 59 | 211 / 201 |
| lm_head 248320×5120 | 241 / 64 | 234 / 232 |
| GDN qkvz FP8 16384×5120 | 234 / 117 | 229 / 218 |
| attention qkv FP8 14336×5120 | 238 / 116 | 223 / 216 |

**Cycle times.** Decode step and MTP cycle, 8k context, checkpoint weights:

| | GEMV | Skinny |
|---|---|---|
| plain step, 1 / 3 slots | 78.3 / 85.9 ms | 81.5 / 88.3 ms |
| cycle, width 1, k=3 / k=7 | 95.6 / 149.5 ms | 94.4 / 111.3 ms |
| cycle, width 2, k=3 / k=7 | 139.7 / 276.0 ms | 102.7 / 126.0 ms |
| cycle, width 3, k=3 | 224.4 ms | 110.5 ms |

**Plain decode keeps the GEMV.** At 1-3 rows the GEMV is still about 4% faster. `set_linear_kernel` chooses per
process: the server uses the skinny GEMM with `--spec mtp` and the GEMV with `--spec none`, and each mode is
internally consistent.

## 2. Adaptive draft length

**Mechanism.**

- One graph exists per (slot range, greedy/sampled, k ∈ {3, 7}) where the verify rows fit 16.
- Each cycle picks the k with the most expected tokens per second: the sum over slots of (1 − a^(k+1)) / (1 − a),
  divided by the cycle time measured at startup.
- a is the slot's decayed per-token acceptance rate, starting at 0.6.
- Drafts beyond the last cycle's k are stale, and are not counted against acceptance.

**Result.** Code and JSON run at k=7, prose at 3. One request's cycles cost:

| k | Checkpoint weights | Re-quantized weights |
|---|---|---|
| 3 | 94 ms | 83 ms |
| 7 | 111 ms | 100 ms |

## 3. NVFP4 re-quantization of the FP8 projections

**What it is.** `nvidia/Qwen3.8-27B-NVFP4` keeps 208 linears in FP8: attention q/k/v/o and GDN
in_proj_qkv/z/out_proj, 7.2 GB of the 17.6 GB read per token. `tools/requant_nvfp4.py` quantizes the same tensors
from the BF16 checkpoint to NVFP4:

- Each 16-value block gets an e4m3 scale, chosen among amax/6 ... amax/4 by squared error.
- Stacked projections share a global scale, so they stay one launch.
- The result is 4.06 GB, written in 56 s.

**Use.** Decode streams these copies; prefill keeps the FP8 weights on its W8A8 GEMM. Weight error rises from
2.7% to 8.5% relative.

**WikiText hides the cost; code shows it.** Perplexity, ctx 2048, 65,504 tokens, weights dequantized to BF16
(`tests/perplexity.py`, `--text code` = the Python standard library):

| Weights | WikiText-103 test | Python code |
|---|---|---|
| BF16 | 6.9404 | |
| `nvidia/Qwen3.8-27B-NVFP4` as shipped | 6.9698 | 1.7257 |
| + attention / GDN re-quantized, round-to-nearest | 6.9789 (+0.13%) | 1.7650 (**+2.3%**) |
| + the same with GPTQ (64 calibration sequences, damping 0.01) | 7.0355 (+0.94%) | 1.7552 (+1.7%) |
| + GPTQ, 128 sequences, damping 0.1 | 7.0280 (+0.84%) | 1.7500 (+1.4%) |
| + GPTQ, 128 sequences, damping 0.3 | 6.9957 (+0.37%) | 1.7519 (+1.5%) |
| + GPTQ, 128 sequences, damping 1.0 | 7.0153 (+0.65%) | 1.7536 (+1.6%) |

**Where the code loss comes from.** It is spread evenly. Re-quantizing one subset at a time:

| Subset | Code perplexity |
|---|---|
| GDN `in_proj_qkv` | +0.62% |
| GDN `in_proj_z` | +0.38% |
| GDN `out_proj` | +0.59% |
| attention q/k/v/o | +0.66% |

That is round-to-nearest NVFP4 error, not one fragile layer.

**What GPTQ did.** It cuts the error the calibration inputs see from 5-8% to 2-4% per layer, but it raises weight
error to 11-12%. Text unlike the calibration set gets worse, hence the WikiText result above.

**The damping sweep.** The calibration set was 64 sequences of WikiText-103 train plus 64 of `transformers`
modeling code, disjoint from both evaluation corpora. Damping trades WikiText against code, and no setting gets
code below +1.4%.

**Conclusion.** NVFP4 for these projections costs real quality on code, about 1.5% perplexity, which is NVIDIA's
likely reason for leaving them in FP8.

**Status.**

- It stays opt-in: `--decode-weights requant`, using the GPTQ damping-0.3 file when present.
- The default decodes the checkpoint's FP8 projections.
- Getting the 18% back without the quality cost would need a mixed format, for example FP8 for the most sensitive
  rows or 6-bit weights. That is a kernel and format project of its own.

**Tool calling.** On the 85-task suite at 2048 tokens the scores sit within the same 8 borderline tasks:

| Configuration | Tasks passed |
|---|---|
| Re-quantized weights | 48 |
| Skinny GEMM with the FP8 projections | 51 |
| Phase 5 / vLLM | 53 / 53 |

Gate for making it the default: ≤ 0.5% on both corpora.

## Next

- Megakernel decode.
- A better drafter for prose. Acceptance there is about 0.45, so speculation adds about 1.15×.
- 4-bit KV for > 128k contexts.

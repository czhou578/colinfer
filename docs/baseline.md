# Baseline measurements (spark-44de)

Ground-truth numbers for this specific unit. Every performance target in `PLAN.md` is
derived from these via `tools/roofline.py`. Re-run after any driver, firmware, kernel or
toolchain change and after a reboot, and append a dated row rather than overwriting.

## Machine state at measurement

| | |
|---|---|
| Date | 2026-10-02 |
| GPU | NVIDIA GB10, compute capability 12.1, 48 SMs, 24 MB L2, SM clock 2.42 GHz |
| Memory | LPDDR5x, 256-bit, 8533 MT/s, spec 273 GB/s (CUDA reports the attribute as 8533 "MHz", i.e. the data rate) |
| Driver / CUDA | 580.173.02, CUDA 13.0 toolkit (nvcc 13.0.88), Linux 6.17.0-1031-nvidia |
| Python stack | torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1, nvidia-cutlass-dsl 4.8.0, CUTLASS 4.8.0 submodule |
| Other load | DeepSeek container stopped, 113 GiB free |

## 1. Memory bandwidth

Source: `bench/bw_bench.cu`. Build `make -C bench`, run `bench/build/bw_bench`.
8 GiB buffer, 10 timed iterations after 2 warmup, medians reported, GB = 1e9 bytes.
Every test verifies its result on the host (all checks passed in both runs).
Two back-to-back runs; the post-reboot repeat is still pending.

| Test | Launch | Bytes moved | Run 1 GB/s | Run 2 GB/s | % of 273 spec |
|---|---|---|---|---|---|
| **read, 16 B loads, 4x unroll (headline)** | 2048x256 | 8.59 GB | 228.8 | 229.5 | 84 |
| read, 16 B loads, no unroll | 2048x256 | 8.59 GB | 233.1 | 231.2 | 85 |
| read, 8 B loads | 2048x256 | 8.59 GB | 234.0 | 233.3 | 86 |
| read, 4 B loads | 2048x256 | 8.59 GB | 224.3 | 224.1 | 82 |
| read, best grid size | 48x256 | 8.59 GB | 239.3 | 239.2 | 88 |
| read, largest grid | 8192x256 | 8.59 GB | 231.2 | 231.3 | 85 |
| write (fill) | 2048x256 | 8.59 GB | 193.4 | 195.1 | 71 |
| read+write, in-place x += c | 2048x256 | 17.18 GB | 215.1 | 215.1 | 79 |
| read+write, copy dst = src | 2048x256 | 17.18 GB | 213.3 | 213.8 | 78 |
| cudaMemcpy device-to-device | - | 17.18 GB | 229.3 | 229.0 | 84 |
| strided: 16 B of every 32 B | 2048x256 | 4.29 GB useful | 116.2 | 114.5 | 42 |
| strided: 16 B of every 64 B | 2048x256 | 2.15 GB useful | 58.3 | 57.8 | 21 |
| strided: 16 B of every 128 B | 2048x256 | 1.07 GB useful | 67.7 | 68.0 | 25 |

Grid sweep for the read kernel (16 B, 4x unroll), run 2, median GB/s:

| Blocks x 256 threads | 48 | 96 | 192 | 384 | 768 | 1536 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|
| GB/s | 239.2 | 236.5 | 234.3 | 231.4 | 230.7 | 231.7 | 229.5 | 231.2 | 231.3 |

### What this fixes for the design

- **Planning number: 233 GB/s for streaming reads** (the plan assumed 230). Best case is
  240 GB/s with one block per SM. Anything claiming more than ~240 GB/s on this unit is
  measuring cache, not DRAM.
- **Launch shape is not a bandwidth lever.** 48 blocks (one per SM, 256 threads each, four
  16 B loads in flight per thread) already saturate; larger grids are 1 to 4% slower.
  Decode GEMV launch geometry can be chosen for the reduction structure, not for load
  count. Unrolling makes no difference either.
- **Use 8 B or 16 B loads.** 4 B loads cost 4%.
- **Writes are slower: 195 GB/s.** A kernel that reads X bytes and writes X bytes moves 2X
  at about 215 GB/s. Decode is almost pure reads, so this only matters for KV append,
  GDN state write-back and prefill outputs.
- **Fetch granularity is 64 B.** Touching 16 B out of every 32 B or 64 B takes exactly as
  long as reading everything; touching 16 B out of every 128 B takes half as long. A warp
  must consume whole 64 B chunks or bandwidth is wasted in proportion. This rules out any
  weight or scale layout where a thread's useful data is interleaved at finer than 64 B
  with data it skips. (Note: the 128 B-stride case implies about 271 GB/s of fetched bytes
  at 64 B granularity, above the 240 GB/s sequential best. This suggests the sequential
  path is capped slightly below raw DRAM by something upstream. Not yet explained; the
  233 GB/s figure is what real streaming kernels get and is the number to plan against.)

### Decode ceiling implied (`tools/roofline.py --bw 233`)

Qwen3.8-27B, batch 1, 8k context, no speculative decoding:

| Weights | GB per token | tok/s at 233 GB/s | 90% kernel-efficiency target |
|---|---|---|---|
| NVFP4 | 15.55 | 15.0 | 13.5 |
| FP8 | 26.39 | 8.8 | 7.9 |
| BF16 | 51.82 | 4.5 | 4.0 |

## 2. GEMM ceiling

Source: `bench/gemm_peak.py` (cuBLAS BF16, cuBLASLt FP8 via `torch._scaled_mm`, FlashInfer NVFP4
as a cross-check) and `bench/gemm_sm120.cu` (CUTLASS 4.8 SM120 kernels, one binary per tile
config, built by `make -C bench gemm -k`). Shapes are the Qwen3.8-27B MLP projections:
N x K = 17408 x 5120 (gate/up) and 5120 x 17408 (down). D = A * B^T, bf16 out, fp32 accumulate.
20 timed iterations after 3 warmup, medians. Every number below passed a numerical check
(CUTLASS: 65,536 sampled outputs against an independent dequantize-and-dot kernel; torch:
against an fp32 matmul; FlashInfer: relative error against fp32). Full output with all 11
CUTLASS configs: `docs/gemm_peak_2026-10-02.txt`.

### Peak TFLOPS by format (best verified kernel per cell, CUTLASS scheduler swizzle 8)

| N x K | Format | M=1 | M=16 | M=256 | M=2048 | M=4096 |
|---|---|---|---|---|---|---|
| 17408 x 5120 | NVFP4 (CUTLASS / FlashInfer b12x) | 0.8 | 12.1 | 164 | 355 | 347 / 373 |
| 17408 x 5120 | FP8 (cuBLASLt) | 0.4 | 6.7 | 96 | 200 | 199 |
| 17408 x 5120 | BF16 (cuBLAS) | 0.2 | 3.6 | 48 | 90 | 96 |
| 5120 x 17408 | NVFP4 (CUTLASS) | 0.7 | 11.8 | 167 | 309 | 335 |
| 5120 x 17408 | FP8 (cuBLASLt) | 0.4 | 6.8 | 94 | 193 | 195 |
| 5120 x 17408 | BF16 (cuBLAS) | 0.2 | 3.5 | 50 | 92 | 93 |

The published 356 TFLOPS NVFP4 / 188 FP8 figures for this chip are reproduced (359-373 / 200).

### CUTLASS NVFP4 tile configs (TFLOPS, swizzle 8)

| Tile, schedule | 17408x5120 M=16 | M=256 | M=2048 | M=4096 | 5120x17408 M=16 | M=256 | M=2048 | M=4096 |
|---|---|---|---|---|---|---|---|---|
| 128x128x128 pingpong | 10.9 | 153 | 330 | 319 | 10.2 | 145 | 281 | 322 |
| 128x128x128 cooperative | 11.3 | 152 | 337 | 320 | 10.7 | 150 | 284 | 326 |
| **128x128x256 cooperative** | **12.1** | **164** | 332 | 307 | **11.8** | **167** | 298 | **335** |
| 128x128x256 pingpong | 11.9 | 160 | 326 | 315 | 11.7 | 162 | 301 | 333 |
| 256x128x128 cooperative | 7.8 | 147 | **355** | **347** | 7.9 | 155 | **309** | 314 |
| 128x64x128 pingpong (undocumented, works) | 8.3 | 154 | 305 | 291 | 8.8 | 154 | 248 | 248 |
| 64x128x128 pingpong (undocumented) | does not compile: TMA tile/smem static_assert | | | | | | | |

### Weight-streaming efficiency at decode-sized M (GB/s of weight bytes, bandwidth ceiling 233)

| Kernel | 17408x5120 M=1 | M=16 | 5120x17408 M=1 | M=16 |
|---|---|---|---|---|
| cuBLAS BF16 (has a real GEMV path) | 231 | 227 | 221 | 221 |
| cuBLASLt FP8 | 194 | 190 | 215 | 213 |
| CUTLASS FP8 128x128x128 coop | 209 | 210 | 214 | 213 |
| **CUTLASS NVFP4 128x128x256 coop** | **217** | **213** | **211** | **207** |
| FlashInfer NVFP4 (b12x / cutlass) | 177 / 181 | 174 / 184 | 166 / 179 | 167 / 180 |

### Tile-scheduler raster swizzle (the L2 finding)

With CUTLASS's default scheduler settings, the down projection at M=4096 collapsed to 131 TFLOPS
(NVFP4) and the gate/up FP8 case to 128, while cuBLASLt held 195. Cause: at that size both
operands are ~35-45 MB against a 24 MB L2, so a plain raster re-streams one operand from DRAM
once per tile row (1.4 GB at 233 GB/s = 6 ms, versus 2 ms of math). Setting the persistent
scheduler's `max_swizzle_size` (exposed as `--swizzle`) fixes it:

| NVFP4 128x128x256 pp, 5120x17408, M=4096 | swizzle 0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| TFLOPS | 131 | 169 | 267 | **336** | 216 |

FP8 128x128x128 coop on 17408x5120 at M=4096 goes 128 -> 197 the same way. Swizzle 8 is
neutral or slightly negative (within 5%) for the cases that were already healthy, so the sweep
above uses 8 everywhere; the true optimum is shape-dependent. FlashInfer's SM120 NVFP4 backends
(b12x and cutlass) do not swizzle and still collapse to 117-132 TFLOPS on this case.

### What this fixes for the design

- **NVFP4 prefill ceiling is 310-370 TFLOPS on the real shapes, FP8 is 195-200, BF16 is 90-96.**
  NVFP4 is 1.75x FP8 and 3.7x BF16 for the same GEMM, so prefill is worth doing in NVFP4 and
  the plan's FP8 fallback costs almost half the throughput.
- **Prefill ceiling in tokens (`tools/roofline.py --bw 233 --tflops ...`, 48.7 GFLOP/token):**
  NVFP4 at 300 TFLOPS sustained = 6,100 tok/s at 2k context, 5,960 at 8k, 5,440 at 32k; FP8 at
  190 = 3,870 / 3,780 / 3,450; BF16 at 93 = 1,890 / 1,850 / 1,690. The plan's 2,500-3,500 tok/s
  target needs only 125-175 TFLOPS sustained across a whole layer, i.e. 40-50% of the NVFP4 GEMM
  ceiling, which leaves room for attention, GDN, norms and quantization overhead.
- **Tile choice for NVFP4:** 128x128x256 cooperative for M <= 256 (prefill chunks and the
  verify path), 256x128x128 cooperative for M >= 2048 on gate/up. K=256 beats K=128 by 5-8% at
  small M. There is no K=64 NVFP4 tile in the SM120 builder; K=64 exists only for FP8 and was
  never the best FP8 config. Shape-dependent, so keep the per-shape tile sweep in the plan.
- **The scheduler swizzle is mandatory for the down projection at long prefill.** Any CUTLASS
  GEMM we ship must set `max_swizzle_size` (8 here), and any third-party NVFP4 GEMM must be
  checked at M=4096, N=5120, K=17408 before trusting it.
- **FP8: use cuBLASLt, not CUTLASS.** cuBLASLt beats every CUTLASS SM120 FP8 config at
  M >= 2048 by 5-30% and matches it at small M. CUTLASS FP8 only matters if an epilogue fusion
  needs it.
- **Decode: a stock CUTLASS NVFP4 GEMM at M=1 already streams weights at 217 GB/s, 93% of the
  measured 233 GB/s**, and 213 GB/s (91%) at M=16. cuBLAS's BF16 GEMV reaches 99%. The plan's
  hand-written NVFP4 GEMV therefore has at most ~7% of raw bandwidth to gain; its value is in
  fused epilogues (silu * up, residual, quantization of the next activation) and in avoiding
  the 128-row tile's wasted MMA work, not in bandwidth. Make it measure against 217, not 233.
- FlashInfer's NVFP4 GEMMs are 15-20% slower than CUTLASS at decode M and collapse on the
  down projection; confirms the plan's choice to use FlashInfer only as an attention stopgap.

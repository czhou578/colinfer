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

## 2. GEMM peak (pending)

`bench/gemm_peak.py` and CUTLASS Example 79 on the model shapes: not yet measured.

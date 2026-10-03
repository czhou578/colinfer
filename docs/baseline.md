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

## 3. Checkpoint inventory: `nvidia/Qwen3.8-27B-NVFP4`

Source: `tools/tensor_inventory.py nvidia/Qwen3.8-27B-NVFP4` (reads safetensors headers only).
Full dump: `docs/nvfp4_inventory.txt`. Where the checkpoints live: `docs/checkpoints.md`.

**The checkpoint is mixed precision, not NVFP4 throughout** (`hf_quant_config.json`:
`quant_algo: MIXED_PRECISION`). What each module class is stored as:

| Module class | Count | Storage | Scales | Bytes streamed per token |
|---|---|---|---|---|
| MLP gate / up / down (64 layers) | 192 | NVFP4: U8 packed e2m1, `[N, K/2]` | `weight_scale` e4m3 `[N, K/16]` (block 16, plain row-major, **not** the CUTLASS 128x4 swizzle), `weight_scale_2` f32 global, `input_scale` f32 static | 9.63 GB (8.56 weights + 1.07 scales) |
| GDN `in_proj_qkv` `[10240,5120]`, `in_proj_z` `[6144,5120]`, `out_proj` `[5120,6144]` (48 layers) | 144 | **FP8 e4m3, per-tensor** | `weight_scale` f32 scalar, `input_scale` f32 scalar | 5.54 GB |
| Full attention q `[12288,5120]`, k/v `[1024,5120]`, o `[5120,6144]` (16 layers) | 64 | **FP8 e4m3, per-tensor** | f32 scalars | 1.68 GB |
| lm_head `[248320, 5120]` | 1 | NVFP4, block 16 | as MLP | 0.72 GB |
| Norms, conv1d `[10240,1,4]`, `in_proj_a/b` `[48,5120]`, `A_log`, `dt_bias` | | BF16 | | 0.05 GB |
| **Per-token total (backbone + lm_head)** | | | | **17.61 GB** |
| Embeddings `[248320, 5120]` | 1 | BF16 (one row per token, not streamed) | | 2.54 GB resident |
| MTP head (1 layer: full attention + MLP + fc `[5120,10240]`) | | **BF16, excluded from quantization** | | 0.85 GB per draft step |
| Vision tower (27 blocks + merger) | | BF16, unused for text | | 0.92 GB, do not load |

### What this changes

- **The plan's roofline assumed 14.98 GB per token at NVFP4; this checkpoint streams 17.61 GB.**
  With FP8 KV at 8k context (0.27 GB) and the GDN state read + written (0.30 GB), a decode step
  moves 18.2 GB: **ceiling 12.8 tok/s at 233 GB/s, 11.5 tok/s at the plan's 90% efficiency
  goal.** The public SGLang number for this checkpoint (12.32 tok/s) is therefore 96% of the
  bandwidth wall, not 85%. There is no kernel-level headroom on the stock checkpoint.
- **To get the plan's 15 tok/s base decode, the 7.2 GB of FP8 attention and GDN projections
  must be re-quantized to NVFP4 by us** (7.22 GB -> 4.06 GB, per-token 14.45 GB, ceiling
  15.5 tok/s). NVIDIA left them at FP8; the GDN in-projections are the likely accuracy
  reason, so this needs the perplexity harness before it is trusted. Decision for Phase 2.
- The NVFP4 scale tensors are plain `[N, K/16]` row-major; the loader must repack them into
  the CUTLASS SM120 scale-factor layout (the plan's "repack cache").
- Activation quantization is static per-tensor (`input_scale` scalar per linear): the
  fused norm -> quant kernels need only a scalar multiply, no per-token amax reduction.
- The MTP head is BF16 (0.85 GB). At 233 GB/s that is 3.6 ms per draft step, a quarter of a
  full base step; quantize it to NVFP4 ourselves before relying on MTP speculative decoding.
- Full-attention o_proj is `[5120, 6144]` and q_proj `[12288, 5120]`: 24 heads x 256 with a
  gate (q is 2x), matching the plan's gated-attention description.

### The other two checkpoints (for the roofline and the parity harness)

| Checkpoint | Scheme | Per-token bytes (backbone + lm_head) | Notes |
|---|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | FP8 e4m3 weights with **128x128 block-wise `weight_scale_inv` (BF16)**, dynamic per-token activation scales, no static input scales | **26.9 GB** (plan assumed 25.8; lm_head is BF16 here, 2.5 GB) | All linears incl. MTP are FP8; needs a block-scaled FP8 GEMM (CUTLASS `sm120_mma_tma_blockwise_scaling`), not the per-tensor path benchmarked in section 2. Ceiling 8.5 tok/s at 8k. Full dump `docs/fp8_inventory.txt`. |
| `Qwen/Qwen3.8-27B` | BF16, nothing quantized | **51.2 GB** (plan assumed 51.25) | Parity reference only. Ceiling 4.5 tok/s at 8k. Full dump `docs/bf16_inventory.txt`. |

## 4. Public-stack baselines on the NVFP4 checkpoint (2026-10-03)

Harness: `~/Projects/model-benchmarks` (`core_runner.py`), YAMLs `models/qwen3.8_27b_nvidia_nvfp4*.yml`,
raw summaries in `docs/baselines_2026-10-03.md`, run dirs under its `results/`. Settings chosen to
match the engine's use case: prefix / radix cache **off** (so prefill is raw), max 4 running requests,
128k context, fp8 KV cache, FlashInfer attention, thinking disabled in the harness prompts.

| Stack | Version / kernels | Notes |
|---|---|---|
| vLLM | 0.25.1 wheel (`~/Projects/model-benchmarks/.venv`, torch 2.11+cu130, flashinfer 0.6.13), `FlashInferCutlassNvFp4LinearKernel` for the NVFP4 MLPs, `FlashInferFP8ScaledMM` for attention/GDN, FlashInfer attention, CUDA graphs | The source build in `~/Projects/vllm` (July main) cannot serve this checkpoint natively: it gates the FlashInfer FP4 kernel to sm_100, excludes FlashInfer's `b12x` from auto-selection, and its Marlin fallback op is not compiled. First start JIT-compiles FlashInfer kernels for 12 min; cached after that (startup 140 s). |
| SGLang | 0.5.21 (`~/Projects/sglang/.venv`, torch 2.13+cu130, flashinfer 0.6.18, sgl-kernel aarch64), `--fp4-gemm-backend flashinfer_cutlass`, GDN backend, fused SiLU*up->FP4 quant before down_proj | Installed natively today. First start 530 s (JIT), then ~2 min. The harness's streaming TTFT probe gets no first-token timing from SGLang, so its TTFT / latency-sweep cells are empty; prefill throughput below is from SGLang's own per-batch log. |

### Decode (single stream, prose prompt, 512 to 2048 output tokens)

| Stack | tok/s | % of 12.8 tok/s ceiling (17.6 GB/token at 233 GB/s) |
|---|---|---|
| vLLM | 12.3 (peak 12.8) | 96% |
| SGLang | 12.3 | 96% |
| vLLM + MTP (`num_speculative_tokens` 3), mean acceptance length 2.1 to 2.45 | 22 to 25 | 1.8 to 2.0x |
| SGLang + NEXTN (3 steps, 4 draft tokens), accept length 2.6 to 4.0 | 22 to 25 on the prose prompt, 34 on the concurrency prompts | 1.8 to 2.8x |

Both stacks sit on the bandwidth wall of the stock checkpoint. The plan's "85% of ceiling" for the
public stacks was based on the 15.0 GB/token assumption; against the real 17.6 GB/token they are
at 96%, so **a better decode kernel cannot beat them on this checkpoint**; only fewer bytes
(re-quantized attention/GDN, section 3) or speculation can.

### Prefill (prefix cache off)

| Prompt tokens | vLLM TTFT median | vLLM prefill tok/s | SGLang prefill tok/s (own log) |
|---|---|---|---|
| 512 | 0.25 s | 2,150 | |
| 2,048 | 0.81 s | 2,530 | ~1,530 (2k chunks) |
| 8,192 | 3.9 s | 2,040 | ~1,660 (8k chunks) |
| 16,384 | 8.1 s | 2,010 | |
| 32,768 | 18.0 s | 1,800 | |
| 65,536 | 43.3 s | 1,515 | |

vLLM's 2,530 tok/s at 2k is 125 TFLOPS effective, about 35% of the measured NVFP4 GEMM ceiling
(section 2); the fall-off beyond 8k is attention at head_dim 256, not GEMM. Public figure in
the plan (1,800 tok/s) is reproduced at 32k.

### Concurrency (256-token outputs, aggregate tok/s)

| Streams | vLLM | SGLang | SGLang + NEXTN |
|---|---|---|---|
| 1 | 12.2 | 12.2 | 34.1 |
| 2 | 23.3 | 23.3 | 44.1 |
| 3 | 30.7 | 30.6 | 70.5 |
| 4 | 45.0 | 44.6 | 76.3 |

Decode batching is nearly free up to 4 streams (bandwidth-bound, weights read once per step),
which is the whole premise of the plan's three-slot scheduler and of batched verification.

## 5. Frozen Phase 2 to 4 targets (from the measurements above)

Units: Qwen3.8-27B, batch 1 unless stated, 8k context, prefix cache off. "Ceiling" means the
roofline from sections 1 to 3; "baseline" means the better of vLLM / SGLang above.

| Metric | Ceiling | Baseline | Target | Why |
|---|---|---|---|---|
| Base decode, stock checkpoint (17.6 GB/token) | 12.8 tok/s | 12.3 | **>= 12.5 tok/s (97%)**, token-exact with HF BF16 | Only 4% of headroom exists; this is a correctness-and-parity milestone, not a speed one. |
| Base decode, own NVFP4 re-quant of attention + GDN (14.45 GB/token) | 15.5 tok/s | none | **>= 14.0 tok/s (90%)** with perplexity within 1% of stock | The only route to the plan's 15 tok/s. Gate on the perplexity harness. |
| Decode step kernel efficiency | 233 GB/s | CUTLASS GEMM at M=1: 217 GB/s (93%) | **>= 215 GB/s effective weight streaming per step** incl. attention, GDN state, norms | A hand-written GEMV has at most 7% over stock CUTLASS; the step must be fused end to end. |
| Three concurrent streams | 3 x 12.8 | 30.7 aggregate | **>= 34 aggregate (>= 11.3 each)** | Batched weight reads; current stacks lose 16% at 3 streams. |
| MTP speculative decode, prose | | 22 to 25 tok/s (acceptance 2.1 to 2.5) | **>= 35 tok/s** (acceptance >= 3.0 at k=4, verify step <= 1.15x base step) | Plan's 35 to 45 band; verification of 5 tokens costs about one base step (section 2, M=16 at 213 GB/s). |
| MTP speculative decode, code | | 34 tok/s (SGLang, mixed prompts) | **>= 50 tok/s** | Plan's "better drafter" band; acceptance on code is higher. |
| Prefill throughput, 2k to 8k prompts | ~6,000 tok/s at 300 TFLOPS | 2,530 / 2,040 | **>= 3,500 tok/s** (>= 175 TFLOPS effective, 50% of the NVFP4 GEMM ceiling) | GEMMs alone allow 6,000; attention, GDN chunk kernels, norms and quant get the other half. |
| Prefill throughput, 32k | ~5,400 | 1,800 | **>= 2,700 tok/s** (TTFT <= 12 s) | Needs an attention kernel that holds >= 60% of tensor peak at head_dim 256. |
| TTFT, 2k prompt | | 0.81 s | **<= 0.6 s** | Follows from 3,500 tok/s. |
| Startup to first token after process start | | 140 s (vLLM, warm JIT) | **<= 30 s** | No JIT, prebuilt kernels, repacked weights cached on disk. |

Re-freeze these after any driver, firmware or toolchain change (re-run sections 1 and 2 first).


# Plan 2: MoE kernels for qwen4_exp (planner 2, Opus, 2026-10-24) — saved verbatim by the lead

## Facts checked
- The expert input_scale is one value per layer for all 512 experts: gate == up (0.001389 in layer 0), down has its own (0.003185).
  weight_scale_2 differs per expert (218 distinct values in layer 0) but gate == up within each expert. So prefill quantizes each
  token once per layer. The kernels still take in_scale[E,3] and gs[E,3].
- torch.topk on CUDA: ties at the top-10 boundary keep the lowest indices (0 mismatches over 16.6k tied rows); the order inside a run
  of equal values is not stable; ~11% of rows with random bf16 logits have a boundary tie. Our rule: logit descending, index
  ascending, compared on the bf16 logits. Reference model: torch.sort(p, descending=True, stable=True)[..., :10].
- CUTLASS 4.8 has the block-scaled ptr-array kernel on SM120: KernelPtrArrayTmaWarpSpecializedCooperative ->
  MainloopSm120ArrayTmaWarpSpecializedBlockScaled, with per-group ptr_SFA/ptr_SFB and layout_SFA*/layout_SFB*, per-group output
  scale factors and norm constant (Sm120BlockScaleFactorRowStore with a pointer element type), Sm90ScalarBroadcastPtrArray for a
  per-group alpha. With host_problem_shapes = nullptr the grid is sm_count and the shapes are read on the device: no host sync.
  Groups with 0 rows look supported (0 tiles), untested. (Headers read; nothing compiled.)

## Decisions (recommendation first)
1. Scale layout: one copy of the expert scales, in the CUTLASS swizzled layout; the decode kernels read it directly (a warp owns rows
   {128b + 32j + 4w + i} -> 64-byte runs, about -2% stream rate). Alternatives: a row-major copy for decode (+7.5 GB), or swizzling
   at every prefill layer (+22% prefill MoE time).
2. Decode grid: grid = 10*M pair slots x column tiles. The warp that holds the first occurrence of an expert in pair order owns it and
   gathers its rows with ballots; the others exit. No schedule kernel, no split-K. (Alternative: all 512 experts with early exit ->
   ~25k mostly empty blocks per layer, ~80 us estimated.)
3. Fixed summation order: the down kernel writes fp32 partials yp[M*10, 2560]; moe_combine sums in k-slot order, adds
   sigmoid(gate) * shared, rounds to bf16 once.
4. Shared expert and router at decode: add a BF16 format to skinny.cu (planner 1 needs it for the BF16 projections too). Stack the
   router [512] and shared_expert_gate [1] into one [520, 2560] pass. NVFP4/INT copies later, through the perplexity gate.
5. Prefill: phase 1 = grouped GEMM gate|up (N 1280, tile 128x128x256, bf16 out), then swiglu_quant, grouped down GEMM (128x128x128,
   K = 640), combine in fixed order. Phase 2 = port the SwigluNvfp4 fusion to the ptr-array epilogue.

## Estimates
| Case | Experts read per layer | Time |
|---|---|---|
| Decode, M = 1 | 27.6 MB | ~6.0 ms per token over 48 layers (+ router 0.6 ms, BF16 shared expert 2.1 ms) |
| Verify, M = 8 | ~75 distinct experts (uniform routing), 206 MB | ~45 ms |
| Verify, M = 4 | ~39 distinct experts | ~23 ms |
- D(M) for uniform independent routing: 19.8, 38.8, 74.7, 138 distinct experts for M = 2, 4, 8, 16. Verify ~ 48 * D * 2.76 MB / 220 GB/s.
- Verify cost grows almost linearly with distinct experts -> best k moves to about 1-3 (~17 ms/token vs ~21.5 ms plain, assuming
  15 ms of non-expert weights per step). Scheduler model: cycle(k) = T0 + 48 * 2.76 MB * D(k+1) / BW; D from distinct-expert counters
  that moe_route writes + an offline tool on real prompts.
- Prefill at 2048 tokens per chunk is limited by weight reads, not math: every chunk touches all experts, 1.415 GB per layer at
  ~200 GB/s = 7.1 ms; the math is 201 GFLOP per layer = ~0.7 ms even with 3.2x padding. MoE ~400 ms per chunk -> ceiling ~5.1k tok/s.
  With 8192-token chunks ~14k tok/s. RECOMMEND CHUNK = 8192 for this model.

## Risks
- The ptr-array block-scaled build is unverified: compile time, smem fit at 128x128x256, groups with 0 rows, argument tensors built on
  the device. 64-byte scale runs may cost more than 2%. PDL ordering: the down kernel reads the routing before its wait, which relies
  on moe_gu triggering only after its own wait. moe_gu cannot prefetch weights before routing is known (~5 us bubble per layer; hide
  it by running the shared expert on a parallel graph branch).

## Interfaces
- From the loader, per layer: w_gu u8[E,1280,1280] (gate rows 0..639, then up rows); sf_gu u8[E,204800] swizzled as an N = 1280,
  K = 2560 operand; w_dn u8[E,2560,320]; sf_dn u8[E,102400] swizzled; gs f32[E,3]; in_scale f32[E,3]; router_w bf16[520,2560]
  (router + shared_expert_gate); shared expert BF16 [1280,2560] + [2560,640]; the BF16 skinny format; a reference model with an
  fp32 slot-order combine.
- To the layer code: moe_forward(x[M,2560] bf16) -> y[M,2560] bf16, shared expert included, no residual. MTP experts become NVFP4
  copies so they run on the same kernels; the skip flag is honored.

## Open questions
1. May the prefill chunk become 8192 tokens? 2. Is the fp32 combine accepted (the reference rounds in bf16 in expert-id order)?

---------------------------------------------------------------------------------------------------------------------------------
# Detailed plan

## A. Weight layout (loader)
Per layer, contiguous, on the GPU: 2,764,800 B per expert, 1.4156 GB per layer.
- w_gu u8[E, 2I, K/2] = [512,1280,1280]: rows 0..639 gate_proj, rows 640..1279 up_proj (concatenated, not interleaved).
- sf_gu u8[E, 2I*K/16]: each expert's [1280,160] swizzled with nvfp4_swizzle_sf at load time. 1280 % 128 = 0 and 160 % 4 = 0: no
  padding. The up half starts at byte 102400 (640 % 128 = 0).
- w_dn u8[E,2560,320]; sf_dn u8[E,2560*40] swizzled, no padding.
- gs f32[E,3] = weight_scale_2 of (gate, up, down); in_scale f32[E,3].
- The loader asserts gate == up for both the input scale and the weight scale (the stacked prefill GEMM needs one alpha per group;
  if it fails, two GEMMs). No row-major copy of the scales.

## B. Decode: csrc/moe.cu
Move fp4x2_scaled, e4m3_to_h2, h2_to_bf2 from skinny.cu into csrc/skinny_core.cuh so both dequantize the same way. Add the BF16
format to skinny.cu (KC = 128 k per 256-byte run, no scales; SwiGLU, bf16 or fp32 output).

1. moe_route(logits bf16[M, Nl >= 513], ids i32[M,10], w f32[M,10], sg f32[M], stats i32[2] | None)
   - One block of 512 threads for M <= 48 at decode, one warp per row; at prefill ceil(M/16) blocks.
   - PDL_WAIT; each lane holds 16 logits and builds the key ord(bf16) << 16 | (0xFFFF - idx); ten rounds of warp-wide argmax give the
     top 10 (logit descending, lowest index first on ties: exactly torch's set); m = max, S = sum expf(l - m) in fp32 (precise expf);
     p_k = expf(l_k - m) / S; w_k = bf16(p_k / sum_k p_k) summed in slot order; sg = bf16(sigmoid(bf16 logit[512])).
   - Stats: a 512-bit bitmap in smem; atomicAdd(stats[0], popc); stats[1] += 1. Skipped when the skip flag is set.
2. moe_gate_up(x bf16[M,K], ids, w_gu, sf_gu, gs, h bf16[M*10, I])
   - Grid ceil(10M * (I/16) / 4) blocks of 4 warps; item = blk*4 + warp; slot = item / 40; ct = item % 40.
   - Prologue: PDL_WAIT then PDL_TRIGGER (the trigger after the wait so moe_down may read ids before its own wait); skip flag -> return;
     e = ids[slot]; if any earlier pair p' < slot has the same expert -> return (a ballot over <= 10M ints); collect the row list =
     pairs p >= slot with ids[p] == e in increasing p order (ballot + popc).
   - Rows owned by the warp: rb = ct/8, w8 = ct%8. Gate rows G(rho) = 128 rb + 32 j + 4 w8 + i with rho = 4j + i (0..15); up rows
     G + 640. 32 weight rows = NT 4 mma n-tiles: n-tiles 0-1 gate, 2-3 up, same 16 output columns.
   - Streaming as skinny.cu: 5 chunks of 512 k; a row's chunk is a 256-byte run; a lane holds 16 uint4 (8 KB per warp) one chunk
     ahead. Scales per chunk: for each of the 8 atoms, 64 contiguous bytes at ((rb*40 + 8c + a) * 512 + 64 * w8); up uses rb + 5:
     2 uint4 per lane. Per-warp scratch 32 x 288 + 1024 B.
   - MMA for each 64-k group and each row group rg < RG (RG = 1 when M <= 16, 3 when M <= 48): A = rows rowlist[16 rg + g] and
     rowlist[16 rg + g + 8] of x (32 B per lane per row); B = the dequantized scratch in skinny's fragment order.
   - Epilogue: h[p, col] = bf16(silu(g * gs0) * (u * gs1)); the pair b = 0, 1 maps to adjacent physical rows -> one bf16x2 store.
3. moe_down(h, ids, w_dn, sf_dn, gs, yp f32[M*10, K])
   - Same slot ownership; 160 column tiles per expert (rb = ct/8 in 0..19). Before PDL_WAIT it reads ids with __ldcg and loads the
     whole K = 640 at once: 10 uint4 per lane (320-byte runs) + 10 scale atoms x 64 B. PDL_WAIT, then 10 groups of mma over the rows
     h[p] of the pair list. yp[p, n] = acc * gs2[e] (no routing weight).
4. moe_combine(yp, w, sg, shared bf16[M,K] | None, y bf16[M,K]): one thread per 8 columns:
   y = bf16(sum_{k=0..9} w[m,k] * yp[10m + k] + sg[m] * shared). Skip flag -> zeros.

Bit identity: routing reads only its row; a row's position inside an mma / its row group does not change its sum (skinny's premise,
tested); k order fixed; no split-K, no atomics on outputs; the combine adds in slot order.

Python class (fast_qwen4exp.py): Qwen4ExpMoE(router_w[520,2560], experts dict(w_gu, sf_gu, w_dn, sf_dn, gs, in_scale),
shared (w_sgu[1280,2560], w_sd[2560,640])). forward(x[M,2560]) -> y: router skinny_bf16 -> moe_route; side branch: the shared expert
(skinny_swiglu_bf16 then skinny_bf16); moe_gate_up -> moe_down -> moe_combine. Buffers sized for M_max = 16 (or 48), shared by all
layers: h 0.6 MB, yp 4.9 MB at M = 48.
Bindings checks: M <= 48; K % 512 == 0; I % 128 == 0; I in {256, 640} for the down template; ids i32 [M,10]; contiguous; the skip
pointer from skinny's g_skip, read before the wait.
Numbers: M = 1: gate_up 18.4 MB + down 9.2 MB = ~135 us per layer.

## C. Prefill: csrc/moe_prefill.cu
1. Router: cuBLAS bf16 matmul against [520, 2560], then moe_route with any M.
2. Plan (torch, no sync): order = argsort(ids.flatten(), stable=True); counts = bincount(512); off = cumsum; SF row offsets
   sfo = cumsum(ceil(m_e/128) * 128).
3. moe_group_args kernel (one thread per expert) writes the CUTLASS arguments per group: problem_shapes[e] = (m_e, N, K); ptr_A,
   ptr_SFA, ptr_D; layout_SFA[e] = tile_atom_to_shape_SFA(m_e, N, K, 1). Per-layer arrays ptr_B, ptr_SFB, layout_SFB, alpha
   pointers and strides are built once at load.
4. moe_gather_quant: one warp per sorted row; quantizes x[t] with in_scale[e] (the code of k_quant) into A[r] and into the swizzled
   SFA of group e at local row r - off[e]. Padding rows zeroed. SF padding <= 512 x 127 rows: 10 MB (gate_up) + 2.6 MB (h scales).
   A needs no data padding (TMA zero-fills out-of-bounds rows).
5. Gate_up grouped GEMM: N = 1280, tile 128x128x256, bf16 D[P,1280], per-group alpha = in * gs.
6. moe_swiglu_quant: silu(g) * u -> NVFP4, per-group swizzled SF of the down A operand, norm constant 1/in_dn.
7. Down grouped GEMM: tile 128x128x128, bf16 out.
8. moe_combine_prefill: per token sum_k w * ydown[inv[t*10 + k]] in k order + sg * shared. The shared expert on cuBLAS bf16
   (~0.22 ms per layer); later group 513 with an NVFP4 copy.
9. Determinism: stable sort, no stream-K, fixed order everywhere.
Type recipe: GemmUniversal<GroupProblemShape<Shape<int,int,int>>, Main, Epi> with RowMajor* / ColumnMajor* layout tags;
KernelPtrArrayTmaWarpSpecializedCooperative; epilogue PtrArrayTmaWarpSpecializedCooperative; fusion arguments alpha_ptr_array,
dAlpha = {_0,_0,1}. Phase-2 fusion: FusionCallbacks on Sm120PtrArrayTmaWarpSpecialized, Sm90ScalarBroadcastPtrArray for alpha,
RowStore with a float_ue4m3_t* element type.
Fallback: a loop over experts with nvfp4_gemm: 1024 launches per layer (~20 us host-side CUTLASS init each) -> ~1 s per chunk,
~2k tok/s: bring-up only.
Time per 2048-token chunk: weights 7.1 ms x 48; gather 0.15 + swiglu 0.3 + combine 0.5 + shared 0.22 + router/plan 0.1 = ~1.3 ms
x 48; total MoE ~400 ms.

## D. Tests and benches
- tests/test_moe.py: moe_route equals the torch stable-sort reference exactly (forced ties: the selected set equals torch.topk's
  set); full moe_forward against an fp32 HF-style loop built with dequant_nvfp4 per expert (row-major scales, swizzled for the
  kernel): rel < 5e-3; row identity for M = 1, 7, 8, 16, 48, each row alone and in random subsets, all / no / partial expert
  overlap; empty experts; the skip flag (zero output); a CUDA graph captured and replayed with new routing equals eager; bad shapes
  rejected; one test at the real E = 512 shape.
- tests/test_moe_prefill.py: group sizes {0, 1, 127, 128, 129, 300, random} vs the quant_emul reference; gather-quant bytes and
  unswizzled SF bit-exact vs nvfp4_quant of the gathered rows; swiglu_quant agrees with the emulation on >= 97% of values; full MoE
  prefill vs the reference rel < 2e-2; two runs torch.equal.
- engine/selftest.py: E = 16, K = 1024, I = 256, M = 3 decode + a small grouped GEMM. Sanitizers on both test files.
- bench/moe_bench.py: decode GB/s for D in {10, 20, 40, 75, 138} cycling over 4 layer copies (nothing from L2); prefill TFLOPS and
  GB/s for T = 2048, 4096, 8192 with uniform and Zipf group sizes. tools/moe_routing_stats.py: D(M) on real prompts.

## E. Work breakdown (~80 h; all can start now with random weights)
| Step | Task | Hours |
|---|---|---|
| 1 | BF16 skinny format | 6 |
| 2 | skinny_core.cuh refactor | 1 |
| 3 | moe_route + tests | 4 |
| 4 | moe_gate_up | 10 |
| 5 | moe_down with prefetch before the wait | 6 |
| 6 | combine, Python class, graph test | 4 |
| 7 | test_moe + sanitizers | 5 |
| 8 | decode bench and tuning (>= 200 GB/s at D = 75) | 7 |
| 9 | CUTLASS grouped compile probe | 4-8 |
| 10 | args, plan, gather | 6 |
| 11 | swiglu_quant, down, combine | 5 |
| 12 | test_moe_prefill | 5 |
| 13 | fused epilogue (optional) | 6 |
| 14 | selftest | 2 |
| 15 | prefill bench + chunk-size study | 3 |
| 16 | routing stats on the real model | 3 |
Critical files: csrc/skinny.cu, csrc/gemm_nvfp4.cu, csrc/bindings.cpp, cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp,
cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp, engine/model/prefill.py.

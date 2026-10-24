# Qwen3.8-Flash-Next (qwen4_exp) on colinfer: the plan

Written 2026-10-24. Three planners (Claude Opus 5.5) each studied one area and reported; this page merges their plans and
records the decisions. The detailed plans are in `docs/history/qwen4exp/`: `plan_model_weights.md`,
`plan_moe_kernels.md`, `plan_hc_qsa_ple_serving.md`, the common brief and the tensor list of the checkpoint.

## 1. The goal

The engine serves `nvidia/Qwen3.8-27B-NVFP4` (dense) today. The goal is to serve the MoE model
`Qwen/Qwen3.8-Flash-Next` from the checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` (132.7 GB) on the same DGX Spark,
with the same guarantees: exact speculation, one request at a time, a perplexity gate of 0.5% for every quantization
beyond the checkpoint's own. The 27B path stays byte-for-byte as it is. The engine picks the model family from
`model_type` in `config.json`.

The model is not "the 27B plus a MoE block". It has five new mechanisms, and the engine needs all five to run it:

| Mechanism | What it is | New code |
|---|---|---|
| MoE | 512 experts per layer, top-10 + 1 shared, intermediate 640, hidden 2560, 48 layers | routing + expert GEMM kernels for decode and prefill |
| Hyper-connections | the residual stream is 4 x 2560 wide; a gated residual replaces the input / post norms | grouped norm, low-rank mix, injection kernels |
| Qwen Sparse Attention (QSA) | GQA 24/2, head dim 256; an indexer (4 q heads, 1 key head of 128) scores 4-token blocks; each query attends to the top 512 blocks plus the tail | indexer, block keys, top-k, sparse decode attention, sparse prefill |
| PLE n-gram table | at layer 2: 16 hashed bigram / trigram heads over a 51.2 GB FP8 table on disk, a gate per stream, a dilated conv | host row reads, hashing, the layer in torch ops first |
| MoE MTP block | one QSA + MoE decoder layer with hyper-connections, FP8 experts | a new drafter (later milestone) |

The Gated DeltaNet layers have the 27B's dimensions (48 / 16 heads, 128 / 128). Their kernels are reused. One
difference: the output gate is `sigmoid(z)`, not `silu(z)`.

## 2. The checkpoint

- 10 shards hold the routed experts in NVFP4, one tensor per expert and projection (`weight` u8 [N, K/2],
  `weight_scale` e4m3 [N, K/16], `weight_scale_2` f32, `input_scale` f32), and everything else in BF16: attention, GDN,
  shared experts, routers, hyper-connections, PLE projections, embeddings, lm_head, the vision tower.
- `input_scale` is one value per layer and projection (gate equals up; down has its own). `weight_scale_2` differs per
  expert (gate equals up). Prefill therefore quantizes a token once per layer.
- `model-fp8-mtp-ple.safetensors` (53.7 GB) holds the n-gram table as 128 shards of F8_E4M3 [2,500,012, 160] with one
  bf16 scale, and the MTP experts in FP8 with 128 x 128 block scales.
- The padded n-gram vocabulary is 320,001,536 = 128 x 2,500,012 rows exactly. Hash products stay below 2^63.
- The chat template is byte-identical to the 27B's. The eos of the n-gram context is `<|endoftext|>` (248044), which
  the template never emits. The MTP block has `fc_embedding` and `fc_hidden` of [2560, 2560] and a
  `pre_fc_norm_hidden` of [10240].

## 3. Decisions

| Topic | Decision | Why | Alternative |
|---|---|---|---|
| Reference model | `engine/model/qwen4exp.py` mirrors transformers op for op; `Qwen4ExpConfig` subclasses `Qwen35Config`; GDN and attention are subclassed | the tiny-model test against transformers proves it on the CPU; `qwen35.py` stays unchanged | a copy of qwen35.py |
| Expert storage on the GPU | per layer: `w_gu` u8 [512, 1280, 1280] (gate rows, then up rows), `w_dn` u8 [512, 2560, 320], scales in the CUTLASS swizzled layout only, `gs` and `in_scale` f32 [512, 3] | one copy of the scales serves decode and prefill; a second copy costs 7.5 GB | row-major scales for decode, swizzled at prefill (+22% prefill time) |
| Loading | the direct reader writes each expert into its stacked slot (`read_tensors(..., into=)`); the scales are swizzled on the device, one layer at a time | no 68 GB of copies, no offline step; about 9 s for 78 GB | offline stacking in the prepared directory |
| MoE decode kernels | grid over the 10 M (row, expert) pairs; the first pair of an expert owns it; fp32 partials per pair, one combine in slot order; a BF16 skinny format for the router and the shared expert | row-invariant by construction; the weights stream as in `skinny.cu` | a grid over all 512 experts with early exit |
| MoE prefill | the CUTLASS 4.8 SM120 block-scaled ptr-array GEMM (gate\|up, then down), a gather-quantize kernel, a fixed-order combine | the kernel exists in the headers; no host sync | a loop over experts with the dense GEMM (about 2k tok/s: bring-up only) |
| Prefill chunk | start at 2048; measure 4096 and 8192 for this family | a chunk reads every expert once (7 ms per layer), so a larger chunk is near-free | keep 2048 |
| Reduction order | the reference keeps HF's order; the kernels sum in slot order in fp32; tests use a tolerance | exactness is required between the kernel's own paths (decode = verify), not against HF's rounding | match HF's bf16 order in the kernel |
| QSA state | raw indexer keys bf16 [L, 128] and a derived block-key cache [L/4, 128] per slot and layer; rope positions [L, 3]; sessions save the raw keys and rebuild the block keys | the pooled keys are needed at every step; raw keys keep the restore exact | raw keys only (4x the indexer bytes) |
| Decode attention | one sparse kernel always, with the dense list for short rows | a per-launch choice of kernel would make a row's bits depend on its neighbours | a dense kernel below 2051 tokens |
| Prefill attention | M1: FA2 causal for prompts of 2051 tokens or fewer; M2a: FA2 with the selection as a custom mask; M2b: a kernel over the union of the selected blocks of each 64-row tile | exact at every stage; FlashInfer's block-sparse wrapper shares no K/V across rows and needs a host plan per layer | FlashInfer BSR |
| Top-k ties | score descending, index ascending, -0.0 as +0.0; the same rule in every path | torch.topk on CUDA keeps the lowest index at the boundary; ties are common (blocks that score 0) | leave it to the kernel |
| N-gram table | stays on disk; the host hashes the ids from the slot's history and reads 16 rows of 160 B per token with O_DIRECT through a thread pool; a bounded row cache later, sized by a hot-set tool | 51 GB does not fit; O_DIRECT keeps the page cache from holding 4 KB per 160 B row | buffered reads with the page cache as the cache; a 25.6 GB resident 4-bit table |
| PLE and hyper-connections in M1 | torch ops inside the graph | one layer (PLE) and small matrices (HC); kernels come with speculation (M3) | kernels from the start |
| Decode copies | phase B: INT6 / INT5 / NVFP4 copies of the BF16 matrices (GDN, attention, HC, shared experts, lm_head) under the perplexity gate | the exact path streams 10.2 GB per token (about 22 tok/s); with copies 4.7 GB (about 49 tok/s) | serve the exact path only |
| Memory default | 2 slots x 262,144 tokens, 16 checkpoints, no YaRN | about 100 GB with the reserve; 3 slots do not fit with the drafter | 3 x 131,072 |
| MTP, vision | later milestones | the MTP semantics come from the shapes and must be verified; images need the new vision class | — |

## 4. Memory and speed

Bytes in GB (1e9). The budget of the decode milestone (2 slots, no drafter, no copies) is about 94 GB plus the 6 GB
growth reserve. The final configuration (copies, drafter, a 2 GB row cache) is about 109 GB of the 122 GB available.

| Item | GB |
|---|---|
| Routed experts, NVFP4 with scales | 67.95 |
| BF16 rest: GDN 4.17, attention 1.24, hyper-connections 1.27, shared experts and routers 0.60, embedding 1.27, lm_head 1.27, PLE 0.07 | 9.88 |
| Decode copies (phase B) | +3.10 |
| Per slot of 262,144 tokens: KV fp8 12 KB, raw indexer keys 3 KB, block keys 0.75 KB per token | 4.2 per slot |
| GDN and PLE state per slot | 0.12 |
| Checkpoint ring, 16 snapshots | 1.85 |
| CUDA context, allocator, pinned buffers, graph pool | about 5.8 |
| Drafter (later) | 2.3 to 3.8 |

Plain decode at 32k context streams about 1.33 GB of experts per token. The non-expert weights add 8.6 GB in BF16,
or 3.1 GB with the copies. At 230 GB/s this gives about 22 tok/s (exact path) and about 49 tok/s (copies). A verify
of M rows reads the union of their experts: about 39 distinct experts at M = 4 and 75 at M = 8 (uniform routing), so
the best draft length is shorter than the 27B's (about 1 to 3). The scheduler needs a cycle-cost model with the
distinct-expert count. The public vLLM build of this checkpoint on one DGX Spark decodes 16.9 tok/s.

Prefill of a 2048-token chunk reads every expert once: about 7 ms per layer, 400 ms per chunk for the MoE, a ceiling
of about 5k tok/s. The MoE math is about 0.7 ms per layer. A chunk of 8192 tokens raises the ceiling to about 14k
tok/s.

## 5. Interfaces and contracts

### 5.1 Modules

| Path | Holds |
|---|---|
| `engine/model/qwen4exp.py` | `Qwen4ExpConfig`, `Qwen4ExpState`, the reference modules, `Qwen4ExpForCausalLM` |
| `engine/weights/ple.py` | the n-gram hashing (numpy and torch), `NgramTable` (rows from the FP8 file) |
| `engine/weights/qwen4exp_loader.py` | `load_qwen4exp(path)`: the reference tree with stacked NVFP4 experts |
| `engine/weights/direct.py` | `read_tensors(..., into=)` |
| `engine/model/moe_kernels.py` | `KernelMoE`: routing, expert GEMMs, combine on the kernels of `csrc/moe.cu` |
| `engine/model/fast_qwen4exp.py` | the decode model on kernels (M1) |
| `engine/model/prefill_qwen4exp.py` | the prefill path (M1 / M2) |
| `engine/model/family.py` | `check_served(path)` today (a refusal of the family before the memory check); `family_of(path)`, the dispatch by `model_type`, at M1 |
| `csrc/moe.cu`, `csrc/moe_prefill.cu` | decode MoE kernels; the grouped GEMM path |
| `csrc/qsa.cu` | indexer prologue, block keys, scores, top-k, sparse decode attention |
| `csrc/hc.cu` | grouped norm, low-rank mix, injection |
| `csrc/{moe,qsa,hc}_bind.cpp` | the bindings of each kernel file; `bindings.cpp` calls `bind_moe`, `bind_qsa`, `bind_hc` |

### 5.2 The MoE contract

- `moe_route(logits bf16 [M, 520], ids i32 [M, 10], w f32 [M, 10], sg f32 [M])`: softmax in fp32, top-10 by (logit
  descending, index ascending), renormalized, cast to bf16; `sg` is the sigmoid of the shared-expert gate (row 512 of
  the stacked router). The reference uses `torch.sort(p, descending=True, stable=True)[..., :10]`.
- `moe_gate_up(x bf16 [M, 2560], ids, w_gu, sf_gu, gs, h bf16 [M * 10, 640])`, `moe_down(h, ids, w_dn, sf_dn, gs,
  yp f32 [M * 10, 2560])`, `moe_combine(yp, w, sg, shared bf16 [M, 2560], y bf16 [M, 2560])`.
- `KernelMoE.forward(x [M, 2560]) -> y [M, 2560]`, the shared expert included, no residual. Every output row is
  bit-identical for any set of rows in the launch. The skip flag of the draft early exit is honored.
- The scales `sf_gu` u8 [E, 204800] and `sf_dn` u8 [E, 102400] are in the swizzled layout of `gemm_nvfp4.cu`
  (`nvfp4_swizzle_sf` per expert, N = 1280 or 2560).

### 5.3 The QSA contract

- Per attention layer and slot: `k`, `v` fp8 [2, L, 256]; `ik` bf16 [L, 128] (raw keys, not normed, not roped);
  `ikb` bf16 [L / 4, 128] (block keys: fp32 mean of 4 raw keys, k_layernorm, rope at the block's first position);
  `rope_pos` int32 [L, 3].
- One stacked projection q | k | v | index = 13,952 rows per weight pass.
- `qsa_score(q_idx [B, T, 4, 128], ikb, pos) -> scores fp32 [B, 16, L / 4]`, `qsa_topk(scores) -> sel i32 [B, 16, 512]`
  ascending plus the count, `qsa_attn(q, k, v, sel, nsel, pos, gate) -> o`, GQA 24/2 with 24 key stripes.
- A row's selection depends only on the row. Decode, verify and prefill select with the same device code.

### 5.4 The hyper-connection contract

The dtype at each op follows HF's rounding (bf16 weights, fp32 math, one rounding per op):

1. `n = bf16(x_s * rsqrt(mean_s(x^2) + eps) * (1 + w))` per 2560-wide stream
2. `d = bf16(n W_down^T)`, `e = bf16(n W_inj^T)`
3. `a = bf16(silu(bf16(d / 4)))`, `u = bf16(a W_up^T)`, `mix = bf16(sigmoid(u))`
4. `p_s = bf16(mix_s * n_s)`, `y = bf16((((p0 + p1) + p2) + p3) / 4)`, `inj_s = 2 bf16(sigmoid(e_s / 4))`
5. after the sublayer: `o = bf16(acc)`, `x4_s = bf16(x4_s + bf16(o * inj_s))`

Kernels: `hc_norm` (the grouped norm), `hc_mix` (the up product, sigmoid, the mean), `hc_post` (the injection; later a
skinny-GEMM epilogue mode). W_down and W_inj stack into one [328, 10240] skinny pass.

### 5.5 The PLE contract

- Ids: `hist = cat(ctx, tokens)`, `p1 = hist[1:-1]`, `p2 = where(p1 == eos, eos, hist[:-2])`,
  `a2 = cur m0 ^ p1 m1`, `a3 = a2 ^ p2 m2`; heads 0-7 = `a2 % size_h + off_h`, heads 8-15 = `a3 % size_h + off_h`.
  The context of a row is the two tokens before it in the slot's history; image tokens hash as 248056.
- `NgramTable.rows(ids int64 [n]) -> u8 [n, 160]` reads the rows from the checkpoint file; the row r is in shard
  r // 2,500,012 at byte `base + data_offset(shard) + (r % 2,500,012) * 160`. The value is `bf16(e4m3 * scale)`.
- The graph reads the rows of its T tokens from a fixed buffer u8 [B, T, 16, 160] that the host fills before the replay
  (`before_replay(tokens)`).
- The PLE conv state bf16 [B, 10240, 9] joins the checkpoint ring and the session files.

### 5.6 The state and serving hooks (M1 / M4)

`FastState`-like states expose `kv_tensors()` (one row per position: k, v, ik, and the block keys as ratio-4 rows) and
`snap_tensors()` (conv, rec, ple_conv). The checkpoint ring, `_copy_kv` and the session files iterate over these. The
27B's order and fingerprint stay identical. `startup_bytes` asks the family for the state shapes.

## 6. Milestones

| Milestone | Proves | Hours (estimate) |
|---|---|---|
| A. Reference and weights | the tiny model matches transformers on the CPU (one-shot, split, stepwise, with a cache, with eos, with an image); the loader reads a synthetic checkpoint in the nvidia layout; `NgramTable` reads rows | 40 |
| K1. MoE kernels | `test_moe.py`: accuracy, row identity, graphs, the skip flag; the grouped GEMM compiles and `test_moe_prefill.py` passes; a bench at 200 GB/s or more at 75 distinct experts | 80 |
| K2. QSA, HC, GDN kernels | `test_qsa.py` (selection equals torch, T = 1 equals T = 16, attention equals SDPA with the mask), `test_hc.py`, the GDN gate flag with both gates | 60 |
| M1. Golden decode | greedy tokens of three prompts of 2051 tokens or fewer equal the reference's, one slot, no speculation | 60 |
| M2. Prefill and long context | chunks, QSA prefill, three slots, passkey and perplexity at long context | 120 |
| M3. Speculation | the MTP drafter; golden with speculation on and off is token-identical | 75 |
| M4. Checkpoints and sessions | the ring and the disk cache hold the new state; `scheduler_check` on this model | 25 |
| M5. Server, copies, docs | family dispatch, flags and defaults, the INT / NVFP4 copies under the gate, the docs | 40 |

Phases A, K1 and K2 run in parallel in three git worktrees (`moe-model`, `moe-moe`, `moe-qsa`), each with its own
kernel build. The 27B's tests must keep passing in every worktree.

## 7. Open questions for the user

1. The default size: 2 slots x 262,144 tokens (recommended) or 3 x 131,072?
2. The exact BF16 path as the first milestone, with the requantized copies in phase B under the perplexity gate?
3. A disk-resident n-gram table with a small row cache, and no resident 4-bit table (25.6 GB)?
4. May the prefill chunk become 4096 or 8192 tokens for this family, if the bench confirms the gain?
5. Vision and the MTP drafter as later milestones?
6. May the MTP work read the vLLM / SGLang `qwen4_exp` code on the web, to confirm the drafter's wiring?

## 8. Risks

- The CUTLASS ptr-array block-scaled GEMM is unverified on sm_121a: compile time, shared memory at 128 x 128 x 256,
  groups with 0 rows, device-built arguments. The fallback is a loop over experts.
- Cold rows of the n-gram table slow a cycle until the row cache holds the hot set. The hit rate is unmeasured.
- The MTP semantics are inferred from the shapes. A wrong guess costs acceptance rate, not correctness.
- Top-k near-ties can differ from HF on real text; the engine's own paths agree with each other by construction.
- The decode copies of the hyper-connection matrices (K = 320 for W_up) need a format of their own.

## 9. Status log

- **2026-10-24.** The checkpoint is downloaded and verified (11 shards, 132.7 GB). Milestone A is merged: the reference
  model is bit-identical to transformers in fp32 on the tiny model (without a cache, with a cache, with an image), the
  loader stacks the experts at load time, and the n-gram table reads from disk. On the real checkpoint, the reference
  loads 77.8 GB in 34 s and gives coherent greedy continuations (`tools/qwen4exp_reference.py`), at about one second
  per token in plain PyTorch. The checkpoint's scales confirm the kernel contract: gate and up share their global scale
  in every expert, and each layer and projection has one input scale. The hot-set tool measured the n-gram table's LRU
  hit rate at 51% to 68% on WikiText and 83% to 90% on TinyStories for caches of 0.5 to 5 GB, so cold rows stay a real
  cost (risk 2).
- **2026-10-24, K1 merged.** The MoE decode kernels (`csrc/moe.cu`) stream 208 GB/s at 10 distinct experts and 238 GB/s
  at 75; a full MoE layer (router, routing, shared expert, experts, combine) takes 196 us at one row, 935 us at 8 rows
  and 1635 us at 16 rows, so 9.4 / 45 / 78 ms over 48 layers. The CUTLASS SM120 grouped NVFP4 GEMM compiles and runs
  on sm_121a (`csrc/moe_prefill.cu`): a layer's MoE prefill takes 8.8 ms at 2048 tokens, 11.3 ms at 4096 and 16.6 ms
  at 8192, which is a MoE ceiling of 4.8k / 7.5k / 10.3k tok/s. The scheduler's raster swizzle (8) matters: without
  it the grouped GEMM runs 2.3 to 3 times slower. The routing weights are bf16-rounded but kept in f32; the decode
  and prefill paths differ by about 2% (the prefill quantizes the activations to NVFP4). Known limit: the split-K
  counters of the skinny GEMM are global, so two split-K launches must not overlap on different streams. memcheck,
  racecheck and synccheck are clean (synccheck needs `--num-cuda-barriers 4096` on the grouped GEMM). The startup
  self-test covers the BF16 skinny format, the MoE decode path and the grouped GEMM.
- **2026-10-24, K2 merged.** The GDN kernels take a `gate_act` flag (sigmoid for this family; the 27B path keeps its
  bits). The decode attention runs GQA 24/2 as well as 24/4 (the tensor-core core moved to `csrc/attn_tc.cuh`, with a
  list mode for the sparse path). The QSA kernels (`csrc/qsa.cu`): the selection equals a stable sort and torch.topk
  with forced ties up to 65536 blocks; `qsa_attn` over 2051 keys takes 9.3 us at T = 1 and 72 us at T = 8 (227 to
  235 GB/s) at any context length; `qsa_score` takes about 24 to 50 us at 65k blocks; `qsa_topk` takes about 70 us
  at 65k blocks (one block per row: about 0.85 ms per token over 12 layers at 262k, the first thing to speed up).
  The hyper-connection kernels (`csrc/hc.cu`): `hc_post` and the grouped norm are bit-exact against transformers;
  `hc_mix` streams W_up at about 200 GB/s (33 us at M = 1); a full gated residual takes 14 / 40 / 74 us at M = 1 / 8 /
  16. The down | inject product runs on `bf16_gemv` until the BF16 skinny GEMM replaces it (torch's bf16 matmul is
  not row-invariant). The host row reader `ops().ple_gather` reaches about 170k cold rows per second at 128 rows
  per call and 370k to 410k at 4096 rows per call on the NVMe drive (O_DIRECT and plain pread within 10%). memcheck,
  racecheck (filtered to the engine's kernels) and synccheck are clean. The startup self-test covers 22 paths. Open:
  the model must record `rope_pos` rows before `qsa_block_keys` when an image is in the context; `qsa_topk` at long
  context; the thread count of the row reader (small warm batches want 1 to 2 threads, cold batches 64).
- **2026-10-24, M4 round 1 merged (the family and the state hooks).** `engine/model/family.py` dispatches by `model_type`:
  `Family(load, new_state, state_bytes, fixed_bytes, prefill, prepare_prefill, defaults, supports, before_replay, feed_buffer)`;
  the qwen3_5 family wraps today's code, the qwen4_exp family imports `fast_qwen4exp` / `prefill_qwen4exp` lazily (defaults
  2 slots x 262,144, 16 checkpoints, no drafter, no images, no YaRN; fixed bytes 85.5e9 with the weights). The states expose
  `kv_tensors()` (per-position tensors, time on dim 2; a tensor with L / r rows is one row per r positions) and
  `snap_tensors()`; the checkpoint ring, `_copy_kv`, the slot reset and the session files iterate over them. The scheduler
  takes `prefill_fn`, `prepare_fn`, `before_replay(slot, tokens)` (called right before each replay with the tokens of the
  rows) and `feed_buffer`; the server's `--slots`, `--max-seq-len` and `--spec` default to the family's values. Proof of
  identity for the 27B: two `tests/scheduler_check.py` runs, before and after, wrote the same session files, hash for
  hash (fingerprint 9af270967b6a20e3), and the whole suite passes (592 tests).
- **2026-10-24, M1 merged: the decode model runs the real checkpoint.** `engine/model/fast_qwen4exp.py` (FastState4,
  Bf16Linear / StackedBf16Linear, KernelGDN4, KernelQSAAttention, KernelPLE in torch ops, KernelMoE, the gated residuals,
  FastQwen4Exp with forward / verify / commit / feed_buffer). On the real checkpoint (`tests/golden_qwen4exp.py`, one slot,
  CUDA graph): two of three golden prompts equal the reference for 32 tokens; the third diverges at token 12 where the
  reference's bf16 margin is 0.125 and its own fp32 logits pick the engine's token (a near-tie; the rule is now a margin
  below 0.2). The perplexity gate passes: 2.3274 (decode path, token by token) against 2.3335 (the reference, chunked)
  over 8 windows of 2048 WikiText tokens, ratio 0.9974 (one window cannot resolve 0.5%: the reference moves 2.4% with
  fp32 expert math). A decode step takes 46.0 ms at 1k context (21.8 tok/s; the plan expected 44); a verify of 8 rows
  64.2 ms; 78.1 GB on the GPU with a 4096-token slot. The profile: BF16 skinny GEMMs 61% (36.5 ms), experts 18%,
  hyper-connections 11% + the GEMV 7% (about 10.7 ms: hc_mix streams at about 100 GB/s and bf16_gemv at 150: the
  cheapest speed-up is to move them to the skinny GEMM), GDN 2%, attention and the indexer under 1%. Limits: the slot
  length must be a multiple of 16; the kernels fix hidden 2560, hc 4, lowrank 320 and 512 blocks.
- **2026-10-24, M2 round 1 merged (the prefill pieces).** `csrc/qsa_prefill.cu`: the indexer scores all rows of a chunk in
  one launch (0.13 / 0.75 / 4.6 ms at prefixes 0 / 16k / 128k for 2048 rows, bit-identical to the decode scores) and the
  union kernel attends each group of 4 rows over the union of their blocks (2.5 / 13.3 / 15.8 ms per layer at the same
  prefixes; masked FlashInfer takes 103 ms at 128k, so the union is the default path). `engine/model/prefill_qwen4exp.py`
  holds the layer functions (tested against the reference modules at the real dims) and the chunk loop; a 4-layer assembly
  matches the reference within 1.2%. The n-gram feeder (`PleFeeder`, a set-associative C++ row cache of 8 ways): a cold
  2048-token chunk takes about 78 ms, the warm decode path 0.03 ms, a 2 GB cache hits 48% of the distinct rows on WikiText.
  The chunk bench (random weights) gives 2489 / 2918 / 3155 tok/s at 2048 / 4096 / 8192 tokens, so the chunk becomes
  4096 (+17%; 8192 adds 8% and holds the lane for 3 s). `csrc/hc_prefill.cu` cut the hyper-connection prefill from 763 to
  167 ms per chunk.
- **2026-10-24, M4 round 2 merged: the server serves Qwen3.8-Flash-Next.** The qwen4_exp family loads the decode model,
  attaches the n-gram feeder (`--ngram-cache-gb`, default 1.0 = 6.25M rows), prefills in 4096-token chunks with the slot's
  previous two tokens, and fills the n-gram row buffer before each replay. `tests/scheduler_check_qwen4exp.py` on the real
  checkpoint: startup 36 s; MemAvailable falls 94.7 GB after the build (state 8.7, row cache 1.1, the rest 84.9; the fixed
  bytes of the family are the measured 87e9); the three golden prompts give 32 tokens equal to the reference through the
  prefill, at 18.7 to 19.3 tok/s with a TTFT of 0.1 to 0.3 s; a 4676-token second turn reuses 4655 tokens (TTFT 0.128 s
  against 1.746 s from scratch); a second slot keeps the first conversation cached; a 7398-token conversation restores
  from disk in 0.161 s with identical output; a 6375-token prompt prefills in two chunks at 1624 tok/s. The server
  (`--port 8011`): ready in 42 s, 88.5 GB allocated, streaming chat, `/v1/messages`, `reasoning_effort`, `/metrics` all
  answer; 19 to 20 tok/s plain decode (the per-step host row read with 23% cache misses costs the difference to the 46 ms
  graph step). The 27B is byte-identical (the session digest d4f42e84... of round 1 again). `NOT_SERVED_YET` is empty.
  Open: the decode speed (the row reads; the hyper-connections on the skinny GEMM), a row prefetch between the scheduler's
  chunks, about 5.6 GB of growth during the first long prompts (the warm-up covers it; the reserve is 6 GB), and the
  drafter's wiring (the ring's drafter hidden of 10240, the drafter's ik / ikb / rope_pos rows in the sessions, a k model
  from the distinct-expert count).
- **2026-10-24, M3 merged: exact speculation.** `engine/spec/mtp4.py` (Mtp4, Mtp4State, Mtp4Cycle) is the drafter of the
  family: the wiring is confirmed from vLLM's source (the projected embedding enters every stream as a plain add; one norm
  over the whole 10240-wide hidden, then the shared fc_hidden per stream; a per-stream norm in the mixer; the target's
  lm_head as the draft head over the 64k draft vocabulary). The FP8 block experts become NVFP4 copies on KernelMoE
  (1.41 GB; requantized once in 7.5 s, then 0.3 to 0.9 s from `~/.cache/colinfer/drafter/quant4-*.safetensors`). The
  drafter selects its sparse blocks on every row with the target's kernels (vLLM reuses step 0; both are exact under the
  accept rule). On the real checkpoint (`tests/golden_qwen4exp_spec.py`): the tokens with speculation equal plain decode
  for k = 1, 2, 3, 4 on the three golden prompts and a 1024-token WikiText prompt, greedy and sampled. Against the
  46.6 ms plain step: k = 2 gives 2.30 tokens per cycle in 59.5 ms on the golden prompts (38.7 tok/s GPU, 1.80x) and 2.10
  in 59.7 ms on WikiText (35.1 tok/s, 1.63x); k = 3 reaches 1.94x on code and 1.61x on prose; k = 4 adds nothing. The
  cost model for the scheduler: cycle_ms = 41.4 + 0.577 x distinct experts + 2.92 x k, with about 10 + 5.5 k distinct
  experts per layer per verify. Recommendation: k = 2 by default, k up to 3 by acceptance. The drafter adds 1.95 GB and
  0.355 GB per 262k-token slot. Remaining: the scheduler wiring (Mtp4State in the ring copies and the sessions, the
  fixed bytes, Supports.mtp, the k model), the low-rank draft head, the early-exit tuning.
- **Next:** M2 (prefill on the real checkpoint and long context) is in progress; then the drafter's scheduler wiring and
  the documentation pass.

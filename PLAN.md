# Plan: a single-user inference engine for DGX Spark

Target: run **Qwen3.8-27B** (and later **DeepSeek V4 Flash**) on one DGX Spark at the
fastest prefill and decode this chip can physically deliver, for one user making at most
three concurrent requests.

Written 2026-09-30 against the actual machine (`spark-44de`), the software already on it,
and published measurements from other GB10 units. Every target below is derived from a
byte or FLOP count (`tools/roofline.py`), not from vibes.

---

## 0. Executive summary

1. **Decode on GB10 is a memory-bandwidth problem and nothing else.** The chip reads
   ~230 GB/s in practice (273 GB/s spec). A 27B dense model at NVFP4 is ~15 GB per token,
   so base decode tops out near **14-15 tok/s** no matter how clever the kernels are.
   The best public stacks already reach 85% of that. Our kernel goal is **>=90%**.
2. **Everything above ~15 tok/s comes from speculative decoding.** Verifying 5 tokens
   costs about the same as generating 1. Qwen3.8-27B ships an MTP head. Public results:
   MTP 30-36 tok/s, DSpark 43, EAGLE 42, a 2B parallel drafter 48-69 on code.
   Target: **35-45 tok/s** with the shipped MTP head, **50+** with a better drafter.
3. **Prefill is compute-bound on `mma.sync` tensor cores.** 27B dense = ~49 GFLOP/token.
   Tuned CUTLASS reaches 356 TFLOPS NVFP4 / 188 TFLOPS FP8 on this chip; vLLM's prefill
   is ~1,800 tok/s (about 88 TFLOPS effective). Target: **2,500-3,500 tok/s** for prompts
   up to 8k, i.e. TTFT under a second for a 2k prompt.
4. **Single-user lets us delete most of what makes vLLM complex.** No paged KV, no radix
   tree, no fairness scheduler, no tensor parallelism, no multi-tenant memory accounting.
   Three fixed slots with contiguous KV, one CUDA graph per decode shape, and a prefix
   checkpoint cache for multi-turn chat.
5. **Stack: Python + PyTorch as allocator and glue, hand-written CUDA C++ for the decode
   hot path, CUTLASS C++ for prefill GEMMs, Triton for fusions and Gated DeltaNet.**
   With CUDA graphs the host language is irrelevant to decode speed.
6. **Order: Qwen3.8-27B first, DeepSeek V4 Flash second.** The 27B is dense, has an
   official NVFP4 checkpoint, an MTP head, and a 262k context. V4 Flash is a 284B MoE with
   shared-KV attention, compressed sparse attention, a lightning indexer, hyper-connections
   and hash routing, and it does not fit in 128 GB at 4 bits. It is a 2-3x bigger project
   that reuses most of the 27B engine's infrastructure.
7. **Operational constraint right now:** the DeepSeek V4 Flash vLLM container holds
   112 GB of the 121 GB unified memory. Developing a 27B engine (~35-45 GB) requires
   stopping it (`DeepSeek-v4-Flash-One-DGX-Spark/stop.sh`). Over-committing this box
   hard-powers-off instead of raising OOM.

---

## 1. The machine (measured 2026-09-30)

| Item | Value | Implication |
|---|---|---|
| GPU | NVIDIA GB10, compute capability **12.1 (sm_121)**, **48 SMs**, 6,144 CUDA cores, boost ~2.4-2.5 GHz | Consumer-Blackwell class, not B200 class |
| L2 | **24 MB** | Weight streaming never fits; activations and KV tiles do |
| Shared memory | 128 KB/SM, **99 KB max per block** | Forces K=64 CTA tiles for block-scaled GEMM; FA tiles for head_dim 256 are tight |
| Tensor cores | 5th gen via **`mma.sync` only**. Native NVFP4/MXFP4 block-scaled MMA (`kind::mxf4nvf4`, m16n8k64), FP8 `kind::f8f6f4`. **No tcgen05, no TMEM, no WGMMA, no thread-block clusters, no TMA multicast** | Write Hopper/Ampere-style kernels, not SM100 kernels. CUTLASS SM120 collectives are the model |
| Peak (measured by others) | NVFP4 dense ~356 TFLOPS (tuned CUTLASS 4.4), FP8 ~188, BF16 ~100-125, FP32 31 | Prefill at NVFP4 activations is worth 2x over FP8 |
| Memory | 128 GB LPDDR5x unified, 273 GB/s spec. **Measured GPU read: 231-234 GB/s typical, 238-263 GB/s on some units**, varies with memory state since boot | Measure this unit (Phase 0) and use it as the denominator for every efficiency number |
| CPU | 10x Cortex-X925 + 10x Cortex-A725, aarch64 | Fast enough that Python overhead is not a decode concern once graphs are used |
| Driver / CUDA | 580.173.02, CUDA 13.0 toolkit (nvcc 13.0.88) | ptxas knows `sm_121a` |
| Disk | 2.8 TB free NVMe | Keep BF16, FP8 and NVFP4 copies of the 27B |
| Host-to-device copies | Pageable copies up to 52x slower than pinned | Pin host buffers when loading weights |

Software already proven to work natively on this box (`~/Projects/vllm/.venv`):
torch 2.13.0+cu132, triton 3.7.1, flashinfer 0.6.15, nvidia-cutlass-dsl 4.6.0, cudnn 9.20,
transformers 5.14.1. The nanoGPT venv has torch 2.12.1+cu130.

### sm_121 toolchain caveats (all current as of today)

- **Triton**: bundled ptxas may not know `sm_121a`. Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
  until Triton 3.8. A stale `~/.triton/cache` has produced silently garbled tokens on sm_121;
  wipe it after any upgrade. Some Mamba/GDN Triton kernels hit "illegal instruction" in async
  mode on SM121: test FLA kernels early.
- **CUTLASS**: SM120/121 block-scaled GEMM exists in C++ (Example 79, `mma.sync` block_scale,
  TN layout only, tiles up to 128x128x128, cluster 1x1x1). **The Python CuTe DSL refuses FP4 on
  sm_120/121** (`admissible_archs = [sm_100a]`), so prefill GEMMs are C++ CUTLASS, not DSL.
  A CUTLASS FP4 kernel compiled for the wrong ISA has produced wrong answers with **no CUDA
  error**: ship a numerical self-test for every GEMM path at startup.
- **FlashInfer**: FA2 prefill/decode, RoPE, sampling, FP8/FP4 quant work on SM121. Its
  GDN kernels are built with sm90a flags (do not load), there is no BF16 GEMM backend, and
  its auto-dispatch checks `minor == 0` so FP4 paths are not auto-selected. Usable as a
  stopgap for attention; not a foundation.
- **PyTorch wheels** carry `sm_120` SASS plus PTX; sm_121 is binary compatible. Build our own
  kernels with `-gencode arch=compute_121a,code=sm_121a` (block-scaled MMA needs the `a` target).
- **Memory**: no `nvidia-smi` memory readout; over-commit of the shared pool can hard power-off
  the box. Cap total engine allocation explicitly and never rely on the OOM path.

---

## 2. The targets

### 2.1 Qwen3.8-27B (`Qwen/Qwen3.8-27B`, released 2026-08-14)

Same architecture as Qwen3.5-27B and Qwen3.6-27B (configs are byte-identical apart from the
transformers version), so everything here applies to all three.

| Field | Value |
|---|---|
| Layers | **64**: pattern `[GDN, GDN, GDN, FullAttn] x 16` (48 Gated DeltaNet + 16 gated full attention) |
| Hidden / MLP | 5120 / 17408, SwiGLU, no MoE |
| Full attention | 24 Q heads, **4 KV heads**, **head_dim 256**, output gate (sigmoid), partial RoPE on 64 of 256 dims, rope_theta 1e7, interleaved mRoPE |
| Gated DeltaNet | 16 key heads x 128, 48 value heads x 128, conv1d kernel 4, SSM state in **fp32** |
| Vocab | **248,320**, untied embeddings and lm_head |
| Context | 262,144 native, YaRN to ~1M |
| MTP | **1 full-attention block shipped** (`mtp.*`), trained multi-step |
| Vision | 27-layer ViT included. Ignore for text-only serving |
| Checkpoints | BF16 `Qwen/Qwen3.8-27B` (~55 GB), FP8 `Qwen/Qwen3.8-27B-FP8` (~28 GB), **NVFP4 `nvidia/Qwen3.8-27B-NVFP4`** (ModelOpt, ~18 GB), GGUF `unsloth/Qwen3.8-27B-GGUF` |

Parameter budget (text only, from `tools/roofline.py`):

| Component | Params | Note |
|---|---|---|
| MLP (64 layers) | 17.11 B | 70% of every decode step's bytes |
| GDN projections (48) | 5.56 B | |
| Full attention (16) | 1.68 B | |
| lm_head | 1.27 B | Read in full every step: second-largest term |
| Embeddings | 1.27 B | One row per token, negligible |
| MTP block | 0.42 B | Only read when speculating |

### 2.2 DeepSeek V4 Flash (`deepseek-ai/DeepSeek-V4-Flash-0731`)

| Field | Value |
|---|---|
| Size | **284B total / 13B active**, 43 layers, hidden 4096 |
| MoE | 256 routed + 1 shared experts, 6 routed per token, expert width 2048, `sqrtsoftplus` router, `noaux_tc`, **hash routing in the first 3 layers**, routed scaling 1.5 |
| Attention | "Shared-KV multi-query": 64 Q heads, **1 KV head, head_dim 512, K = V** (one 512-d latent per token), q low-rank 1024, grouped low-rank output (8 x 1024), RoPE on 64 dims, per-head attention sinks |
| Sparsity | Layers 0-1 pure sliding window (128). Layers 2-42 alternate **CSA** (4:1 compression + lightning indexer top-512, 64 index heads x 128, FP4 indexer) and **HCA** (128:1 compression, dense over compressed). Every layer also keeps a 128-token raw window |
| Residual | **mHC hyper-connections**, 4 streams, 20 Sinkhorn iterations |
| Draft | **3-layer DSpark module** in the 0731 checkpoint (block size 5) |
| Context | 1,048,576 (YaRN x16 over 65,536) |
| Native dtypes | Experts **FP4 (QAT)**, everything else FP8 e4m3 128x128 blocks |

Memory fitting is the first-order problem: 277B expert params at NVFP4 is ~147 GB. The
stack running on this box today (EXL3 3.0 bpw experts, REAP-pruned to 216 experts, ~99 GB)
gets 20-24 tok/s plain and **44-47 tok/s with DSpark**. Any custom engine needs ~3 bpw
experts to fit.

### 2.3 Roofline-derived ceilings (bandwidth 230 GB/s measured, 273 spec)

Qwen3.8-27B, batch 1, 8k context, FP8 KV, FP8 lm_head:

| Weights | GB/token | tok/s @273 | tok/s @230 |
|---|---|---|---|
| BF16 | 51.8 | 5.3 | 4.4 |
| FP8 | 26.4 | 10.3 | 8.7 |
| **NVFP4** | **15.5** | **17.6** | **14.8** |
| INT4 g128 | 14.5 | 18.8 | 15.9 |

Where a 70 ms decode step goes at NVFP4 (230 GB/s):

| Term | Bytes | ms |
|---|---|---|
| Backbone weights | 13.7 GB | 59.6 |
| lm_head (FP8) | 1.27 GB | 5.5 |
| KV read at 8k (FP8, 16 layers) | 0.26 GB | 1.1 |
| GDN state read+write (fp32) | 0.30 GB | 1.3 |
| ~400 kernel nodes in a graph | - | ~1.5 |
| Sampling + host readback | - | ~0.5 |
| **Total** | | **~69.5 ms = 14.4 tok/s** |

KV grows at 32 KB/token (FP8): 1 GB at 32k, 4.3 GB at 128k. At 128k the KV read alone adds
19 ms per step, so long-context decode drops to ~11 tok/s. GDN state is fixed at 151 MB per
sequence regardless of context, which is the hybrid architecture's gift.

Prefill (linear layers 48.7 GFLOP/token; attention adds 1.6 GFLOP/token at 8k, 25.8 at 128k):

| Sustained TFLOPS | tok/s @8k | tok/s @32k | tok/s @128k |
|---|---|---|---|
| 100 (vLLM today, approx.) | 1,990 | 1,810 | 1,340 |
| 150 | 2,980 | 2,720 | 2,010 |
| 200 | 3,970 | 3,630 | 2,680 |

DeepSeek V4 Flash (216 experts, active bytes/token): experts at 3 bpw + attention at FP8 =
8.9 GB (ceiling 26 tok/s @230), experts at 3 bpw + **attention at FP4** = 6.4 GB (35 tok/s).
The attention weights (5.25B params, FP8 today) are now the single largest byte consumer per
token in that model, which is the main untapped lever there.

### 2.4 Public baselines on one DGX Spark, Qwen3.8-27B (what we must beat)

| Engine | Quant | Prefill tok/s | Decode | Decode + spec |
|---|---|---|---|---|
| vLLM 0.27.1 | NVFP4 | 1,794 | 11.5 | MTP 22.0 |
| vLLM 0.27.1 | FP8 | 1,914 | 8.2 | - |
| SGLang (flashinfer_cutlass FP4) | NVFP4 | - | **12.32** (85% of ceiling) | MTP 35.7, DSpark 42.9, EAGLE 41.6, DFlash2 2B draft 54-69 (code) / 28 (prose) |
| llama.cpp b10423 | UD-Q4_K_XL | 839 | 11.6 | draft-mtp ~18 |
| Ollama 0.32 | Q4 | 731 | - | MTP 26.5 |

### 2.5 Our targets (this unit, after Phase 0 measurement; numbers assume 230 GB/s)

| Metric | Target | Stretch |
|---|---|---|
| Decode, no spec, 8k ctx | **>=13.5 tok/s** (>=90% of measured BW) | 14.5 |
| Decode, MTP spec, code/chat, T=0 | **>=35 tok/s** | 45 |
| Decode, better drafter, code | - | 55+ |
| Prefill 2k prompt | **>=2,500 tok/s** (TTFT <= 0.8 s) | 3,500 |
| Prefill 32k prompt | >=2,000 tok/s (TTFT <= 16 s) | 2,800 |
| Multi-turn TTFT (prefix hit) | <= 150 ms + new tokens / prefill rate | |
| Context | 128k in Phase 2, 262k by Phase 5 | |
| 3 concurrent requests | decode per-request >= 0.8x of single | |
| Quality | NVFP4 perplexity and tool-call score within noise of vLLM on the same checkpoint | |

---

## 3. Design principles for a one-user engine

What we drop relative to vLLM/SGLang, and why it is safe here:

| Dropped | Replaced by |
|---|---|
| Paged KV cache, block tables | **3 fixed slots**, contiguous KV per slot, preallocated to max context |
| Radix-tree prefix cache | Per-slot **prefix checkpoint**: token ids + KV length + GDN state snapshots at turn boundaries and every 4k tokens |
| General scheduler, preemption, swapping | A loop: admit to a free slot, run one prefill chunk if any slot is prefilling, run one decode step for all decoding slots |
| Tensor/pipeline parallel, NCCL | Nothing. One GPU |
| Dynamic batching across many shapes | **One CUDA graph per (active slots in 1..3, draft length in {0,k})**, ~6-8 graphs |
| Generic quant dispatch (AWQ, GPTQ, Marlin, ...) | One weight format: **NVFP4**, plus FP8 for lm_head and sensitive layers |
| Python on the per-step path | Zero. A step is: write slot metadata into static device buffers, replay graph, read back accepted token ids |

What we keep: chunked prefill (so a new request's long prompt does not freeze the other
slots), streaming, an OpenAI-compatible API (so `model-benchmarks` and `llama-benchy` work
unchanged), proper sampling, tool-call and thinking parsing.

---

## 4. Architecture

```
                 HTTP (OpenAI API, SSE)            engine/server/
                          |
                 Request queue + 3 slots           engine/runtime/scheduler.py
                          |
        +-----------------+------------------+
        |                                    |
   Prefill path (chunked, compute-bound)  Decode path (graph replay, BW-bound)
   - CUTLASS SM120 NVFP4 GEMM (W4A4)      - NVFP4 GEMV / skinny GEMM (M<=16)
   - FA2 prefill, head_dim 256, GQA 6:1   - flash-decoding over FP8 KV
   - FLA chunked Gated DeltaNet           - fused GDN recurrent step
   - fused RMSNorm + act-quant            - fused norm/residual/SiLU
   - writes FP8 KV + GDN checkpoints      - FP8 lm_head + fused sampling
        |                                    |
        +-----------------+------------------+
                          |
            Static memory plan (allocated once)    engine/kv/
            weights | KV slots | GDN states | checkpoints | workspace | graphs
```

### 4.1 Weights and formats

- **Primary format: NVFP4** (E2M1 values, E4M3 scale per 16 elements, FP32 per-tensor
  scale, plus a per-tensor input scale for activation quantization). Load directly from the
  ModelOpt checkpoint `nvidia/Qwen3.8-27B-NVFP4`: `weight` (uint8, [N, K/2]),
  `weight_scale` (e4m3, [N, K/16]), `weight_scale_2` (fp32), `input_scale` (fp32). Verify
  at load time which modules ModelOpt left unquantized (typically lm_head, embeddings,
  GDN conv, norms).
- **lm_head: FP8** (1.27 GB). Try NVFP4 lm_head in Phase 6 (saves ~2.5 ms/step, 4%).
- **Embeddings: FP8 or BF16**, gathered by row.
- GDN `A_log`, `dt_bias`, conv weights, all norms: fp32/bf16.
- Repack on load into the layouts the kernels want (e.g. interleave gate/up rows so one
  GEMV emits `silu(gate) * up` directly; tile-swizzle for the CUTLASS GEMM) and cache the
  repacked file on disk so startup is a pinned-memory read, not a conversion.
- **FP8 and BF16 copies are kept on disk as references**, loaded only by the correctness
  harness. If W4A4 prefill turns out to hurt quality, the fallback is an FP8 weight copy
  for prefill only (27 GB extra, affordable), with NVFP4 for decode.

### 4.2 Memory plan (Qwen3.8-27B, 3 slots, 128k each)

| Region | Size |
|---|---|
| NVFP4 backbone + MTP | ~14.3 GB |
| lm_head FP8 + embeddings FP8 | ~2.6 GB |
| KV, FP8, 16 layers, 3 x 128k | 12.6 GB (25 GB at 262k) |
| GDN state, fp32, 3 slots | 0.45 GB |
| GDN checkpoints, 32 x 151 MB ring | 4.8 GB |
| Prefill workspace (4k chunk: MLP intermediate 4096 x 17408 x 2 B and attention) | ~1 GB |
| CUDA graphs, decode buffers, logits (248k x 16 x 2 B) | ~1 GB |
| **Total** | **~37 GB** (50 GB at 262k) |

Set a hard cap at allocation time; never grow at runtime.

### 4.3 Decode step (the kernel list)

For M = (active slots) x (1 + draft length) query rows, M <= 16:

1. **Fused RMSNorm + activation quantization** (one kernel per layer, output bf16 and,
   for the skinny-GEMM path, NVFP4 activations with per-16 scales).
2. **NVFP4 GEMV (M <= 4)**: CUDA cores. Each CTA owns a band of N rows; 16-byte vector
   loads of packed FP4 and scales, dequant in registers, FMA against activations resident
   in shared memory, split-K across warps, one reduction. Fusions: residual add in the
   epilogue, `silu(gate) * up` in the epilogue of the interleaved gate/up GEMV. Acceptance
   bar: **>=90% of measured read bandwidth** on the 5120x17408 shape, measured in isolation.
3. **NVFP4 skinny GEMM (5 <= M <= 16)**: same memory access pattern but the math on
   `mma.sync` (bf16 m16n8k16 after in-register dequant, Marlin-style), because at M=16 the
   FMA count (~780 GFLOP per step) would take ~40 ms on CUDA cores and stop being hidden
   behind the weight stream. This is the speculative-decode verify path.
4. **Full-attention decode (16 layers)**: flash-decoding split-KV kernel over FP8 KV.
   GQA 6:1 means each KV tile is loaded once and used by 6 query heads and all M rows.
   Fuse partial RoPE on the 64 rotary dims, the sigmoid output gate, and the K/V FP8
   quantization of the new tokens' entries. Stopgap: FlashInfer's FA2 decode (works on sm_121).
5. **GDN recurrent step (48 layers)**: one fused kernel per layer: conv1d step, SiLU,
   L2-normalize q/k, decay `exp(g)`, delta rule `S = S*g + beta*(v - S k) k^T`, `o = S q`,
   gated RMSNorm, then the out-proj GEMV. The state is 151 MB per slot; reading and writing
   it costs ~1.3 ms, fine. For spec decode the kernel runs the M steps sequentially per slot.
   **Rollback**: snapshot the state before the verify step; after acceptance of n tokens,
   restore and replay n steps from the cached per-token q/k/v/g/beta (no weight traffic).
   Reference: FLA `fused_recurrent_gated_delta_rule` (Triton).
6. **lm_head FP8 GEMV** over 248k x 5120 for the M rows, bf16 logits.
7. **Fused sampling**: temperature, top-k, top-p, min-p, repetition penalty, Philox RNG
   per slot, and for spec decode the acceptance test (greedy match or rejection sampling)
   plus bonus token. Writes accepted ids and counts to a device buffer.
8. **Draft step (spec decode)**: the MTP block run k times autoregressively on the last
   hidden state (its own tiny KV), or the n-gram lookup kernel. Lives inside the same graph.

The whole step is **one CUDA graph**. Host work per step: copy ~1 KB of slot metadata
into static buffers, launch, one `cudaStreamSynchronize` (or event poll), read back
<=48 ints. Python overhead ~100 us against a 70 ms step.

### 4.4 Prefill path

1. **Activation quantization to NVFP4** fused into RMSNorm output (per-16 scales,
   per-tensor input scale from the checkpoint).
2. **NVFP4 x NVFP4 block-scaled GEMM** via CUTLASS SM120 collectives (Example 79 lineage):
   TN layout, 128x128x64 or 128x128x128 tiles (99 KB SMEM), cooperative schedule, epilogue
   fusions for residual add and SwiGLU. Target **>=180 TFLOPS sustained** on the MLP
   shapes at M=2048-4096. This is the hardest kernel in the project and the one with the
   most published prior art on this exact chip.
   Fallback for quality-sensitive layers: dequant to FP8 and use the FP8 `f8f6f4` MMA.
3. **FA2 prefill** for the 16 full-attention layers: head_dim 256, GQA, causal, partial
   RoPE and output gate fused, writes FP8 KV. Start with FlashInfer FA2 (confirmed on
   sm_121), then write our own if it is under ~60% of tensor peak. Tiles must respect
   99 KB SMEM: 64-row Q tiles with K/V streamed in 32-row tiles, or FP8 K/V tiles.
4. **Chunked Gated DeltaNet** for the 48 GDN layers: FLA `chunk_gated_delta_rule`
   (Triton) first. It is O(T * d^2), ~0.25 GFLOP/token, negligible FLOPs but memory-heavy;
   profile it, because Triton on sm_121 has had both performance and correctness issues.
   Port to CUDA only if it is more than ~15% of prefill time.
5. **Chunked prefill** with 2k-4k token chunks. Between chunks, run one decode step for
   the other slots. Chunk size is a knob; for a single user it may be better to let a new
   request prefill at full speed and stall the others briefly.
6. **Prefix checkpoints**: after each chunk and at end of generation, record (token ids,
   KV length, GDN state snapshot) in the slot's checkpoint ring. On a new request, find the
   longest checkpoint whose tokens are a prefix of the new prompt, resume there, and prefill
   only the remainder. Multi-turn chat becomes near-constant TTFT. Note the full-attention
   KV supports resuming at any length, but GDN state only at checkpoint positions, hence
   the ring.

### 4.5 Speculative decoding (ordered by effort / payoff)

1. **N-gram / prompt lookup**: zero cost, large wins on code edits and structured output.
2. **Shipped MTP head** (one full-attention block + fc), run multi-step for k=3-5 drafts.
   The block uses the target's final hidden state and its own small KV cache. Public
   results on this model and chip: 2.1-2.9x.
3. **Chain, then small tree**: with M <= 16 and a bandwidth-bound verify, a 2-wide tree at
   depth 4 costs almost nothing extra and raises acceptance length.
4. **Better drafters** (Phase 6): EAGLE-3 head (community checkpoints, or train one on the
   Spark), a DFlash-style parallel block drafter using Qwen3.8-2B, DSpark-style confidence
   scheduling. These are what pushes code generation past 50 tok/s.

Correctness: greedy acceptance for T=0, rejection sampling for T>0 (distribution-preserving);
on reject, truncate the KV length pointer and roll back GDN state (section 4.3.5).

### 4.6 Serving layer

- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming, `usage` with
  prefill/decode token counts and timings, `/metrics` with step-time histograms.
- Chat template via `transformers` `apply_chat_template` on Qwen3.8's shipped Jinja;
  `enable_thinking` via `chat_template_kwargs`; split `<think>` into `reasoning_content`;
  parse Qwen's `<tool_call>` JSON format into OpenAI `tool_calls`; `tools` passed through
  to the template.
- Stop sequences, `max_tokens`, `logprobs` (top-k from the fused sampler), seeds.
- Engine thread owns the GPU; the HTTP layer talks to it through queues; tokens are
  detokenized incrementally with the HF fast tokenizer.

### 4.7 Correctness and benchmarking harness (built in Phase 0-1, used forever)

- **Kernel unit tests**: every kernel against a torch reference, tolerances per dtype,
  including a startup self-test of each GEMM path (the sm_121 silent-wrong-answer case).
- **Model parity**: greedy generation on 30 fixed prompts x 128 tokens compared with
  (a) HF transformers BF16 for the Phase 1 BF16 path, token-exact; (b) vLLM running the
  same NVFP4 checkpoint for the quantized path, with logit KL on the first 32 steps.
- **Perplexity** on WikiText (already in the HF cache) per weight configuration.
- **Quality gate**: the existing 85-task tool-calling suite in `~/Projects/model-benchmarks`
  (OpenAI-compatible, so it runs unchanged against this engine).
- **Performance**: `bench/decode_bench.py` (tok/s vs context 1k..128k, with and without
  spec, 1-3 slots), `bench/prefill_bench.py` (tok/s and TTFT vs prompt 128..64k),
  `llama-benchy` runs, and `nsys` traces checked into `bench/traces/` at each milestone.
  Always report efficiency as a fraction of **this unit's measured bandwidth**, after a
  fresh boot, with power and clocks logged (GpuMonitor from model-benchmarks).

---

## 5. Technology choices

| Layer | Choice | Why | Rejected |
|---|---|---|---|
| Host | Python 3.12, PyTorch (cu130/cu132 wheels), uv | Allocator, safetensors, tokenizers, graphs; zero per-step cost | Pure C++/CUDA engine: 2-3x effort, no decode gain under graphs. Revisit only if we want a single static binary |
| Decode kernels | CUDA C++ (`.cu`), built with `torch.utils.cpp_extension` or CMake + nanobind, `-arch=sm_121a` | Only way to hit >=90% of bandwidth with fused epilogues | Triton GEMV: typically 75-85% of BW, harder to control vector widths and SMEM |
| Prefill GEMM | CUTLASS 4.8 C++ SM120 block-scaled collectives | The only mature NVFP4 `mma.sync` path; 356 TFLOPS demonstrated | CuTe DSL (blocks FP4 on sm_121); cuBLAS (no NVFP4 on this arch via public API); Triton (no block-scaled MMA on sm_12x yet) |
| Linear attention | FLA Triton kernels (chunked + recurrent), vendored and patched | Reference-quality GDN implementation; port hot parts to CUDA later | Writing chunked GDN from scratch first: large and not on the critical path |
| Attention | FlashInfer FA2 (stopgap) then own FA2-style CUDA | FlashInfer works on sm_121 for FA2; own kernel needed for head_dim 256 tile tuning and FP8 KV fusions | FA4 (crashes on sm_121a today), cuDNN SDPA (BF16 KV only) |
| Fusions | Triton for norm/quant/activation, CUDA where it must live inside another kernel | Fast to iterate | |
| Build | `uv` project, CMake for `csrc/`, `ccache` | | |

Environment rules: `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, `TORCH_CUDA_ARCH_LIST=12.1a`,
wipe `~/.triton/cache` on toolchain changes, pin host memory for loads, and record driver +
firmware versions with every benchmark (a firmware update moved Qwen3.8-27B 4-8% on other units).

---

## 6. Phased roadmap

Estimates are focused engineering weeks for one person who already knows this material.
Each phase has an exit criterion; do not start the next phase's performance work until the
previous phase's correctness gate is green.

### Phase 0: Ground truth (week 1)

- Create the `uv` environment (torch cu130, triton, transformers, safetensors, tokenizers,
  flashinfer, pytest, nsys). Check out CUTLASS 4.8 as a submodule.
- `bench/bw_bench.cu`: saturating read-sum over an 8 GiB buffer, 2048x256 launch; also
  read+write and a strided-16B variant. Record **this unit's GB/s**. Repeat after a reboot.
- `bench/gemm_peak.py` + CUTLASS Example 79 build: measured TFLOPS for BF16 (cuBLAS), FP8
  and NVFP4 block-scaled at M in {1, 16, 256, 2048, 4096}, N x K = 17408 x 5120 and
  5120 x 17408. This tells us the real prefill ceiling and whether 128x128x128 or K=64 tiles win.
- Download `Qwen/Qwen3.8-27B` (BF16), `-FP8`, `nvidia/Qwen3.8-27B-NVFP4`. Dump the NVFP4
  checkpoint's tensor inventory (which modules are quantized, scale shapes).
- Baseline this unit with vLLM (`vllm/vllm-openai:cu130-nightly`) and SGLang on the NVFP4
  checkpoint: decode at 1k/8k/32k, MTP on/off, prefill 2k/8k/32k, using `model-benchmarks`.
  **This requires stopping the DeepSeek container.**
- Exit: a `docs/baseline.md` with measured bandwidth, GEMM peaks and engine baselines.

### Phase 1: Correct reference engine (weeks 2-3)

- Text-only Qwen3.5-family model in plain PyTorch: embeddings, 48 GDN layers (FLA
  kernels), 16 gated-attention layers (torch SDPA), SwiGLU MLP, final norm, lm_head.
  Load BF16 and FP8 (dequant to BF16) checkpoints. Static per-slot KV and GDN state.
- Greedy and sampled generation, single slot, no graphs, no fusion.
- NVFP4 loader: unpack E2M1 + scales to BF16 on load (slow, correct) so the quantized
  weights are validated before any custom kernel exists.
- Parity harness (4.7) green against HF transformers BF16 (token-exact on 30 prompts) and
  against vLLM NVFP4 (logit KL small, perplexity within 0.5%).
- Exit: correctness harness and perplexity numbers committed. Speed irrelevant.

### Phase 2: Decode at the bandwidth roofline (weeks 4-7)

- Week 4: NVFP4 GEMV kernel, unit-tested, >=90% of measured BW on all five linear shapes
  in the model. Fused norm + residual + SwiGLU epilogues.
- Week 5: FP8 KV cache + flash-decoding kernel (or FlashInfer decode) with fused partial
  RoPE and output gate; fused GDN step kernel; FP8 lm_head GEMV; fused sampler.
- Week 6: static buffers, slot metadata on device, **CUDA graph for the full step**, 1-3
  slots. Remove every per-step Python allocation and sync.
- Week 7: profile with nsys, close gaps (launch bubbles, L2 thrash from the lm_head, state
  traffic), long-context decode to 128k.
- Exit: **>=13.5 tok/s at 8k context** on this unit, >=11 at 128k, parity harness still green,
  3-slot decode >= 0.8x per slot.

### Phase 3: Prefill at the compute roofline (weeks 8-11)

- Week 8: CUTLASS SM120 NVFP4 GEMM integrated with fused activation quant; tile sweep;
  startup self-test. Target >=180 TFLOPS on MLP shapes at M=4096.
- Week 9: FA2 prefill for head_dim 256 GQA (FlashInfer first), FP8 KV write fused; FLA
  chunked GDN profiled on sm_121 and either kept or ported.
- Week 10: chunked prefill loop interleaved with decode; prefix checkpoint ring; GDN
  checkpoint resume; multi-turn TTFT test.
- Week 11: long prompts (32k, 128k, 262k): attention becomes 25-50% of FLOPs, memory for
  chunk workspaces, numerical stability of FP8 KV at depth.
- Exit: **>=2,500 tok/s at 2k-8k prompts, TTFT(2k) <= 0.8 s, 32k prompt <= 16 s**, quality
  gate (tool-calling suite, perplexity) unchanged from Phase 1.

### Phase 4: Speculative decoding (weeks 12-15)

- Week 12: NVFP4 skinny GEMM (M <= 16) on `mma.sync`; verify step graph shapes; n-gram
  drafter; greedy acceptance; GDN snapshot/rollback; KV truncation.
- Week 13: MTP head loaded and run multi-step (k=3-5); acceptance-length telemetry per
  request type (code, prose, JSON).
- Week 14: rejection sampling for T>0; small tree drafts; per-request adaptive k.
- Week 15: tune against real workloads (the agent traffic you actually send it).
- Exit: **>=35 tok/s** on a code/chat mix at T=0, >=30 at T=0.7, output distribution
  verified unchanged (greedy outputs identical with and without spec).

### Phase 5: Daily driver (weeks 16-18)

- OpenAI-compatible server, streaming, tool calls, thinking, stop sequences, logprobs,
  metrics, graceful handling of the 3-slot limit (queue the 4th).
- Run the full `model-benchmarks` suite and `llama-benchy` against it; publish
  `docs/results.md` versus the Phase 0 baselines.
- Startup under 60 s from the repacked weight cache; systemd unit; memory cap enforced.
- Exit: you switch your own tooling to it.

### Phase 6: Pushing past the public numbers (ongoing)

- Persistent **megakernel decode**: one cooperative kernel per step that streams weights
  layer to layer without launch boundaries and overlaps the lm_head read with the last
  layers' compute. Expected +5-10% on base decode.
- Better drafters (EAGLE-3 trained locally on the Spark; DFlash-style 2B parallel drafter).
- NVFP4 lm_head; FP8 attention/GDN projections if NVFP4 shows quality loss there.
- Sub-4-bit weights for decode (3 bpw trellis or TurboQuant-style): 24B x 0.375 B = 9 GB,
  a base-decode ceiling near 23 tok/s, at a quality cost to be measured.
- 4-bit KV for >128k contexts; the vision tower if you want image input.

### Phase 7: DeepSeek V4 Flash (after Phase 5; 10-16 weeks)

What changes versus the 27B engine, in order of effort:

1. **Weights that fit**: 3 bpw routed experts (EXL3 trellis decode kernels exist in
   ExLlamaV3; TurboQuant 3-bit exists in the llama.cpp-dgx fork) plus NVFP4 for attention
   and shared experts. Roofline says ~6.4 GB active per token: a base ceiling of ~35 tok/s
   at 230 GB/s, versus ~26 with FP8 attention as served today. **Quantizing the attention
   path to 4 bits is the biggest single win available on this model.**
2. **MoE decode kernel**: router (`sqrtsoftplus`, `noaux_tc` bias, top-6, hash tables for
   layers 0-2), then a grouped GEMV over 7 experts x 3 matrices with the same bandwidth
   discipline as the dense GEMV. Prefill: grouped block-scaled GEMM (CUTLASS SM120 has
   a mixed-input block-scaled grouped GEMM since 4.5).
3. **Attention**: single shared 512-d K=V latent, 64 Q heads, sliding window 128 everywhere,
   CSA (4:1 learned pooling + lightning indexer top-512 in FP4) on even layers, HCA (128:1)
   on odd layers, attention sinks, grouped low-rank output projection, YaRN to 1M.
4. **mHC hyper-connections** (4 residual streams, Sinkhorn-normalized mixing) replace the
   residual stream: changes every layer's input/output plumbing and the GDN-free step graph.
5. **DSpark** 3-layer draft from the checkpoint (block size 5), which is what takes the
   current stack from ~22 to 44-47 tok/s. With a 35 tok/s base the same multiplier lands
   near 70.

Local references for this phase already on disk: `~/code/ds4` (C/CUDA engine for V4 Flash:
MoE decode, CSA attention, mHC fused stage), and the `sparkinfer` image patches under
`~/Projects/DeepSeek-v4-Flash-One-DGX-Spark/image-patch/` (MLA prefill, tiny-decode MoE).

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CUTLASS NVFP4 GEMM on sm_121 produces wrong results silently or underperforms | Medium | Startup self-test; tile sweep in Phase 0; FP8 fallback path kept alive; follow CUTLASS PR #3438 |
| Triton kernels (FLA GDN) misbehave on sm_121 | Medium | Test in Phase 0; `TRITON_PTXAS_PATH`; cache wipes; CUDA port of the recurrent step is planned anyway |
| NVFP4 quality on GDN in-projections or lm_head | Medium | Per-layer dtype override in the loader; perplexity + tool-calling gate per change |
| MTP acceptance on the hybrid model is low (one vLLM test saw no gain) | Medium | Measure acceptance per domain early (week 13); n-gram and better drafters as backup |
| GDN rollback bugs under speculation | Medium | Greedy-identity test: outputs with and without spec must match exactly |
| Memory over-commit hard-resets the box | Low-Medium | Hard allocation cap, no runtime growth, stop the DeepSeek container during development |
| Thermal/power throttling at 140 W during long prefill | Low | Log clocks/power in every benchmark; compare to reboot-fresh numbers |
| Toolchain drift (driver, firmware, torch cu13x) | Certain | Pin versions in `pyproject.toml`/`docs/baseline.md`; re-run Phase 0 microbenchmarks after any update |
| Scope creep into DeepSeek before the 27B is done | High | Phase gates above |

---

## 8. Repository layout

```
colin-inference-engine/
  PLAN.md                     this document
  pyproject.toml              uv project; pinned torch/triton/transformers
  engine/
    model/qwen35.py           text-only Qwen3.5/3.6/3.8 architecture
    model/deepseek_v4.py      Phase 7
    weights/loader.py         safetensors -> device, pinned reads, repack cache
    weights/nvfp4.py          format helpers, dequant reference
    kv/slots.py               KV slots, GDN state, checkpoint ring
    runtime/prefill.py        chunked prefill
    runtime/decode.py         step builder + CUDA graph capture/replay
    runtime/scheduler.py      3-slot admission and interleaving
    runtime/sampler.py        sampling params -> device buffers
    spec/ngram.py, spec/mtp.py
    server/api.py             OpenAI-compatible HTTP
    server/chat.py            templates, thinking, tool-call parsing
  csrc/
    gemv_nvfp4.cu             M<=4 decode GEMV
    skinny_gemm_nvfp4.cu      5<=M<=16 verify path
    gemm_nvfp4_sm120.cu       CUTLASS prefill GEMM + epilogues
    attn_decode.cu, attn_prefill.cu
    gdn_step.cu               fused recurrent Gated DeltaNet
    norm_quant.cu, sampling.cu
    bindings.cpp
    third_party/cutlass       submodule
  kernels_triton/             fusions; vendored FLA chunk kernels with sm_121 patches
  tests/                      kernel unit tests, parity, perplexity, spec identity
  bench/                      bw_bench.cu, gemm_peak.py, decode_bench.py, prefill_bench.py,
                              compare_vllm.sh, traces/
  tools/roofline.py           (exists) byte/FLOP model behind every target in this plan
  docs/baseline.md, docs/results.md, docs/notes/
```

---

## 9. First week, concretely

1. Stop the DeepSeek container (`~/Projects/DeepSeek-v4-Flash-One-DGX-Spark/stop.sh`),
   confirm `free -h` shows >100 GB available.
2. `uv init`, install torch cu130 + triton + transformers + safetensors + flashinfer; verify
   `torch.cuda.get_device_properties(0)` reports 48 SMs, 24 MB L2, capability 12.1.
3. Write and run `bench/bw_bench.cu` (`nvcc -O3 -arch=sm_121a`). Record GB/s.
4. Clone CUTLASS 4.8, build Example 79 for `sm_121a`, run the NVFP4 GEMM profiler on the
   model's shapes. Record TFLOPS per tile config. Also cuBLAS BF16 and CUTLASS FP8.
5. Download the three Qwen3.8-27B checkpoints; dump the NVFP4 tensor inventory.
6. Run vLLM and SGLang baselines on the NVFP4 checkpoint with `model-benchmarks`.
7. Re-run `tools/roofline.py --bw <measured> --tflops <measured>` and freeze the Phase 2-4
   targets in `docs/baseline.md`.
8. Start Phase 1: the plain-PyTorch model definition and the parity harness.

---

## 10. Reading list

- DeepSeek-V4 report (arXiv 2606.19348), DSpark (arXiv 2607.05147), DeepSeek-V4.1-Flash (arXiv 2609.19969)
- Qwen3-Next / Qwen3.5 model cards; Gated DeltaNet (Yang et al.); flash-linear-attention repo
- CUTLASS `blackwell_functionality.md` (SM120 section), Example 79, CUTLASS issues #2800/#2947, PR #3438
- NVIDIA forum: "PSA: state of FP4/NVFP4 support for DGX Spark in vLLM" (K=64 tile fix), "FP4 on DGX Spark: why it doesn't scale like you'd expect" (356 TFLOPS measurement)
- FlashInfer SM121 audit (issue #3170); Triton issues #8539, #8335; vLLM PR #52708
- Chips and Cheese, "Analyzing NVIDIA GB10's GPU"; `antirez/ds4` issue #773 (bandwidth ceiling discussion)
- Marlin (W4A16 skinny GEMM), FlashDecoding, EAGLE-3, Medusa/tree verification, Hazy Research "low-latency megakernel"
- Local: `~/Projects/nanoGPT-inference` (your own spec-decode, CUDA-graph and chunked-prefill prototypes), `~/code/ds4`, the `sparkinfer` patches

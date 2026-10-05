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

- Full re-quantization stays opt-in: `--decode-weights requant`, using the GPTQ damping-0.3 file when present.

**AWQ, and why only attention made it.** `tools/awq_nvfp4.py` scales each input channel j by s_j = E[x_j²]^(α/2)
before quantizing (W diag(s)); decode divides the activations by s (`_awq_in` in `engine/model/fast.py`). α is
searched per group of linears sharing an input, by the output error the calibration inputs see (128 sequences, half
WikiText-103 train, half `transformers` code; the same Hessians as GPTQ). α = 0 is round-to-nearest, so calibration
error can only drop; it does, by 17% (α ≈ 0.5 everywhere): attention qkv 0.052 → 0.035, GDN qkvz 0.059 → 0.044,
o_proj / out_proj ~10%. Perplexity (reference engine, ctx 2048, 65,504 tokens; shipped weights 6.9698 / 1.7257):

| NVFP4 for | WikiText | Python code |
|---|---|---|
| All 208, round-to-nearest | 6.9789 (+0.13%) | 1.7650 (+2.3%) |
| All 208, AWQ | 7.0419 (+1.03%) | 1.7508 (+1.45%) |
| AWQ, attention q/k/v only | 6.9719 (+0.03%) | |
| AWQ, attention o_proj only | 6.9657 (-0.06%) | |
| AWQ, GDN in_proj_qkv / z only | 7.0097 (+0.57%) | |
| AWQ, GDN out_proj only | 7.0048 (+0.50%) | |
| **AWQ, attention q/k/v/o (64 linears)** | **6.9663 (-0.05%)** | **1.7332 (+0.43%)** |

- AWQ helps attention (round-to-nearest attention alone cost code +0.66%) and hurts Gated DeltaNet on WikiText,
  although it lowers those linears' output error too. GDN projection errors pass through the recurrent state, where
  a per-linear output-error objective does not predict their cost; GPTQ showed the same mismatch.
- The scales stay within 0.2-5 (no channel is crushed), so this is not a range problem.
- Attention-only AWQ passes the gate (≤ 0.5% on both corpora): **`--decode-weights auto`, the default, now decodes the
  64 attention linears from `attn_gdn_nvfp4_awq_attn.safetensors` when it exists** (`tools/awq_nvfp4.py --groups
  self_attn`). The GDN projections stay FP8.
- Speed (`bench/decode_bench.py`, 8k): plain decode 81.4 → 79.0 ms, MTP cycles k=3 91.9 → 89.2 ms, k=7 104.5 → 102.2 ms,
  width 3 k=3 101.8 → 99.4 ms: ~3%. The attention linears are 1.7B of the 7.2B FP8 parameters; the other 15%
  is in GDN, which needs a better format than NVFP4 or a quantizer that models the recurrence.
- `tests/scheduler_check.py --requant` with the AWQ file: every output identical with and without speculation.
- Tool calling (85 tasks, 2048 tokens): 50 with AWQ attention, 51 with the FP8 projections; one schema task differs
  (`schema_title_max_length_fail`). Full re-quantization lost three (48).

**Tool calling.** On the 85-task suite at 2048 tokens the scores sit within the same 8 borderline tasks:

| Configuration | Tasks passed |
|---|---|
| Re-quantized weights | 48 |
| AWQ attention only (the default since) | 50 |
| Skinny GEMM with the FP8 projections | 51 |
| Phase 5 / vLLM | 53 / 53 |

Gate for making it the default: ≤ 0.5% on both corpora.

## 4. Decode-step overhead ("megakernel")

A trace of a plain decode step (`nsys`) shows it is already 98% GPU-busy: 17.6 GB in 77.5 ms, 227 GB/s, about 95% of
what this machine streams (238-240 GB/s). Of the remaining 5%:

- **1.5 ms of idle gaps** between kernels.
- **GDN state read / write:** 0.3 GB.
- **KV reads:** 0.27 GB at 8k.

A persistent megakernel could recover at most the gaps, about 2%. The two cheaper changes below captured more:

- **L2 prefetch.** Before its first activation read, the skinny GEMM pulls its next two chunks into L2.
- **Programmatic dependent launch** (`csrc/pdl.cuh`).
  - Every decode-path kernel signals its dependents first.
  - The GEMM is launched with programmatic serialization, so it starts streaming weights while the preceding norm,
    attention or GDN kernel finishes, and waits (`griddepcontrol.wait`) only before it reads activations.

| | Before | Prefetch + PDL |
|---|---|---|
| Plain decode step, 8k | 81.5 ms | 77-79 ms (12.6-13.0 tok/s) |
| Cycle, width 1, k=3 / k=7 | 94.4 / 111.3 ms | 89-92 / 106-108 ms |
| Cycle, width 3, k=3 | 110.5 ms | 103-106 ms |

`COLINFER_PDL=0` turns PDL off.

## 5. 4-bit KV cache

Opt-in with `--kv fp4`.

**Format.** A (token, head) row of 256 values is stored as 144 bytes: 128 bytes of e2m1 plus 16 e4m3 block scales,
0.56× of FP8. The scales are stored ×16 so block maxima from 0.006 to 168 stay in e4m3's normal range (measured on
this model: K 0.04-23, V 0.17-108).

**Implementation.**

- **Write:** the attention prologue quantizes on write, with 16-lane shuffles for the block maximum and nibble pairing.
- **Read:** the decode kernel dequantizes per lane.
- **Prefill:** dequantizes the cached prefix to BF16 for FlashInfer (`kv4_to_bf16`).
- **Cross-slot copies:** rows stay one tensor, so cross-slot prefix copies work unchanged.

**Quality.**

| Prefill-engine perplexity (W4A4), ctx 2048 | WikiText | Python code |
|---|---|---|
| BF16 KV | 7.0910 | 1.8015 |
| FP8 KV | 7.0909 | 1.8012 |
| FP4 KV | 7.1141 (+0.33%) | 1.8054 (+0.23%) |

- Passkey retrieval at 126k: 3/3.
- Choosing each block's scale by squared error, allowing amax/5, made perplexity worse (+0.51%). Clipping a block
  maximum hurts attention more than coarser steps do.

**Speed at 128k context:**

| | FP8 KV | FP4 KV |
|---|---|---|
| Plain decode step | 96.0 ms | 93.2 ms |
| MTP cycle, k=3 / k=7 | 144 / 214 ms | 148 / 223 ms |

The bytes halve but dequantizing costs more per element, so FP4 is mainly a memory option: 3 × 262k slots in
14.5 GB instead of 25.8 GB. (With the tensor-core attention of section 6, fp4 is also faster at long context: 128k plain
decode 89.1 vs 97.3 ms, MTP cycles 107 / 134 ms vs 115 / 140 ms for k=3 / k=7, width 3 k=3 153 vs 175 ms.)

**The real long-context cost is in multi-row attention.** The decode kernel runs one block per verify row, so a
k-draft cycle reads the KV cache k+1 times. At 128k a k=3 cycle costs 144 ms against 89 ms at 8k.

**What I tried.** One block per (KV head, split) with one warp per verify row, which reads KV once and keeps every row
bit-identical to plain decode. It was 1.7× slower: one warp per row on CUDA cores is too serial. Section 6 has the fix.

## 6. Tensor-core multi-row decode attention

`csrc/attn_decode.cu`, namespace `tc`; the default for fp8 and fp4 caches (`COLINFER_ATTN_TC=0` selects the split-KV
kernel). One block owns (slot, KV head, stripe of key tiles) and serves every query row of the slot: up to 48 rows
(6 query heads × T new tokens) as three m16 tiles of `mma.sync.m16n8k16` (f16 operands, fp32 accumulate).

**Bit identity across T.** Plain decode (T = 1) and a k-draft verify must give a row the same bits, so nothing a row
computes may depend on T:

- Keys go in 32-key tiles at fixed positions; block z of 12 takes tiles z, z+12, z+24, ...; the combine folds the 12
  partials in a fixed order. None of this depends on T or on the sequence length.
- mma rows are independent. A row's softmax is 8 lanes with fixed reduction trees, the same code for every row.
- A tile entirely past a row's length (a longer row of the same slot needs it) is an exact no-op: its scores are
  -inf, p = 0, the running max does not move and the rescale factor is set to exactly 1.

`tests/test_attn_decode_tc.py` checks each row of 2-12-row launches bit for bit against one-row launches, and
`tests/spec_check.py` / `tests/scheduler_check.py` still pass (greedy and seeded-sampled output identical with and without
speculation, at every batch width).

**Operands.**

- K is used straight from the raw cache bytes in shared memory. A lane's mma fragment for k-step j is head dims
  16j + 4c .. +3 (c = lane % 4), one 32-bit load. That is a permutation of the k dimension, which Q's fragment
  repeats (8 consecutive bytes of the f16 Q row). e4m3 values, and e2m1 × e4m3-scale values, are exact in f16.
- V is converted to an f16 tile and read with `ldmatrix.trans`; P is rounded to f16. Error against fp32 attention is
  ~5e-4 relative, below the bf16 output's own rounding.

**Two things that mattered for speed.**

| Change | 128k, fp8, T=1 | T=8 |
|---|---|---|
| First version (raw rows padded to 272 bytes against bank conflicts) | 189 GB/s | 189 GB/s |
| Rows 256 bytes apart, XOR-swizzled 16-byte chunks instead | 233 GB/s | 188 GB/s |
| Softmax on 8 lanes per row, 4 rows per warp at once (was a warp per row) | 232 GB/s | 232 GB/s |

The padding cost 20% of the cp.async streaming rate even with the compute removed (a loads-only build streamed
187 GB/s with 272-byte rows, 232 with 256-byte rows). More pipeline stages (2 → 4) and contiguous per-block chunks
changed nothing.

**Per layer** (`bench/attn_bench.py`, 128k context):

| KV | Slots | T = 1 (plain) | T = 4 (k=3) | T = 8 (k=7) |
|---|---|---|---|---|
| fp8, split-KV | 1 | 1.21 ms | 3.11 ms | 6.09 ms |
| fp8, tensor-core | 1 | 1.16 ms | 1.14 ms | 1.16 ms |
| fp8, split-KV | 3 | 3.55 ms | 9.11 ms | 17.7 ms |
| fp8, tensor-core | 3 | 3.45 ms | 3.51 ms | 3.53 ms |
| fp4, split-KV | 1 | 1.06 ms | 3.29 ms | 6.46 ms |
| fp4, tensor-core | 1 | 0.66 ms | 0.67 ms | 0.87 ms |

fp8 streams 228-236 GB/s at every T: verifying 8 rows costs what plain decode costs. fp4 plain decode becomes 1.6×
faster than before (227 GB/s); its T = 8 is compute-bound at ~175 GB/s (dequantization and the three m-tiles).

**MTP drafter cache in fp8.** Each of the k draft steps of a cycle reads the drafter's own KV cache (the MTP layer's),
which was bf16: 512 MB per step per slot at 128k. It is now e4m3 like the target's (`MtpState`, `COLINFER_MTP_KV_FP8=0`
for bf16), read by the same kernel. Acceptance is unchanged: with the same target kernel, `tools/eval_drafter.py` gives
2.76 / 3.39 tokens per cycle (k=3 / k=7) either way, every category within 0.01. (A first comparison suggested fp8 cost
code acceptance; it compared against an evaluation run before the softmax change above, whose greedy text differs.)
The head's K and V sit well inside e4m3's range (|K| median 0.95, max 19; |V| median 1.6, max 39).

**Decode cycle** (`bench/decode_bench.py`, fp8 KV; before = split-KV attention and a bf16 drafter cache):

| | 8k before | 8k after | 128k before | 128k after |
|---|---|---|---|---|
| Plain decode, width 1 | 80.8 ms | 80.7 ms | 98.7 ms | 97.3 ms |
| MTP cycle, width 1, k=3 | 93.3 ms | 91.1 ms | 148.5 ms | **110.6 ms** |
| MTP cycle, width 1, k=7 | 109.6 ms | 103.4 ms | 219.1 ms | **126.8 ms** |
| MTP cycle, width 2, k=7 | 124.6 ms | 111.7 ms | 339.1 ms | **159.5 ms** |
| MTP cycle, width 3, k=3 | 108.5 ms | 100.8 ms | 269.1 ms | **160.5 ms** |

At 128k a k=7 cycle went from 2.2× a plain step to 1.3×; at 8k cycles are 2-10% shorter.

## 7. Fine-tuning the MTP head as a multi-step drafter

The checkpoint's MTP head is trained for one step: (embedding of x_{i+1}, target hidden h_i) → x_{i+2}. The engine
chains it, so drafts 2..k feed the head its own output instead of a target hidden state, inputs it never trained on.
`tools/train_drafter.py` fine-tunes it EAGLE-3 style ("training-time test"): each row is unrolled to depth D exactly
as drafting runs, against the target's top-32 next-token distribution (soft cross-entropy, 64k draft vocabulary), on
the target's own replies (`tools/drafter_data.py`). Embedding and lm_head stay frozen. `tests/test_drafter_train.py`
checks the unroll against incremental chained drafting.

**Data:** 1,500 sampled replies (T = 0.7) to generated prompts, 490k tokens: 45% prose, 15% Q&A, 25% code, 15%
structured; thinking on for half. 75 are held out.

**Runs** (held-out top-1 agreement with the target, per draft depth):

| | lr | Depth | Epochs | Depth 1 | Depth 3 | Depth 5 | Depth 7 |
|---|---|---|---|---|---|---|---|
| Checkpoint head | | | | 0.802 | 0.672 | 0.619 | 0.586 |
| Run 1 | 3e-5 | 5 | 2 | 0.808 | 0.694 | 0.655 | |
| Run 2 (stopped after epoch 0: worse than before) | 1e-4 | 7 | 3 | 0.769 | 0.626 | 0.582 | 0.561 |
| Run 3 | 3e-5 | 7 | 4 | 0.807 | 0.692 | 0.657 | 0.635 |
| Run 4 (the default), 2.7× the data | 3e-5 | 5 | 2 | 0.812 | 0.700 | 0.668 | |

**End to end** (`tools/eval_drafter.py`: 40 fresh prompts, greedy, the engine's real speculative cycle; tokens per
cycle, acceptance in parentheses):

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| Checkpoint head, k=3 | 3.25 (0.75) | 2.45 (0.49) | 2.57 | 2.71 | 2.67 |
| Run 1, k=3 | 3.45 (0.82) | 2.50 (0.50) | 2.67 | 2.77 | 2.75 |
| Run 3, k=3 | 3.41 (0.80) | 2.50 (0.50) | 2.63 | 2.76 | 2.74 |
| Checkpoint head, k=7 | 4.46 (0.50) | 2.74 (0.25) | 3.01 | 3.22 | 3.16 |
| Run 1, k=7 | 5.29 (0.62) | 2.84 (0.27) | 3.16 | 3.39 | 3.36 |
| Run 3, k=7 | 5.18 (0.61) | 2.84 (0.27) | 3.19 | 3.40 | 3.36 |

- Fine-tuning helps most on code (+19% tokens per cycle at k=7) and least on prose (+4%).
- Run 3's deeper unroll and twice the epochs gained nothing end to end over run 1. Training longer on the same
  1,500 replies has stopped paying.

**Run 4: more data.** 2,402 more replies from new prompts (`drafter_data.py --seed 2`; seed 1 is the evaluation's), 942k
tokens, so 3,827 training replies and 1.43M tokens. The held-out split is the same 75 replies as before (it is drawn
from the first data file only). Same evaluation, rerun with the current attention kernel: the target's greedy text
differs slightly from the table above (section 6 changed its softmax reduction order), so compare within this table:

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| Checkpoint head, k=3 | 3.13 | 2.44 | 2.66 | 2.70 | 2.67 |
| Run 1, k=3 | 3.31 | 2.49 | 2.75 | 2.76 | 2.75 |
| Run 4, k=3 | 3.36 | 2.48 | 2.72 | 2.82 | 2.76 |
| Checkpoint head, k=7 | 4.16 | 2.72 | 3.12 | 3.19 | 3.14 |
| Run 1, k=7 | 4.80 | 2.82 | 3.39 | 3.34 | 3.35 |
| Run 4, k=7 | 4.97 | 2.82 | 3.34 | 3.46 | 3.39 |

- 2.7× the data: held-out agreement +1.3 points at depth 5, code and structured output +3-4% at k=7, prose flat.
- Run 4 is now `~/.cache/colinfer/drafter/mtp_ft.safetensors`, which the server loads by default (`--drafter-weights
  auto`); run 1 is kept as `mtp_ft_r1.safetensors`.
- Prose is stuck at ~0.5 acceptance per token with this head. Fine-tuning a one-layer head on more of the same data
  helps code-like text, not prose.

**EAGLE-3-style multi-layer features: no gain.** EAGLE-3 feeds its drafter the target's low, middle and high-layer
hidden states, not only the last one. `prefill(return_layers=...)` returns the residual stream after given layers, and
`train_drafter.py --layers 3,23,43` fuses them into the depth-1 hidden input:
h0 = h + B A [rms(l3), rms(l23), rms(l43)], rank 256, B = 0 at the start (exactly run 4), trained from run 4 on the same
3,827 replies (features 80 GB, streamed from disk a shard at a time).

| | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---|---|---|---|---|---|
| Run 4 (start) | 0.812 | 0.739 | 0.700 | 0.680 | 0.667 |
| + layer fusion, epoch 0 | 0.806 | 0.730 | 0.691 | 0.667 | 0.653 |
| + layer fusion, epoch 1 (end) | 0.811 | 0.737 | 0.701 | 0.681 | 0.667 |

- The trained fusion term is under 1% of the final hidden's RMS: the gradient found no draft signal in the earlier
  layers that the final hidden lacks. (Epoch 0's dip is the restart at full learning rate; run 4 shows the same dip.)
- A full-rank fusion ([0 0 0 I] + learned, 105M parameters, lr 1e-4) was worse: -2 points after one epoch, stopped.
- Not integrated into the engine. With one decoder layer and ~1.4M training tokens the drafter's limit on prose is
  its capacity and data, not its inputs. Remaining routes: a larger drafter (several layers, or a small parallel
  drafter model) trained on far more data, which is days of generation and training on this machine.

## 8. Long-prompt prefill: where the time goes

Prefill (`bench/prefill_bench.py --kv-fp8`, chunk 2048): 3,057 tok/s at 8k, 2,556 at 32k, 2,017 at 64k.
Per-kernel GPU time of the 2,048-token chunk that follows 30,720 prompt tokens (`--profile --profile-at 30720`), 998 ms:

| Kernel | ms | Share | Rate |
|---|---|---|---|
| FlashInfer causal prefill attention (16 layers) | 327 | 33% | 78 TFLOPS |
| CUTLASS NVFP4 W4A4 GEMMs (MLP) | 239 | 24% | ~293 TFLOPS (peak ~355) |
| cuBLASLt FP8 GEMMs (attention / GDN projections) | 180 | 18% | ~164 TFLOPS (peak ~200) |
| SiLU-mul + quantize, add + RMSNorm, GDN chunk kernels, conv, norms | ~250 | 25% | |

The GEMMs are near their peaks; attention is the term that grows with context. Its alternatives on this shape (2,048
queries, 6:1 GQA, head_dim 256, lower-right causal; `torch` 2.13):

| Context | FlashInfer | torch SDPA flash | torch SDPA mem-efficient | cuDNN |
|---|---|---|---|---|
| 8k | 87 TFLOPS | 68 | 27 | no kernel for sm_121 |
| 32k | 79 | 62 | 25 | |
| 64k | 79 | 62 | 25 | |

FlashInfer runs at ~81% of the measured BF16 GEMM peak (96 TFLOPS), so a BF16 kernel of our own has little room. The
lever is FP8 tensor cores (~200 TFLOPS): Q K^T in e4m3 can read the FP8 K cache as it is (a lane's m16n8k32 fragment is 4
consecutive head dims of one key, no permutation needed) with Q quantized per (token, head). Quality, emulated in the
prefill engine (`COLINFER_EMU_Q_FP8=1`, Q rounded to e4m3 before FlashInfer; FP8 KV, perplexity):

| | WikiText ctx 2048 | Code ctx 2048 | Code ctx 8192 |
|---|---|---|---|
| BF16 Q | 7.0909 | 1.8012 | 1.6085 |
| e4m3 Q, per (token, head) scale | 7.0777 | 1.7985 | 1.6105 (+0.12%) |

Estimate for a kernel doing Q K^T in FP8 and P V in f16 (P rounded to f16, V converted from the FP8 cache), each at 75%
of its peak: ~97 TFLOPS-equivalent vs FlashInfer's 79, i.e. attention 1.2× faster: ~6% shorter prefill at 32k, ~10-12%
at 64k-128k. Not built yet.

## Next

- A better drafter for prose. Acceptance there is about 0.45-0.50, so speculation adds about 1.15×. Section 7.
- Tensor-core multi-row decode attention: done (section 6).
- FP8 (Q K^T) prefill attention for long prompts (section 8): ~6% at 32k, ~10-12% at 64k-128k.

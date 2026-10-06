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
| AWQ attention only | 50 |
| INT6 attention + INT5 GDN (the default since, section 9) | 51 |
| Skinny GEMM with the FP8 projections | 51 |
| Phase 5 / vLLM | 53 / 53 |

Gate for making it the default: ≤ 0.5% on both corpora.

## 4. Decode-step overhead ("megakernel")

A trace of a plain decode step (`nsys`) shows it is already 98% GPU-busy: 17.6 GB in 77.5 ms, 227 GB/s, about 95% of
what this machine streams (238-240 GB/s). Of the remaining 5%:

- **1.5 ms of idle gaps** between kernels.
- **GDN state read / write:** 0.3 GB.
- **KV reads:** 0.27 GB at 8k.

A persistent megakernel could recover at most the gaps, about 2%. The two cheaper changes below captured more (with them,
a later trace of an MTP cycle on the INT6 / INT5 weights of section 9 shows 0.09 ms idle in 83.7 ms, 0.1%: nothing is
left for a megakernel to take):

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

### The FP8-QK prefill attention kernel (`csrc/attn_prefill.cu`)

- FA2 structure: a block is 64 query rows of one query head (4 warps × 16 rows), KV tiles of 32 keys double-buffered
  with cp.async, two blocks per SM; the heaviest causal tiles start first, and the 6 heads of a KV head are adjacent so
  their KV stream is shared in L2.
- S = Q K^T on `mma.m16n8k32` e4m3. Q is rounded to e4m3 in the kernel (per (token, head) scale, held as A fragments in
  registers); K is read as stored. A lane loads 16 consecutive bytes of a K row (one 128-bit shared load per two
  k32 steps; the head-dim order inside the dot product is permuted identically for Q). Rows are 256 bytes apart with
  chunks XOR'ed by row parity: no bank conflicts.
- Online softmax in registers; P rounded to f16 and reused as the A fragment of P V (f16 `m16n8k16`, V converted from
  e4m3 to an f16 tile, `ldmatrix.trans`).
- Accuracy against fp32 attention given the same e4m3 Q: 0.4-0.8% relative on random data, 0.17-0.35% on the model's
  own attention inputs, the same as FlashInfer's error given that Q. `tests/test_attn_prefill_fp8.py`.

**Attention alone** (2,048 queries after L - 2,048 cached tokens):

| Context | FlashInfer | FP8 kernel, 32-key tiles | 64-key tiles |
|---|---|---|---|
| 8k | 4.24 ms (85 TFLOPS) | 3.40 ms (106) | 4.15 ms (87) |
| 32k | 19.4 ms (82) | 15.4 ms (104) | 18.6 ms (86) |
| 64k | 39.4 ms (82) | 32.0 ms (102) | 37.4 ms (87) |

**The cost is Q's rounding, layer by layer.** On real activations (a 2,048-token Python file), rounding Q to e4m3 moves
attention outputs by 0.4-1.3% in the first and last few attention layers and by 2-4% in layers 23-51: their scores are
dominated by a few large q·k terms, which e4m3's 3-bit mantissa rounds. Perplexity (prefill engine, FP8 KV):

| | FlashInfer | FP8 kernel |
|---|---|---|
| WikiText, ctx 2048 | 7.0909 | 7.1086 (+0.25%) |
| Code, ctx 2048 | 1.8012 | 1.8053 (+0.23%) |
| Code, ctx 8192 | 1.6085 | 1.6135 (+0.31%) |
| Code, ctx 16384 | 1.5650 | 1.5691 (+0.26%) |

**Default: the FP8 kernel for chunks whose context passes 16k** (`COLINFER_ATTN_FP8_MIN_CTX`, 16384; off with
`COLINFER_ATTN_FP8_PREFILL=0`), where attention is a large share of the time. Shorter prompts are computed exactly as
before; the MTP head's prefill follows the same rule.

| Prompt | FlashInfer | Default (FP8 past 16k) |
|---|---|---|
| 8k | 2.64 s | 2.65 s (unchanged path) |
| 32k | 12.70 s | 12.44 s (-2%) |
| 64k | 32.05-32.32 s | 30.27-30.49 s (-5.6%) |
| 128k | 90.9 s | 80.9 s (-11%) |

**Trying to remove the quality cost.**

- FlashInfer for layers 23-51 (their outputs move most): WikiText back to +0.03%, but code +0.27% at ctx 2048 and
  +0.43% at 8k (no better than all-FP8): the per-layer output error does not predict code perplexity.
- Which head dims carry the error (real activations): keeping the 32 dims with the largest mean |q|·|k| in bf16 cuts
  the middle layers' error from 3-4% to 1-1.5% (64 dims: ~1%). They are not the RoPE dims and are spread over most
  16-dim chunks, while the kernel can only treat whole 16-byte K chunks specially, so a residual-Q correction would cost
  most of the speedup.
- Conclusion: +0.25-0.4% is the price of e4m3 Q at this speed; it stays limited to chunks past 16k.

Passkey retrieval with it: 6/6 at 31k and 126k. `tests/scheduler_check.py` with the threshold at 0 (every prefill on the
kernel, chunked prefill, prefix reuse, MTP): passed.

## 9. INT5 / INT6 for the attention and GDN projections (the default)

NVFP4 for the 144 Gated DeltaNet projections fails the quality gate whatever the method (sections 3: round-to-nearest,
GPTQ, AWQ; +0.4-0.6% code perplexity per projection type). The checkpoint's FP8 weights themselves are 2.67% off the
BF16 originals; a format with that error but fewer bits is what decode needs. Relative weight error on GDN layers 4 /
30 / 52 (all three projections alike):

| Format (e4m3 scale per 16 weights + fp32 global scale) | Bits / weight | Error vs BF16 |
|---|---|---|
| Checkpoint FP8 (per-tensor scale) | 8 | 2.67% |
| INT6 | 6.5 | **2.24%** |
| FP6 e2m3 | 6.5 | 2.58% |
| INT5 | 5.5 | 4.2% |
| FP6 e3m2 | 6.5 | 4.5% |
| NVFP4 | 4.5 | 8.5% |

`tools/int6_requant.py` quantizes from the BF16 originals (block scale searched among amax/qmax × {1 ... 0.8} by
squared error; stacked projections share the global scale). Perplexity (reference engine, ctx 2048; shipped weights
6.9698 / 1.7257):

| Decode copies | WikiText | Python code | Bytes per token saved vs FP8 |
|---|---|---|---|
| INT6, all 208 | 6.9697 (0.00%) | 1.7250 (-0.04%) | 1.35 GB |
| INT6 GDN + AWQ NVFP4 attention | 6.9445 (-0.36%) | 1.7310 (+0.31%) | 1.77 GB |
| INT5 GDN, FP8 attention | 6.9387 (-0.45%) | 1.7288 (+0.18%) | 1.74 GB |
| INT5, all 208 | 6.9431 (-0.38%) | 1.7333 (+0.44%) | 2.27 GB |
| **INT5 GDN + INT6 attention (default)** | **6.9552 (-0.21%)** | **1.7296 (+0.23%)** | **2.06 GB** |

INT6 is lossless here; INT5 on GDN costs a fifth of the gate.

**Kernel** (`csrc/skinny.cu`, formats INT6 / INT5). The code's low 4 bits are stored exactly like NVFP4's nibbles
([N, K/2]), so they stream and land in the per-warp scratch the same way; the high 2 (1) bits are a second plane
[N, K/4] ([N, K/8]), 128 (64) bytes per row per 512-k chunk, where a k-step's 4 codes share one byte (nibble).
Dequantization builds bf16 directly: 0x4300 | c is 128 + c, minus 160 (144) is q exactly, times the bf16 block scale
rounds once (q × s has up to 9 significant bits; ~0.2% relative). Rows stay bit-identical for every M
(`tests/test_skinny_int.py`). Against FP8 on the GDN shapes: INT6 1.17-1.27× (bytes: 1.23×), INT5 1.32-1.45× (1.45×).

**Decode** (`bench/decode_bench.py`, 8k):

| | FP8 | AWQ attention (previous default) | INT6 attention + INT5 GDN |
|---|---|---|---|
| Plain decode, width 1 | 81.8 ms | 78.8 ms | **74.0 ms** (13.5 tok/s) |
| Plain decode, width 3 | 88.0 ms | 85.6 ms | 80.7 ms |
| MTP cycle, width 1, k=3 / k=7 | 91.8 / 103.6 ms | 89.7 / 101.5 ms | **84.7 / 97.2 ms** |
| MTP cycle, width 2, k=7 | 112.6 ms | 110.6 ms | 105.2 ms |
| MTP cycle, width 3, k=3 | 101.8 ms | 99.1 ms | 95.5 ms |

- `--decode-weights auto` (default) now picks `int` when both files exist (`tools/int6_requant.py --bits 6 --filter
  self_attn` and `--bits 5 --filter linear_attn`), else `awq-attn`, else the FP8 weights.
- The decode copies live next to the FP8 weights (prefill's W8A8 GEMM reads those): +5.2 GB of GPU memory (server
  startup 61.1 GB allocated, was 57.5).
- `tests/scheduler_check.py --requant`: every output identical with and without speculation.
- Tool calling (85 tasks, 2048 tokens): 51, the same as the FP8 projections (two schema tasks swap, one each way).

## 10. Smaller items

- **Draft length vs context.** `_pick_k` compared k = 3 / 7 by cycle times measured at startup (~0 context). At long
  context a cycle also reads the target's KV once (the multi-row attention of section 6, whatever k) and the drafter's
  KV once per draft step; both are now added per slot from the cache formats (~240 GB/s: at 128k ~17 ms + ~1.1 ms per
  draft step), so the k = 7 / k = 3 trade-off reflects the real cost (calibrated on `bench/decode_bench.py` 8k vs 128k).
- **Kernel gaps.** A trace of an MTP cycle on the INT6 / INT5 weights (`bench/trace_summary.py`, which now also splits
  idle time by kernel pair) has 0.09 ms idle in 83.7 ms: programmatic dependent launch already overlaps every boundary
  that matters. No megakernel work is left to do.
- **FP8 prefill attention quality.** Section 8: per-layer fallback and per-dim analysis; the +0.25-0.4% stays.
- **GDN prefill.** A 2,048-token chunk at position 0 is 626 ms of GPU time; FLA's chunked delta-rule kernels are 92 ms of
  it (state recurrence 31, output 27, WY recompute 22, L2 norm 6.6, KK^T solve 5.5), SwiGLU + NVFP4 quantization 44.5,
  add + RMSNorm 30.5. The causal conv kernel now normalizes q and k per head itself (one warp per head, FLA's eps), so
  FLA runs without its l2norm pass: prefill 1-2% faster (2k 0.620 s, 8k 2.613 s); perplexity within the noise of the
  changed reduction order (WikiText 7.0785, code 1.8027 vs 7.0909 / 1.8012). The recurrence kernel is pinned to 2 warps
  on Blackwell by FLA (a Triton race) and the rest is autotuned; more needs a CUDA port of the chunked delta rule
  (~7% if it halved those 92 ms), as does fusing SwiGLU + quantization into the GEMM epilogue (up to ~7%).
- **Prefill chunk size** (`bench/prefill_bench.py --chunk`, after sections 14-15), seconds for 2k / 8k / 32k / 64k prompts:

  | Chunk | 2k | 8k | 32k | 64k |
  |---|---|---|---|---|
  | 1,024 | 0.555 | 2.355 | 11.18 | |
  | 1,536 | 0.562 | 2.282 | 10.80 | |
  | **2,048 (default)** | **0.534** | **2.260** | **10.78** | **26.22** |
  | 4,096 | 0.542 | 2.572 | 11.99 | 28.12 |
  | 8,192 | 0.547 | 3.745 | 16.80 | 38.07 |

  2,048 stays. Larger chunks lose in cuBLASLt's FP8 GEMMs (the attention / GDN projections): 478 ms for a 4,096-token
  chunk against 162 ms per 2,048 tokens, 47% slower per token (its algorithm choice at M = 4,096 on sm_121).
- **Dropping the FP8 copies of the re-quantized projections (5.2 GB) was not done.** Prefill's W8A8 GEMMs would have to
  rebuild FP8 weights from the INT copies every chunk: ~12 GB more traffic per 2,048-token chunk (~5-8% slower prefill)
  and a second rounding of the prefill weights, for memory that is not short (61 GB allocated, 80 GB cap).

## 11. Sub-4-bit MLP weights: not viable as scalar formats

The MLP is 17.1B parameters, ~9.6 GB of the ~15.5 GB decode reads per token; each bit per weight saved is ~2.1 GB (~13%).
`tools/lowbit_sim.py` quantizes the BF16 originals to scalar codebooks with an e4m3 scale per block. Weight error vs BF16
(gate / up / down of layers 2, 31, 60, all alike):

| Format | Bits / weight | Error |
|---|---|---|
| NVFP4 as shipped | 4.5 | 8.4-8.8% |
| INT4, block 16 | 4.5 | 8.3-8.4% |
| INT3 (±0.5 .. ±3.5), block 8 | 4.0 | 13.5% |
| **NF3 (8 normal quantiles), block 16** | **3.5** | **14.8%** |
| INT3 (±0.5 .. ±3.5), block 16 | 3.5 | 15.7% |
| INT3 (-3 .. 3), block 16 | 3.5 | 18% |
| INT3 (±0.5 .. ±3.5), block 32 | 3.25 | 17.4% |
| INT2 (±0.5, ±1.5), block 16 | 2.5 | 32% |

Perplexity with NF3 (the best 3.5-bit format), attention / GDN as shipped (6.9698 / 1.7257):

| MLP in NF3 | WikiText | Python code |
|---|---|---|
| All 64 layers | 7.2742 (+4.4%) | 1.8701 (+8.4%) |
| Layers 16-47 | 7.1142 (+2.1%) | 1.8095 (+4.9%) |

The default (section 9) leaves ~0.27% of the code budget; even a few NF3 layers would spend it. Below 4 bits the
remaining route is vector / trellis quantization with Hessian-aware rounding (QTIP, EXL3), which published results put
at roughly +1-2% perplexity at 3 bits on large models over 16-bit weights, also beyond the gate here (the shipped NVFP4
MLP is already +0.4% over BF16: 6.9698 vs 6.9404). Not pursued.

## 12. Two draft chains per cycle: measured, not built

A second chain from the drafter's second-choice first token (PLAN.md 4.5 item 3, the smallest tree) only helps when the
first draft is wrong and the second right; after that it continues exactly like the teacher-forced unroll. So
`tools/twochain_sim.py` replays speculative cycles along the 75 held-out replies with the fine-tuned head and counts
tokens per cycle; its one-chain numbers reproduce the engine's (`tools/eval_drafter.py`: 2.76 / 3.39).

| | k=3, one chain | two chains | gain | k=7, one chain | two chains | gain |
|---|---|---|---|---|---|---|
| All | 2.76 | 2.96 | +7.2% | 3.41 | 3.71 | +8.8% |
| Prose | 2.60 | 2.81 | +8.2% | 3.05 | 3.35 | +9.8% |
| Q&A | 2.60 | 2.82 | +8.4% | 3.09 | 3.38 | +9.4% |
| Code | 3.15 | 3.31 | +5.0% | 4.43 | 4.74 | +7.0% |
| Structured | 2.82 | 3.00 | +6.4% | 3.58 | 3.83 | +6.9% |

The second chain doubles the verify rows: a k=3 cycle 4 -> 8 rows costs +5.5% (91.8 -> 96.9 ms), k=7 8 -> 16 rows +9%
(103.6 -> 112.6 ms), plus GDN rows and its own drafting. Net: about +1.5% at k=3 and nothing at k=7, and nothing when
two or three requests share a cycle. Not built.

## 13. A two-layer drafter: no gain

`tools/train_drafter.py --extra-layers 1` stacks a second decoder layer on the MTP head: a copy of the trained layer with
its output projections (o_proj, down_proj) at zero, so training starts exactly at run 4 (checked: identical unroll
outputs); +372M parameters, lr 5e-5 for the new layer, 2 epochs on the same 3,827 replies.

| Held-out agreement | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---|---|---|---|---|---|
| Run 4 (start) | 0.812 | 0.739 | 0.700 | 0.680 | 0.667 |
| After epoch 0 | 0.804 | 0.729 | 0.691 | 0.667 | 0.653 |
| After epoch 1 | 0.811 | 0.737 | 0.701 | 0.681 | 0.667 |

With the multi-layer features of section 7 and two chains (section 12), that rules out the drafter's inputs, its depth
and the verify structure as the limit: on 1.4M tokens of the target's replies, this head has learned what the data holds.
The remaining drafter lever is far more data (5-10x: about a day of generation and training on this machine).

## 14. SwiGLU + NVFP4 quantization in the up GEMM's epilogue

The prefill MLP ran the stacked gate | up GEMM to bf16 [M, 2 × 17,408], then `k_silu_mul_quant` read that back to write
the NVFP4 input of the down GEMM: ~300 MB of traffic per layer at 2,048 tokens around the GEMMs, 44.5 ms per chunk
(section 10). Now the gate GEMM writes bf16 g, and the up GEMM's epilogue computes silu(g) · α · acc and quantizes it
to NVFP4 with e4m3 scales per 16 in the layout the down GEMM reads (`csrc/gemm_nvfp4.cu`, `SwigluNvfp4`: a CUTLASS
epilogue visitor tree on SM120's block-scale-factor store, which scales by amax · nc / 6 with nc = 1 / the down
projection's input scale, as `quant_nvfp4` does). Traffic per layer drops to ~160 MB.

- Same output as the unfused path up to rounding ties (99% of the e2m1 codes equal; the fused path no longer rounds u
  and the product to bf16): error against fp32 0.131 / 0.158 both ways; `tests/test_gemm_nvfp4.py`.
- Per layer at 2,048 tokens: 2.81 ms (GEMM 2.10 + silu_mul_quant 0.71) -> 2.34 ms.
- Prefill: 2k 0.619 -> 0.596 s (3,434 tok/s), 8k 2.598 -> 2.503 s (3,273 tok/s), 3.7% faster.
- Perplexity (prefill engine): WikiText 7.0707, code 1.8026 (7.0785 / 1.8027 before). `COLINFER_FUSED_SWIGLU=0`: old path.

## 15. Gated DeltaNet chunked prefill in CUDA

FLA's `chunk_gated_delta_rule` was 92 ms of a 2,048-token chunk (48 layers: cumsum, KK^T + triangular solve, w / u
recompute, the state recurrence writing every chunk's state, the output pass; the recurrence pinned to 2 warps on
Blackwell). `csrc/gdn_prefill.cu` does the same math in two kernels (per value head, 64-token chunks, in-chunk cumulative
decay G; see the file header):

- `k_wy`, all chunks in parallel: A = strictly-lower β_i (k_i·k_j) e^(G_i − G_j) from a tensor-core K K^T, then
  T = (I + A)^-1 by 16 × 16 blocks (diagonal blocks by substitution, then three block rows of small products).
- `k_chunk`, one block per (value head, 64-wide V slice), sequential over chunks with the state on chip (fp32 registers,
  bf16 copy in shared memory): U = T diag(β) V, W = T diag(β e^G) K, V_new = U − W S, P = (Q K^T) ∘ decay mask,
  O = scale (e^G ∘ Q S + P V_new), S ← e^(G_C) S + K^T (e^(G_C − G) ∘ V_new), all on mma.sync bf16; the next
  chunk's K / V stream in with cp.async.

Against FLA (`tests/test_gdn_prefill.py`): output and final state within 0.2-0.35% relative (bf16 rounding).

Per layer at 2,048 tokens: FLA 1.93 ms, CUDA **0.82 ms** (k_chunk 0.61, k_wy 0.19). How it got there:

- k_wy first solved column by column (64 threads, local-memory chains: 0.575 ms); the blocked inverse, and T sharing K's
  shared memory (two blocks per SM), brought it to 0.19 ms.
- k_chunk first loaded T and Q rows and the strided g / β with synchronous global loads at every chunk (0.76 ms). Now
  k_wy also writes each chunk's G and β contiguously, and T, Q, V stream into single shared buffers refilled as soon as
  the chunk is done with them (T after V_new, V and Q after the state update): 0.61 ms in 99.3 KB of shared memory.
- The state update reads V_new decayed to the chunk end, rounded to bf16 once from fp32, as FLA does. (Scaling the
  stored bf16 V_new instead, a second rounding, moved WikiText perplexity +0.27%.)

| Prompt | FLA | CUDA |
|---|---|---|
| 2k | 0.593 s | **0.536 s** (3,818 tok/s) |
| 8k | 2.485 s | **2.290 s** (3,577 tok/s) |
| 32k | 11.64 s | **10.87 s** (3,014 tok/s) |
| 126k (passkey TTFT) | 76.6 s | **66-67 s** |

Perplexity (prefill engine, FP8 KV): WikiText 7.0807, code 1.8010, code ctx 8192 1.6073 against 7.0707 / 1.8026 / 1.6107
with FLA (differences of both signs, the size other rounding-only changes produce). Passkey 6/6 at 31k and 126k;
`tests/scheduler_check.py` passes. `COLINFER_GDN_CUDA=0` selects FLA.

## 16. DFlash2 (block-diffusion drafter) vs the MTP head

SGLang + DFlash2 (`z-lab/Qwen3.8-27B-DFlash2`: 1.9B parameters, 5 non-causal layers over the target's residual stream
after layers 5 / 19 / 33 / 47 / 61, an 8-token block per forward, a top-16 candidate selector) is reported at 40-50 tok/s
on one DGX Spark (MT-Bench acceptance 4.10 per verify, HumanEval 4.39). `tools/dflash_sim.py` ports its forward (from
SGLang's `srt/models/dflash.py`) and replays greedy cycles with our target's hidden states.

**Acceptance (tokens per verify step) on the engine's greedy replies to the 40 eval prompts** (half with thinking on):

| | MTP k=7, fine-tuned head (ours) | DFlash2, our port | DFlash2 in SGLang 0.5.21 | best of MTP and DFlash2 per cycle |
|---|---|---|---|---|
| Code | 5.64 | 5.27 | 4.85 | 5.33 |
| Prose | 2.79 | 2.83 | 2.91 | 2.95 |
| Q&A | 3.35 | 3.29 | 3.29 | 3.42 |
| Structured | 3.56 | 3.54 | 3.30 | 3.68 |
| All | 3.45 | 3.42 | 3.37 | 3.54 |

- The port agrees with SGLang's own DFlash2 (3.42 vs 3.37 overall), so the comparison is fair: on these prompts DFlash2
  drafts no better than the fine-tuned MTP head. The published 4.1-4.4 come from easier benchmarks.
- Two chains, one from each drafter, would add < 5% tokens per cycle for twice the verify rows: not worth it.

**End to end, the same 40 requests** (greedy, 256 tokens, token-id prompts, wall time including prefill):

| tok/s | SGLang + DFlash2 | colinfer (defaults) |
|---|---|---|
| Code | 41.2 | **50.7** |
| Prose | 25.2 | **30.0** |
| Q&A | 28.3 | **33.7** |
| Structured | 28.4 | **34.5** |
| All | 29.0 | **34.8** |

The engine is ~20% faster than SGLang + DFlash2 on identical requests. Prose stays near 30 tok/s with either drafter:
~0.45 acceptance per drafted token is what both learn on this model's prose.

## 17. Where a speculative cycle's time goes, and an NVFP4 drafter

A trace of a k=7 cycle at 8k on the INT6 / INT5 weights (`bench/traces/cycle_k7_8k_int`, 98.6 ms):

| Part | Time |
|---|---|
| GDN commit of the accepted rows + MTP drafting (catch-up row + 6 chained steps) | 22.9 ms (20.3 ms of weight GEMMs) |
| Target verify, 8 rows: 64 layers + lm_head | 75.7 ms (≈15.4 GB: ≈204 GB/s against ≈238 peak) |

Each draft step re-streams the MTP head (FP8 projections + MLP: 423 MB) and the draft lm_head (NVFP4, 69,632 rows: ≈200
MB). `Bf16Linear.to_lowbit` (`engine/spec/mtp.py`) now gives the head INT6 / INT5 / NVFP4 decode copies (round-to-nearest
from BF16, block-16 e4m3 scales) on the existing skinny GEMM paths; `COLINFER_MTP_FORMAT` selects one.

| Drafter weights | k=3 cycle | k=7 cycle | Tokens per cycle k=3 / k=7 (`tools/eval_drafter.py`) |
|---|---|---|---|
| FP8 (before) | 91.6 ms | 103.8 ms | 2.78 / 3.45 |
| INT6 | 90.6 ms | 101.5 ms | 2.77 / 3.44 |
| INT5 | | | 2.77 / 3.43 |
| **NVFP4 (default)** | **89.1 ms** | **98.4 ms** | **2.77 / 3.41** |
| NVFP4, draft vocabulary 32k | 88.1 ms | 95.8 ms | 2.70 / 3.30 |
| NVFP4, draft vocabulary 16k | 87.8 ms | 94.3 ms | 2.60 / 3.12 |

(Cycle times with the FP8 target projections; the draft vocabulary stays 64k: a smaller one loses as much acceptance
as it saves time.) Drafts change speed, never outputs: `tests/spec_check.py` and `tests/scheduler_check.py` pass.

The 40-request comparison of section 16, rerun (server defaults; tok/s):

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| FP8 drafter | 50.7 | 30.0 | 33.7 | 34.5 | 34.8 |
| **NVFP4 drafter** | **53.9** | **31.3** | **36.3** | **36.5** | **36.8** |
| SGLang + DFlash2 | 41.2 | 25.2 | 28.3 | 28.4 | 29.0 |

What is left between a k=7 cycle (≈93 ms now) and its byte floor (≈70 ms at 238 GB/s): the verify pass streams at
≈86% of peak (the long-K down projection and the small output projections are the weaker shapes; GDN recurrence,
attention and norms are ≈4-5 ms of serial work per cycle), and the drafter still streams ≈0.44 GB per step.

## Next

- A better drafter for prose. Acceptance there is about 0.45-0.50, so speculation adds about 1.15×. Sections 7, 12, 13:
  features, two chains and a second layer do not help; more data is what is left.
- Tensor-core multi-row decode attention: done (section 6).
- FP8 (Q K^T) prefill attention for long prompts: done (section 8), 6% at 64k, 11% at 128k.
- GDN projections below 8 bits: done (section 9), INT5 GDN + INT6 attention, decode 9.5% faster than FP8.
- A scheme that keeps Q's precision in layers 23-51 (their outputs move 2-4% under e4m3 Q) would remove most of its
  +0.25-0.3% perplexity.

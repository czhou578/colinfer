# Phase 6: pushing past the public numbers (progress, 2026-10-04)

> Dated log. On 2026-10-06, the project removed from the code the alternatives that this log measured and the switches
> that it names. These include the `COLINFER_*` environment variables, the CUDA-core GEMV, the bf16 / fp4 KV caches,
> the NVFP4 / AWQ / GPTQ decode copies, FLA, the n-gram drafter and the FP8 / INT drafter formats. Only the choice of
> each measurement stays. `docs/architecture.md` describes the current engine.

Three changes so far. All of them keep the greedy and seeded-sampled output token-identical across batch widths, draft
lengths and speculation on/off (`tests/scheduler_check.py`, `tests/spec_check.py`).

1. **Tensor-core skinny GEMM** (`csrc/skinny.cu`) for each decode-path linear.
2. **Adaptive draft length:** k = 3 or 7, chosen for each cycle.
3. **NVFP4 re-quantization** of the FP8 attention and GDN projections of the checkpoint, for decode.

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

Plain decode without speculation, 8k context (`bench/decode_bench.py`): 12.8 → **15.1 tok/s** with the re-quantized
weights. This would meet the frozen target of `docs/history/baseline.md` (≥ 14.0 tok/s) and its 14.5 stretch.

**The re-quantization fails its quality gate on code, so it is opt-in** (`--decode-weights requant`, section 3). With
the default FP8 projections:

- Single requests run at 66 / 71 / 50 / 24 tok/s on code edit / JSON / code generation / prose.
- Plain decode runs at 12.8 tok/s.

**Harness.** `~/Projects/model-benchmarks` core_runner, with the same settings as the Phase 0 baselines. 256-token
greedy outputs, 8 requests per concurrency level.

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

This is the decode test of the harness. Without speculation, the default server decodes at 12.6-12.7 tok/s, and at
15.3-15.4 with the re-quantized weights.

## 1. Skinny GEMM on tensor cores

**Problem.** The Phase 2 GEMV does M fused multiply-adds per weight on CUDA cores. It streams at 98% of the bandwidth
for 1-4 rows, but it becomes compute-bound past that: +45% time at 8 rows, 2-4× at 12-16. This limited speculation to
k=3 for one request, and to k=1 for three.

**Kernel.**

- `mma.sync.m16n8k16` with BF16 × BF16 → fp32. The kernel dequantizes the weights exactly into BF16, because e2m1 ×
  e4m3 has at most 5 significant bits.
- The weights stay in the layout of the checkpoint. These are the same tensors that the CUTLASS prefill reads, so there
  is no second copy.
- A warp owns 16 weight rows and walks K in 256-byte chunks per row.

**Read pattern.** What I learned on this LPDDR5x:

| Contiguous run per load instruction | Read rate on a 600 MB tensor |
|---|---|
| 512 B | 238 GB/s |
| 256 B | 235 GB/s |
| 128 B | ~225 GB/s |
| 64 B | 190-210 GB/s |

The rows of the table are 1 row × 512 B, 2 × 256, 4 × 128 and 8 × 64 B per instruction. A direct load of mma fragments
from memory gives the 64-byte pattern, because each lane needs its own row. Thus the kernel loads 2 rows × 256 B per
instruction into registers, one chunk ahead. Then it transposes the data into fragment order through a per-warp
shared-memory scratch, with only `__syncwarp`. Thus the warps never wait for each other.

These did not work:

- **Block-wide activation staging with barriers:** 10% slower.
- **A `cp.async` shared-memory pipeline:** it takes the L1 away from the activations. The long-K down projection fell
  to 100 GB/s at M=16, because the activation reads thrashed to L2.

**Details.**

- **Fragment order:** a per-lane K-permutation puts the B fragments of a lane in contiguous scratch bytes, and its A
  fragments in 32 contiguous activation bytes.
- **Block scales:** the kernel loads four chunks at a time (128 B per row).
- **Deterministic split-K:**
  - The split count depends only on the matrix shape.
  - The last item that finishes a tile adds the partials in a fixed order.
  - This removes the tail of grids that are only 1.3 waves.

**Bit identity.** Each output row is bit-identical for any M from 1 to 16: the rows beyond M are zeros. All decode
paths use the same kernel, so speculation never changes the outputs.

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

**Plain decode keeps the GEMV.** At 1-3 rows, the GEMV is still about 4% faster. `set_linear_kernel` chooses the kernel
for each process. The server uses the skinny GEMM with `--spec mtp` and the GEMV with `--spec none`. Each mode is
consistent inside itself.

## 2. Adaptive draft length

**Mechanism.**

- One graph exists for each (slot range, greedy/sampled, k ∈ {3, 7}) where the verify rows fit 16.
- Each cycle picks the k with the most expected tokens per second. This is the sum over the slots of
  (1 − a^(k+1)) / (1 − a), divided by the cycle time measured at startup.
- a is the decayed per-token acceptance rate of the slot. It starts at 0.6.
- Drafts beyond the k of the last cycle are stale. They do not count against the acceptance.

**Result.** Code and JSON run at k=7, and prose at 3. The cycles of one request cost:

| k | Checkpoint weights | Re-quantized weights |
|---|---|---|
| 3 | 94 ms | 83 ms |
| 7 | 111 ms | 100 ms |

## 3. NVFP4 re-quantization of the FP8 projections

**What it is.** `nvidia/Qwen3.8-27B-NVFP4` keeps 208 linears in FP8: attention q/k/v/o and GDN
in_proj_qkv/z/out_proj. These are 7.2 GB of the 17.6 GB read per token. `tools/requant_nvfp4.py` quantizes the same
tensors from the BF16 checkpoint to NVFP4:

- Each 16-value block gets an e4m3 scale. The tool chooses it among amax/6 ... amax/4 by the squared error.
- Stacked projections share a global scale, so they stay one launch.
- The result is 4.06 GB, written in 56 s.

**Use.** Decode streams these copies. Prefill keeps the FP8 weights on its W8A8 GEMM. The weight error increases from
2.7% to 8.5% relative.

**WikiText hides the cost, and code shows it.** Perplexity, ctx 2048, 65,504 tokens, weights dequantized to BF16
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

**Where the code loss comes from.** It is spread evenly. The re-quantization of one subset at a time gives:

| Subset | Code perplexity |
|---|---|
| GDN `in_proj_qkv` | +0.62% |
| GDN `in_proj_z` | +0.38% |
| GDN `out_proj` | +0.59% |
| attention q/k/v/o | +0.66% |

This is the round-to-nearest NVFP4 error, not one fragile layer.

**What GPTQ did.** It decreases the error that the calibration inputs see from 5-8% to 2-4% per layer. But it increases
the weight error to 11-12%. Text that is not like the calibration set gets worse. This explains the WikiText result
above.

**The damping sweep.** The calibration set was 64 sequences of WikiText-103 train plus 64 of `transformers` modeling
code. It did not overlap either evaluation corpus. The damping trades WikiText against code, and no setting gets code
below +1.4%.

**Conclusion.** For these projections, NVFP4 costs real quality on code, about 1.5% perplexity. This is probably why
NVIDIA left them in FP8.

**Status.**

- The full re-quantization stays opt-in: `--decode-weights requant`. It uses the GPTQ damping-0.3 file when the file
  is present.

**AWQ, and why only attention passed.** `tools/awq_nvfp4.py` scales each input channel j by s_j = E[x_j²]^(α/2)
before the quantization (W diag(s)). Decode divides the activations by s (`_awq_in` in `engine/model/fast.py`). The
tool searches α for each group of linears that share an input. It uses the output error that the calibration inputs see
(128 sequences, half WikiText-103 train, half `transformers` code, the same Hessians as GPTQ).

α = 0 is round-to-nearest, so the calibration error can only decrease. It decreased by 17% (α ≈ 0.5 for all groups):
attention qkv 0.052 → 0.035, GDN qkvz 0.059 → 0.044, o_proj / out_proj ~10%. Perplexity (reference engine, ctx 2048,
65,504 tokens, shipped weights 6.9698 / 1.7257):

| NVFP4 for | WikiText | Python code |
|---|---|---|
| All 208, round-to-nearest | 6.9789 (+0.13%) | 1.7650 (+2.3%) |
| All 208, AWQ | 7.0419 (+1.03%) | 1.7508 (+1.45%) |
| AWQ, attention q/k/v only | 6.9719 (+0.03%) | |
| AWQ, attention o_proj only | 6.9657 (-0.06%) | |
| AWQ, GDN in_proj_qkv / z only | 7.0097 (+0.57%) | |
| AWQ, GDN out_proj only | 7.0048 (+0.50%) | |
| **AWQ, attention q/k/v/o (64 linears)** | **6.9663 (-0.05%)** | **1.7332 (+0.43%)** |

- AWQ helps attention (round-to-nearest attention alone cost code +0.66%). It hurts Gated DeltaNet on WikiText, but it
  also decreases the output error of those linears. The errors of the GDN projections go through the recurrent state.
  There, a per-linear output-error objective does not predict their cost. GPTQ showed the same mismatch.
- The scales stay within 0.2-5 (no channel goes to near zero), so this is not a range problem.
- Attention-only AWQ passes the gate (≤ 0.5% on both corpora). **Thus `--decode-weights auto`, the default, now decodes
  the 64 attention linears from `attn_gdn_nvfp4_awq_attn.safetensors` when it exists** (`tools/awq_nvfp4.py --groups
  self_attn`). The GDN projections stay FP8.
- Speed (`bench/decode_bench.py`, 8k), ~3% faster:
  - plain decode 81.4 → 79.0 ms
  - MTP cycles k=3 91.9 → 89.2 ms, k=7 104.5 → 102.2 ms
  - width 3 k=3 101.8 → 99.4 ms
- The attention linears are 1.7B of the 7.2B FP8 parameters. The other 15% is in GDN. GDN needs a better format than
  NVFP4, or a quantizer that models the recurrence.
- `tests/scheduler_check.py --requant` with the AWQ file: all outputs are identical with and without speculation.
- Tool calling (85 tasks, 2048 tokens): 50 with AWQ attention, 51 with the FP8 projections. One schema task differs
  (`schema_title_max_length_fail`). The full re-quantization lost three (48).

**Tool calling.** On the 85-task suite at 2048 tokens, the scores differ only within the same 8 borderline tasks:

| Configuration | Tasks passed |
|---|---|
| Re-quantized weights | 48 |
| AWQ attention only | 50 |
| INT6 attention + INT5 GDN (the default since, section 9) | 51 |
| Skinny GEMM with the FP8 projections | 51 |
| Phase 5 / vLLM | 53 / 53 |

The gate to make a configuration the default: ≤ 0.5% on both corpora.

## 4. Decode-step overhead ("megakernel")

A trace of a plain decode step (`nsys`) shows that the GPU is already busy 98% of the time: 17.6 GB in 77.5 ms,
227 GB/s. This is about 95% of what this machine streams (238-240 GB/s). The remaining 5%:

- **1.5 ms of idle gaps** between kernels.
- **GDN state read / write:** 0.3 GB.
- **KV reads:** 0.27 GB at 8k.

A persistent megakernel could recover at most the gaps, about 2%. The two cheaper changes below recovered more. With
them, a later trace of an MTP cycle on the INT6 / INT5 weights of section 9 shows 0.09 ms idle in 83.7 ms (0.1%).
A megakernel has nothing left to gain.

- **L2 prefetch.** Before its first activation read, the skinny GEMM pulls its next two chunks into L2.
- **Programmatic dependent launch** (`csrc/pdl.cuh`).
  - Each decode-path kernel signals its dependents first.
  - The GEMM launches with programmatic serialization. Thus it starts to stream weights while the norm, attention or
    GDN kernel before it finishes. It waits (`griddepcontrol.wait`) only before it reads the activations.

| | Before | Prefetch + PDL |
|---|---|---|
| Plain decode step, 8k | 81.5 ms | 77-79 ms (12.6-13.0 tok/s) |
| Cycle, width 1, k=3 / k=7 | 94.4 / 111.3 ms | 89-92 / 106-108 ms |
| Cycle, width 3, k=3 | 110.5 ms | 103-106 ms |

`COLINFER_PDL=0` turns PDL off.

## 5. 4-bit KV cache

Opt-in with `--kv fp4`.

**Format.** The cache stores a (token, head) row of 256 values in 144 bytes: 128 bytes of e2m1 plus 16 e4m3 block
scales. This is 0.56× of FP8. The format stores the scales ×16. Thus block maxima from 0.006 to 168 stay in the normal
range of e4m3 (measured on this model: K 0.04-23, V 0.17-108).

**Implementation.**

- **Write:** the attention prologue quantizes on write, with 16-lane shuffles for the block maximum and nibble pairing.
- **Read:** the decode kernel dequantizes per lane.
- **Prefill:** it dequantizes the cached prefix to BF16 for FlashInfer (`kv4_to_bf16`).
- **Cross-slot copies:** the rows stay one tensor, so the cross-slot prefix copies work with no change.

**Quality.**

| Prefill-engine perplexity (W4A4), ctx 2048 | WikiText | Python code |
|---|---|---|
| BF16 KV | 7.0910 | 1.8015 |
| FP8 KV | 7.0909 | 1.8012 |
| FP4 KV | 7.1141 (+0.33%) | 1.8054 (+0.23%) |

- Passkey retrieval at 126k: 3/3.
- A choice of the scale of each block by squared error, which allows amax/5, made the perplexity worse (+0.51%). A
  clipped block maximum hurts attention more than coarser steps do.

**Speed at 128k context:**

| | FP8 KV | FP4 KV |
|---|---|---|
| Plain decode step | 96.0 ms | 93.2 ms |
| MTP cycle, k=3 / k=7 | 144 / 214 ms | 148 / 223 ms |

The bytes halve, but the dequantization costs more per element. Thus FP4 is mainly a memory option: 3 × 262k slots in
14.5 GB instead of 25.8 GB. With the tensor-core attention of section 6, fp4 is also faster at long context:

- 128k plain decode 89.1 vs 97.3 ms
- MTP cycles 107 / 134 ms vs 115 / 140 ms for k=3 / k=7
- width 3 k=3 153 vs 175 ms

**The real long-context cost is in multi-row attention.** The decode kernel runs one block per verify row, so a k-draft
cycle reads the KV cache k+1 times. At 128k, a k=3 cycle costs 144 ms, against 89 ms at 8k.

**What I tried.** One block per (KV head, split), with one warp per verify row. This reads the KV once and keeps each
row bit-identical to plain decode. It was 1.7× slower, because one warp per row on CUDA cores is too serial. Section 6
has the fix.

## 6. Tensor-core multi-row decode attention

`csrc/attn_decode.cu`, namespace `tc`. It is the default for fp8 and fp4 caches (`COLINFER_ATTN_TC=0` selects the
split-KV kernel). One block owns (slot, KV head, stripe of key tiles) and serves all query rows of the slot. These are
up to 48 rows (6 query heads × T new tokens), as three m16 tiles of `mma.sync.m16n8k16` (f16 operands, fp32
accumulate).

**Bit identity across T.** Plain decode (T = 1) and a k-draft verify must give a row the same bits. Thus nothing that
a row computes can depend on T:

- Keys go in 32-key tiles at fixed positions. Block z of 12 takes tiles z, z+12, z+24, ..., and the combine folds the
  12 partials in a fixed order. None of this depends on T or on the sequence length.
- The mma rows are independent. The softmax of a row is 8 lanes with fixed reduction trees, the same code for each row.
- A tile that is fully past the length of a row (a longer row of the same slot needs it) is an exact no-op. Its scores
  are -inf, p = 0, the running max does not move, and the rescale factor is exactly 1.

`tests/test_attn_decode_tc.py` checks each row of 2-12-row launches bit for bit against one-row launches.
`tests/spec_check.py` / `tests/scheduler_check.py` still pass: the greedy and seeded-sampled outputs are identical with
and without speculation, at each batch width.

**Operands.**

- The kernel uses K directly from the raw cache bytes in shared memory. The mma fragment of a lane for k-step j is head
  dims 16j + 4c .. +3 (c = lane % 4), one 32-bit load. This is a permutation of the k dimension. The fragment of Q
  repeats it (8 consecutive bytes of the f16 Q row). e4m3 values, and e2m1 × e4m3-scale values, are exact in f16.
- The kernel converts V to an f16 tile and reads it with `ldmatrix.trans`. It rounds P to f16. The error against fp32
  attention is ~5e-4 relative, below the rounding of the bf16 output.

**Two things that mattered for speed.**

| Change | 128k, fp8, T=1 | T=8 |
|---|---|---|
| First version (raw rows padded to 272 bytes against bank conflicts) | 189 GB/s | 189 GB/s |
| Rows 256 bytes apart, XOR-swizzled 16-byte chunks instead | 233 GB/s | 188 GB/s |
| Softmax on 8 lanes per row, 4 rows per warp at once (was a warp per row) | 232 GB/s | 232 GB/s |

The padding cost 20% of the cp.async streaming rate, even with the compute removed. A loads-only build streamed 187 GB/s
with 272-byte rows, and 232 with 256-byte rows. More pipeline stages (2 → 4) and contiguous per-block chunks changed
nothing.

**Per layer** (`bench/attn_bench.py`, 128k context):

| KV | Slots | T = 1 (plain) | T = 4 (k=3) | T = 8 (k=7) |
|---|---|---|---|---|
| fp8, split-KV | 1 | 1.21 ms | 3.11 ms | 6.09 ms |
| fp8, tensor-core | 1 | 1.16 ms | 1.14 ms | 1.16 ms |
| fp8, split-KV | 3 | 3.55 ms | 9.11 ms | 17.7 ms |
| fp8, tensor-core | 3 | 3.45 ms | 3.51 ms | 3.53 ms |
| fp4, split-KV | 1 | 1.06 ms | 3.29 ms | 6.46 ms |
| fp4, tensor-core | 1 | 0.66 ms | 0.67 ms | 0.87 ms |

fp8 streams at 228-236 GB/s at each T: a verify of 8 rows costs the same as plain decode. fp4 plain decode becomes 1.6×
faster than before (227 GB/s). Its T = 8 is compute-bound at ~175 GB/s (the dequantization and the three m-tiles).

**MTP drafter cache in fp8.** Each of the k draft steps of a cycle reads the own KV cache of the drafter (that of the MTP
layer). This cache was bf16: 512 MB per step per slot at 128k. It is now e4m3 like that of the target (`MtpState`,
`COLINFER_MTP_KV_FP8=0` for bf16), and the same kernel reads it.

The acceptance did not change. With the same target kernel, `tools/eval_drafter.py` gives 2.76 / 3.39 tokens per cycle
(k=3 / k=7) either way, with each category within 0.01. A first comparison suggested that fp8 cost code acceptance. But
it compared against an evaluation run from before the softmax change above, whose greedy text differs. The K and V of the
head are well inside the range of e4m3 (|K| median 0.95, max 19, |V| median 1.6, max 39).

**Decode cycle** (`bench/decode_bench.py`, fp8 KV. Before = split-KV attention and a bf16 drafter cache):

| | 8k before | 8k after | 128k before | 128k after |
|---|---|---|---|---|
| Plain decode, width 1 | 80.8 ms | 80.7 ms | 98.7 ms | 97.3 ms |
| MTP cycle, width 1, k=3 | 93.3 ms | 91.1 ms | 148.5 ms | **110.6 ms** |
| MTP cycle, width 1, k=7 | 109.6 ms | 103.4 ms | 219.1 ms | **126.8 ms** |
| MTP cycle, width 2, k=7 | 124.6 ms | 111.7 ms | 339.1 ms | **159.5 ms** |
| MTP cycle, width 3, k=3 | 108.5 ms | 100.8 ms | 269.1 ms | **160.5 ms** |

At 128k, a k=7 cycle went from 2.2× a plain step to 1.3×. At 8k, the cycles are 2-10% shorter.

## 7. Fine-tuning the MTP head as a multi-step drafter

The MTP head of the checkpoint has training for one step only: (embedding of x_{i+1}, target hidden h_i) → x_{i+2}.
The engine chains it, so drafts 2..k feed the head its own output instead of a target hidden state. The head never
trained on these inputs.

`tools/train_drafter.py` fine-tunes the head in the EAGLE-3 style ("training-time test"). It unrolls each row to depth
D exactly as the draft steps run. The loss is a soft cross-entropy against the top-32 next-token distribution of the
target (64k draft vocabulary), on the own replies of the target (`tools/drafter_data.py`). The embedding and the
lm_head stay frozen. `tests/test_drafter_train.py` checks the unroll against incremental chained drafting.

**Data:** 1,500 sampled replies (T = 0.7) to generated prompts, 490k tokens: 45% prose, 15% Q&A, 25% code, 15%
structured, thinking on for half. We hold out 75 replies.

**Runs** (held-out top-1 agreement with the target, per draft depth):

| | lr | Depth | Epochs | Depth 1 | Depth 3 | Depth 5 | Depth 7 |
|---|---|---|---|---|---|---|---|
| Checkpoint head | | | | 0.802 | 0.672 | 0.619 | 0.586 |
| Run 1 | 3e-5 | 5 | 2 | 0.808 | 0.694 | 0.655 | |
| Run 2 (stopped after epoch 0: worse than before) | 1e-4 | 7 | 3 | 0.769 | 0.626 | 0.582 | 0.561 |
| Run 3 | 3e-5 | 7 | 4 | 0.807 | 0.692 | 0.657 | 0.635 |
| Run 4 (the default), 2.7× the data | 3e-5 | 5 | 2 | 0.812 | 0.700 | 0.668 | |

**End to end** (`tools/eval_drafter.py`: 40 fresh prompts, greedy, the real speculative cycle of the engine. Tokens per
cycle, with the acceptance in parentheses):

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| Checkpoint head, k=3 | 3.25 (0.75) | 2.45 (0.49) | 2.57 | 2.71 | 2.67 |
| Run 1, k=3 | 3.45 (0.82) | 2.50 (0.50) | 2.67 | 2.77 | 2.75 |
| Run 3, k=3 | 3.41 (0.80) | 2.50 (0.50) | 2.63 | 2.76 | 2.74 |
| Checkpoint head, k=7 | 4.46 (0.50) | 2.74 (0.25) | 3.01 | 3.22 | 3.16 |
| Run 1, k=7 | 5.29 (0.62) | 2.84 (0.27) | 3.16 | 3.39 | 3.36 |
| Run 3, k=7 | 5.18 (0.61) | 2.84 (0.27) | 3.19 | 3.40 | 3.36 |

- The fine-tune helps most on code (+19% tokens per cycle at k=7) and least on prose (+4%).
- The deeper unroll and twice the epochs of run 3 gained nothing end to end over run 1. More training on the same
  1,500 replies no longer helps.

**Run 4: more data.** We generated 2,402 more replies from new prompts (`drafter_data.py --seed 2`, because seed 1 is for the
evaluation), 942k tokens. This gives 3,827 training replies and 1.43M tokens. The held-out split is the same 75
replies as before, because it comes from the first data file only.

We ran the same evaluation again with the current attention kernel. The greedy text of the target differs slightly from
the table above, because section 6 changed the order of its softmax reduction. Thus compare only within this table:

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| Checkpoint head, k=3 | 3.13 | 2.44 | 2.66 | 2.70 | 2.67 |
| Run 1, k=3 | 3.31 | 2.49 | 2.75 | 2.76 | 2.75 |
| Run 4, k=3 | 3.36 | 2.48 | 2.72 | 2.82 | 2.76 |
| Checkpoint head, k=7 | 4.16 | 2.72 | 3.12 | 3.19 | 3.14 |
| Run 1, k=7 | 4.80 | 2.82 | 3.39 | 3.34 | 3.35 |
| Run 4, k=7 | 4.97 | 2.82 | 3.34 | 3.46 | 3.39 |

- With 2.7× the data: held-out agreement +1.3 points at depth 5, code and structured output +3-4% at k=7, prose flat.
- Run 4 is now `~/.cache/colinfer/drafter/mtp_ft.safetensors`, which the server loads by default (`--drafter-weights
  auto`). We keep run 1 as `mtp_ft_r1.safetensors`.
- With this head, prose stays at ~0.5 acceptance per token. A fine-tune of a one-layer head on more of the same data
  helps code-like text, not prose.

**EAGLE-3-style multi-layer features: no gain.** EAGLE-3 gives its drafter the low, middle and high-layer hidden states
of the target, not only the last one. `prefill(return_layers=...)` returns the residual stream after the given layers.
`train_drafter.py --layers 3,23,43` fuses them into the depth-1 hidden input:
h0 = h + B A [rms(l3), rms(l23), rms(l43)], rank 256, B = 0 at the start (exactly run 4). We trained it from run 4 on
the same 3,827 replies (features 80 GB, streamed from disk one shard at a time).

| | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---|---|---|---|---|---|
| Run 4 (start) | 0.812 | 0.739 | 0.700 | 0.680 | 0.667 |
| + layer fusion, epoch 0 | 0.806 | 0.730 | 0.691 | 0.667 | 0.653 |
| + layer fusion, epoch 1 (end) | 0.811 | 0.737 | 0.701 | 0.681 | 0.667 |

- The trained fusion term is less than 1% of the RMS of the final hidden. Thus the gradient found no draft signal in
  the earlier layers that the final hidden does not have. (The dip at epoch 0 is the restart at the full learning rate.
  Run 4 shows the same dip.)
- A full-rank fusion ([0 0 0 I] + learned, 105M parameters, lr 1e-4) was worse: -2 points after one epoch. We stopped it.
- We did not integrate this into the engine. With one decoder layer and ~1.4M training tokens, the capacity and the data
  of the drafter limit it on prose, not its inputs.
- The remaining routes: a larger drafter (several layers, or a small parallel drafter model), trained on much more data.
  This is days of generation and training on this machine.

## 8. Long-prompt prefill: where the time goes

Prefill (`bench/prefill_bench.py --kv-fp8`, chunk 2048): 3,057 tok/s at 8k, 2,556 at 32k, 2,017 at 64k. The per-kernel
GPU time of the 2,048-token chunk that follows 30,720 prompt tokens (`--profile --profile-at 30720`) is 998 ms:

| Kernel | ms | Share | Rate |
|---|---|---|---|
| FlashInfer causal prefill attention (16 layers) | 327 | 33% | 78 TFLOPS |
| CUTLASS NVFP4 W4A4 GEMMs (MLP) | 239 | 24% | ~293 TFLOPS (peak ~355) |
| cuBLASLt FP8 GEMMs (attention / GDN projections) | 180 | 18% | ~164 TFLOPS (peak ~200) |
| SiLU-mul + quantize, add + RMSNorm, GDN chunk kernels, conv, norms | ~250 | 25% | |

The GEMMs are near their peaks. Attention is the term that grows with the context. These are its alternatives on this
shape (2,048 queries, 6:1 GQA, head_dim 256, lower-right causal, `torch` 2.13):

| Context | FlashInfer | torch SDPA flash | torch SDPA mem-efficient | cuDNN |
|---|---|---|---|---|
| 8k | 87 TFLOPS | 68 | 27 | no kernel for sm_121 |
| 32k | 79 | 62 | 25 | |
| 64k | 79 | 62 | 25 | |

FlashInfer runs at ~81% of the measured BF16 GEMM peak (96 TFLOPS), so our own BF16 kernel has little room. The
opportunity is the FP8 tensor cores (~200 TFLOPS). Q K^T in e4m3 can read the FP8 K cache as it is, with Q quantized per
(token, head). The m16n8k32 fragment of a lane is 4 consecutive head dims of one key, so no permutation is necessary.

The quality cost, emulated in the prefill engine (`COLINFER_EMU_Q_FP8=1`, Q rounded to e4m3 before FlashInfer, FP8 KV,
perplexity):

| | WikiText ctx 2048 | Code ctx 2048 | Code ctx 8192 |
|---|---|---|---|
| BF16 Q | 7.0909 | 1.8012 | 1.6085 |
| e4m3 Q, per (token, head) scale | 7.0777 | 1.7985 | 1.6105 (+0.12%) |

### The FP8-QK prefill attention kernel (`csrc/attn_prefill.cu`)

- **FA2 structure.** A block is 64 query rows of one query head (4 warps × 16 rows). The KV tiles of 32 keys are
  double-buffered with cp.async, and each SM runs two blocks. The heaviest causal tiles start first. The 6 heads of a KV
  head are adjacent, so they share their KV stream in L2.
- **S = Q K^T on `mma.m16n8k32` e4m3.** The kernel rounds Q to e4m3, with a per (token, head) scale, and holds it as A
  fragments in registers. It reads K as stored. A lane loads 16 consecutive bytes of a K row (one 128-bit shared load per
  two k32 steps). The head-dim order inside the dot product has the same permutation for Q. The rows are 256 bytes
  apart, with the chunks XOR'ed by the row parity, so there are no bank conflicts.
- **Softmax.** The online softmax runs in registers. The kernel rounds P to f16 and uses it as the A fragment of P V (f16
  `m16n8k16`, with V converted from e4m3 to an f16 tile, `ldmatrix.trans`).
- **Accuracy** against fp32 attention with the same e4m3 Q: 0.4-0.8% relative on random data, and 0.17-0.35% on the own
  attention inputs of the model. This is the same as the error of FlashInfer with that Q
  (`tests/test_attn_prefill_fp8.py`).

**Attention alone** (2,048 queries after L - 2,048 cached tokens):

| Context | FlashInfer | FP8 kernel, 32-key tiles | 64-key tiles |
|---|---|---|---|
| 8k | 4.24 ms (85 TFLOPS) | 3.40 ms (106) | 4.15 ms (87) |
| 32k | 19.4 ms (82) | 15.4 ms (104) | 18.6 ms (86) |
| 64k | 39.4 ms (82) | 32.0 ms (102) | 37.4 ms (87) |

**The cost is the rounding of Q, layer by layer.** On real activations (a 2,048-token Python file), the rounding of Q to
e4m3 moves the attention outputs by 0.4-1.3% in the first and last few attention layers. In layers 23-51, it moves them
by 2-4%. A
few large q·k terms dominate the scores of these layers, and the 3-bit mantissa of e4m3 rounds them. Perplexity
(prefill engine, FP8 KV):

| | FlashInfer | FP8 kernel |
|---|---|---|
| WikiText, ctx 2048 | 7.0909 | 7.1086 (+0.25%) |
| Code, ctx 2048 | 1.8012 | 1.8053 (+0.23%) |
| Code, ctx 8192 | 1.6085 | 1.6135 (+0.31%) |
| Code, ctx 16384 | 1.5650 | 1.5691 (+0.26%) |

**The default: the FP8 kernel for the chunks whose context is past 16k** (`COLINFER_ATTN_FP8_MIN_CTX`, 16384, off with
`COLINFER_ATTN_FP8_PREFILL=0`). There, attention is a large share of the time. Shorter prompts use exactly the same
computation as before. The prefill of the MTP head follows the same rule.

| Prompt | FlashInfer | Default (FP8 past 16k) |
|---|---|---|
| 8k | 2.64 s | 2.65 s (unchanged path) |
| 32k | 12.70 s | 12.44 s (-2%) |
| 64k | 32.05-32.32 s | 30.27-30.49 s (-5.6%) |
| 128k | 90.9 s | 80.9 s (-11%) |

**Attempts to remove the quality cost.**

- FlashInfer for layers 23-51 (their outputs move most): WikiText goes back to +0.03%, but code is +0.27% at ctx 2048
  and +0.43% at 8k. This is no better than all-FP8: the per-layer output error does not predict the code perplexity.
- Which head dims carry the error (real activations)? If the 32 dims with the largest mean |q|·|k| stay in bf16, the
  error of the middle layers goes from 3-4% to 1-1.5% (64 dims: ~1%). These are not the RoPE dims, and they are spread
  over most 16-dim chunks. But the kernel can give special treatment only to whole 16-byte K chunks. Thus a residual-Q
  correction would cost most of the speedup.
- Conclusion: +0.25-0.4% is the price of e4m3 Q at this speed. It stays limited to the chunks past 16k.

Passkey retrieval with this kernel: 6/6 at 31k and 126k. `tests/scheduler_check.py` with the threshold at 0 (all
prefills on the kernel, chunked prefill, prefix reuse, MTP): passed.

## 9. INT5 / INT6 for the attention and GDN projections (the default)

NVFP4 for the 144 Gated DeltaNet projections fails the quality gate with each method (section 3: round-to-nearest,
GPTQ, AWQ, +0.4-0.6% code perplexity per projection type). The FP8 weights of the checkpoint are themselves 2.67% off
the BF16 originals. Decode needs a format with that error but fewer bits. The relative weight error on GDN layers 4 / 30
/ 52 (all three projections alike):

| Format (e4m3 scale per 16 weights + fp32 global scale) | Bits / weight | Error vs BF16 |
|---|---|---|
| Checkpoint FP8 (per-tensor scale) | 8 | 2.67% |
| INT6 | 6.5 | **2.24%** |
| FP6 e2m3 | 6.5 | 2.58% |
| INT5 | 5.5 | 4.2% |
| FP6 e3m2 | 6.5 | 4.5% |
| NVFP4 | 4.5 | 8.5% |

`tools/int6_requant.py` quantizes from the BF16 originals. It searches the block scale among amax/qmax × {1 ... 0.8} by
the squared error, and stacked projections share the global scale. Perplexity (reference engine, ctx 2048, shipped
weights 6.9698 / 1.7257):

| Decode copies | WikiText | Python code | Bytes per token saved vs FP8 |
|---|---|---|---|
| INT6, all 208 | 6.9697 (0.00%) | 1.7250 (-0.04%) | 1.35 GB |
| INT6 GDN + AWQ NVFP4 attention | 6.9445 (-0.36%) | 1.7310 (+0.31%) | 1.77 GB |
| INT5 GDN, FP8 attention | 6.9387 (-0.45%) | 1.7288 (+0.18%) | 1.74 GB |
| INT5, all 208 | 6.9431 (-0.38%) | 1.7333 (+0.44%) | 2.27 GB |
| **INT5 GDN + INT6 attention (default)** | **6.9552 (-0.21%)** | **1.7296 (+0.23%)** | **2.06 GB** |

INT6 has no loss here. INT5 on GDN costs a fifth of the gate.

**Kernel** (`csrc/skinny.cu`, formats INT6 / INT5). The file stores the low 4 bits of the code exactly like the
nibbles of NVFP4 ([N, K/2]). Thus they stream and land in the per-warp scratch the same way. The high 2 (1) bits are a
second plane [N, K/4] ([N, K/8]), 128 (64) bytes per row per 512-k chunk. In this plane, the 4 codes of a k-step share
one byte (nibble).

The dequantization builds bf16 directly. 0x4300 | c is 128 + c, and minus 160 (144) this is exactly q. The product
with the bf16 block scale rounds once (q × s has up to 9 significant bits, ~0.2% relative). The rows stay bit-identical
for each M (`tests/test_skinny_int.py`). Against FP8 on the GDN shapes: INT6 1.17-1.27× (bytes: 1.23×), INT5 1.32-1.45×
(1.45×).

**Decode** (`bench/decode_bench.py`, 8k):

| | FP8 | AWQ attention (previous default) | INT6 attention + INT5 GDN |
|---|---|---|---|
| Plain decode, width 1 | 81.8 ms | 78.8 ms | **74.0 ms** (13.5 tok/s) |
| Plain decode, width 3 | 88.0 ms | 85.6 ms | 80.7 ms |
| MTP cycle, width 1, k=3 / k=7 | 91.8 / 103.6 ms | 89.7 / 101.5 ms | **84.7 / 97.2 ms** |
| MTP cycle, width 2, k=7 | 112.6 ms | 110.6 ms | 105.2 ms |
| MTP cycle, width 3, k=3 | 101.8 ms | 99.1 ms | 95.5 ms |

- `--decode-weights auto` (the default) now picks `int` when both files exist (`tools/int6_requant.py --bits 6 --filter
  self_attn` and `--bits 5 --filter linear_attn`). If not, it picks `awq-attn`, and if that is also missing, the FP8
  weights.
- The engine keeps the decode copies next to the FP8 weights, which the W8A8 GEMM of prefill reads. This costs +5.2 GB of GPU
  memory (server startup 61.1 GB allocated, was 57.5).
- `tests/scheduler_check.py --requant`: all outputs are identical with and without speculation.
- Tool calling (85 tasks, 2048 tokens): 51, the same as the FP8 projections (two schema tasks swap, one each way).

## 10. Smaller items

- **Draft length vs context.** `_pick_k` compared k = 3 / 7 by the cycle times measured at startup (~0 context). At long
  context, a cycle also reads the KV of the target once (the multi-row attention of section 6, whatever k). It also reads
  the KV of the drafter once per draft step. The scheduler now adds both per slot from the cache formats (~240 GB/s: at
  128k ~17 ms + ~1.1 ms per draft step). Thus the k = 7 / k = 3 trade-off shows the real cost (calibrated on
  `bench/decode_bench.py` 8k vs 128k).
- **Kernel gaps.** A trace of an MTP cycle on the INT6 / INT5 weights (`bench/trace_summary.py`, which now also splits
  the idle time by kernel pair) has 0.09 ms idle in 83.7 ms. Programmatic dependent launch already overlaps each boundary
  that matters. No megakernel work remains.
- **FP8 prefill attention quality.** See section 8: the per-layer fallback and the per-dim analysis. The +0.25-0.4%
  stays.
- **GDN prefill.** A 2,048-token chunk at position 0 is 626 ms of GPU time:
  - The chunked delta-rule kernels of FLA are 92 ms of it (state recurrence 31, output 27, WY recompute 22, L2 norm
    6.6, KK^T solve 5.5).
  - SwiGLU + NVFP4 quantization: 44.5. Add + RMSNorm: 30.5.

  The causal conv kernel now normalizes q and k per head itself (one warp per head, the eps of FLA). Thus FLA runs
  without its l2norm pass: prefill is 1-2% faster (2k 0.620 s, 8k 2.613 s). The perplexity stays within the noise of the
  changed reduction order (WikiText 7.0785, code 1.8027 vs 7.0909 / 1.8012). FLA pins the recurrence kernel to 2 warps
  on Blackwell (a Triton race), and it autotunes the rest. More speed needs a CUDA port of the chunked delta rule (~7% if
  it halved those 92 ms). A fusion of SwiGLU + quantization into the GEMM epilogue could give up to ~7% more.
- **Prefill chunk size** (`bench/prefill_bench.py --chunk`, after sections 14-15), seconds for 2k / 8k / 32k / 64k
  prompts:

  | Chunk | 2k | 8k | 32k | 64k |
  |---|---|---|---|---|
  | 1,024 | 0.555 | 2.355 | 11.18 | |
  | 1,536 | 0.562 | 2.282 | 10.80 | |
  | **2,048 (default)** | **0.534** | **2.260** | **10.78** | **26.22** |
  | 4,096 | 0.542 | 2.572 | 11.99 | 28.12 |
  | 8,192 | 0.547 | 3.745 | 16.80 | 38.07 |

  2,048 stays. Larger chunks lose in the FP8 GEMMs of cuBLASLt (the attention / GDN projections): 478 ms for a
  4,096-token chunk, against 162 ms per 2,048 tokens. This is 47% slower per token (its algorithm choice at M = 4,096 on
  sm_121).
- **We did not drop the FP8 copies of the re-quantized projections (5.2 GB).** The W8A8 GEMMs of prefill would have to
  build the FP8 weights again from the INT copies for each chunk. This is ~12 GB more traffic per 2,048-token chunk (~5-8%
  slower prefill), and a second rounding of the prefill weights. The memory is not short (61 GB allocated, 80 GB cap).

## 11. Sub-4-bit MLP weights: not possible as scalar formats

The MLP has 17.1B parameters, ~9.6 GB of the ~15.5 GB that decode reads per token. Each bit per weight that we remove
saves ~2.1 GB (~13%). `tools/lowbit_sim.py` quantizes the BF16 originals to scalar codebooks with an e4m3 scale per
block. Weight error vs BF16 (gate / up / down of layers 2, 31, 60, all alike):

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

Perplexity with NF3 (the best 3.5-bit format), with attention / GDN as shipped (6.9698 / 1.7257):

| MLP in NF3 | WikiText | Python code |
|---|---|---|
| All 64 layers | 7.2742 (+4.4%) | 1.8701 (+8.4%) |
| Layers 16-47 | 7.1142 (+2.1%) | 1.8095 (+4.9%) |

The default (section 9) leaves ~0.27% of the code budget, and even a few NF3 layers would use it all. Below 4 bits, the
remaining route is vector / trellis quantization with Hessian-aware rounding (QTIP, EXL3). The published results put
this at roughly +1-2% perplexity at 3 bits on large models, over 16-bit weights. This is also past the gate here: the
shipped NVFP4 MLP is already +0.4% over BF16 (6.9698 vs 6.9404). We did not continue.

## 12. Two draft chains per cycle: measured, not built

A second chain starts from the second-choice first token of the drafter (PLAN.md 4.5 item 3, the smallest tree). It
helps only when the first draft is wrong and the second is right. After that, it continues exactly like the
teacher-forced unroll. Thus `tools/twochain_sim.py` replays speculative cycles along the 75 held-out replies with the
fine-tuned head, and counts the tokens per cycle. Its one-chain numbers reproduce those of the engine
(`tools/eval_drafter.py`: 2.76 / 3.39).

| | k=3, one chain | two chains | gain | k=7, one chain | two chains | gain |
|---|---|---|---|---|---|---|
| All | 2.76 | 2.96 | +7.2% | 3.41 | 3.71 | +8.8% |
| Prose | 2.60 | 2.81 | +8.2% | 3.05 | 3.35 | +9.8% |
| Q&A | 2.60 | 2.82 | +8.4% | 3.09 | 3.38 | +9.4% |
| Code | 3.15 | 3.31 | +5.0% | 4.43 | 4.74 | +7.0% |
| Structured | 2.82 | 3.00 | +6.4% | 3.58 | 3.83 | +6.9% |

The second chain doubles the verify rows. A k=3 cycle with 4 -> 8 rows costs +5.5% (91.8 -> 96.9 ms). A k=7 cycle with
8 -> 16 rows costs +9% (103.6 -> 112.6 ms). The GDN rows and the drafts of the second chain add more. The net result is
about +1.5% at k=3 and nothing at k=7. When two or three requests share a cycle, it gives nothing. We did not build it.

## 13. A two-layer drafter: no gain

`tools/train_drafter.py --extra-layers 1` adds a second decoder layer on top of the MTP head. It is a copy of the trained
layer with its output projections (o_proj, down_proj) at zero. Thus the training starts exactly at run 4 (we checked:
identical unroll outputs). It adds 372M parameters. We trained it with lr 5e-5 for the new layer, for 2 epochs on the
same 3,827 replies.

| Held-out agreement | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---|---|---|---|---|---|
| Run 4 (start) | 0.812 | 0.739 | 0.700 | 0.680 | 0.667 |
| After epoch 0 | 0.804 | 0.729 | 0.691 | 0.667 | 0.653 |
| After epoch 1 | 0.811 | 0.737 | 0.701 | 0.681 | 0.667 |

Section 7 (multi-layer features), section 12 (two chains) and this section show the same thing. The limit is not the
inputs of the drafter, its depth or the verify structure. On 1.4M tokens of the replies of the target, this head
has learned what the data holds. The remaining way to improve the drafter is much more data (5-10x: about a day of
generation and training on this machine).

## 14. SwiGLU + NVFP4 quantization in the epilogue of the up GEMM

The prefill MLP ran the stacked gate | up GEMM to bf16 [M, 2 × 17,408]. Then `k_silu_mul_quant` read that back to write
the NVFP4 input of the down GEMM. This was ~300 MB of traffic per layer at 2,048 tokens around the GEMMs, 44.5 ms per
chunk (section 10).

Now the gate GEMM writes bf16 g. The epilogue of the up GEMM computes silu(g) · α · acc and quantizes it to NVFP4, with
e4m3 scales per 16, in the layout that the down GEMM reads (`csrc/gemm_nvfp4.cu`, `SwigluNvfp4`). This is a CUTLASS
epilogue visitor tree on the block-scale-factor store of SM120. It scales by amax · nc / 6, with nc = 1 / the input
scale of the down projection, as `quant_nvfp4` does. The traffic per layer drops to ~160 MB.

- The output is the same as the unfused path, up to rounding ties (99% of the e2m1 codes are equal). The fused path no
  longer rounds u and the product to bf16. The error against fp32 is 0.131 / 0.158 both ways (`tests/test_gemm_nvfp4.py`).
- Per layer at 2,048 tokens: 2.81 ms (GEMM 2.10 + silu_mul_quant 0.71) -> 2.34 ms.
- Prefill: 2k 0.619 -> 0.596 s (3,434 tok/s), 8k 2.598 -> 2.503 s (3,273 tok/s), 3.7% faster.
- Perplexity (prefill engine): WikiText 7.0707, code 1.8026 (7.0785 / 1.8027 before). `COLINFER_FUSED_SWIGLU=0` selects
  the old path.

## 15. Gated DeltaNet chunked prefill in CUDA

The `chunk_gated_delta_rule` of FLA was 92 ms of a 2,048-token chunk. Its 48 layers ran cumsum, KK^T + triangular
solve, the w / u recompute, the state recurrence (which wrote the state of each chunk) and the output pass. FLA pinned
the recurrence to 2 warps on Blackwell. `csrc/gdn_prefill.cu` does the same math in two kernels, per value head, with
64-token chunks and an in-chunk cumulative decay G (see the file header):

- `k_wy` runs all chunks in parallel. It computes A = strictly-lower β_i (k_i·k_j) e^(G_i − G_j) from a tensor-core
  K K^T. Then it computes T = (I + A)^-1 by 16 × 16 blocks (the diagonal blocks by substitution, then three block rows
  of small products).
- `k_chunk` runs one block per (value head, 64-wide V slice). It goes through the chunks in sequence, with the state on
  chip (fp32 registers, a bf16 copy in shared memory). It computes U = T diag(β) V, W = T diag(β e^G) K and
  V_new = U − W S. Then it computes P = (Q K^T) ∘ decay mask and O = scale (e^G ∘ Q S + P V_new). Last, it updates
  S ← e^(G_C) S + K^T (e^(G_C − G) ∘ V_new). All of this runs on mma.sync bf16. The K / V of the next chunk stream in with cp.async.

Against FLA (`tests/test_gdn_prefill.py`): the output and the final state are within 0.2-0.35% relative (bf16 rounding).

Per layer at 2,048 tokens: FLA 1.93 ms, CUDA **0.82 ms** (k_chunk 0.61, k_wy 0.19). The steps to get there:

- k_wy first solved column by column (64 threads, local-memory chains: 0.575 ms). The blocked inverse, and T in the
  shared memory of K (two blocks per SM), brought it to 0.19 ms.
- k_chunk first loaded the T and Q rows and the strided g / β with synchronous global loads at each chunk (0.76 ms). Now
  k_wy also writes the G and β of each chunk contiguously. T, Q and V stream into single shared buffers. The kernel fills
  each buffer again as soon as the chunk no longer needs it (T after V_new, V and Q after the state update). This gives
  0.61 ms in 99.3 KB of shared memory.
- The state update reads V_new decayed to the chunk end, rounded to bf16 once from fp32, as FLA does. (A scale of the
  stored bf16 V_new instead, a second rounding, moved the WikiText perplexity by +0.27%.)

| Prompt | FLA | CUDA |
|---|---|---|
| 2k | 0.593 s | **0.536 s** (3,818 tok/s) |
| 8k | 2.485 s | **2.290 s** (3,577 tok/s) |
| 32k | 11.64 s | **10.87 s** (3,014 tok/s) |
| 126k (passkey TTFT) | 76.6 s | **66-67 s** |

Perplexity (prefill engine, FP8 KV): WikiText 7.0807, code 1.8010, code ctx 8192 1.6073, against 7.0707 / 1.8026 /
1.6107 with FLA. The differences have both signs, and they are the size that other rounding-only changes produce.
Passkey 6/6 at 31k and 126k. `tests/scheduler_check.py` passes. `COLINFER_GDN_CUDA=0` selects FLA.

## 16. DFlash2 (block-diffusion drafter) vs the MTP head

Public reports put SGLang + DFlash2 at 40-50 tok/s on one DGX Spark (MT-Bench acceptance 4.10 per verify, HumanEval
4.39). `z-lab/Qwen3.8-27B-DFlash2` has 1.9B parameters:

- 5 non-causal layers over the residual stream of the target after layers 5 / 19 / 33 / 47 / 61
- an 8-token block per forward
- a top-16 candidate selector

 `tools/dflash_sim.py` ports its
forward (from `srt/models/dflash.py` in SGLang) and replays greedy cycles with the hidden states of our target.

**Acceptance (tokens per verify step) on the greedy replies of the engine to the 40 eval prompts** (half with thinking
on):

| | MTP k=7, fine-tuned head (ours) | DFlash2, our port | DFlash2 in SGLang 0.5.21 | best of MTP and DFlash2 per cycle |
|---|---|---|---|---|
| Code | 5.64 | 5.27 | 4.85 | 5.33 |
| Prose | 2.79 | 2.83 | 2.91 | 2.95 |
| Q&A | 3.35 | 3.29 | 3.29 | 3.42 |
| Structured | 3.56 | 3.54 | 3.30 | 3.68 |
| All | 3.45 | 3.42 | 3.37 | 3.54 |

- The port agrees with the own DFlash2 of SGLang (3.42 vs 3.37 overall), so the comparison is fair. On these prompts,
  DFlash2 drafts no better than the fine-tuned MTP head. The published 4.1-4.4 come from easier benchmarks.
- Two chains, one from each drafter, would add < 5% tokens per cycle for twice the verify rows. This is not worth it.

**End to end, the same 40 requests** (greedy, 256 tokens, token-id prompts, wall time with prefill):

| tok/s | SGLang + DFlash2 | colinfer (defaults) |
|---|---|---|
| Code | 41.2 | **50.7** |
| Prose | 25.2 | **30.0** |
| Q&A | 28.3 | **33.7** |
| Structured | 28.4 | **34.5** |
| All | 29.0 | **34.8** |

On identical requests, the engine is ~20% faster than SGLang + DFlash2. Prose stays near 30 tok/s with either drafter.
Both drafters learn ~0.45 acceptance per drafted token on the prose of this model.

## 17. Where the time of a speculative cycle goes, and an NVFP4 drafter

A trace of a k=7 cycle at 8k on the INT6 / INT5 weights (`bench/traces/cycle_k7_8k_int`, 98.6 ms):

| Part | Time |
|---|---|
| GDN commit of the accepted rows + MTP drafting (catch-up row + 6 chained steps) | 22.9 ms (20.3 ms of weight GEMMs) |
| Target verify, 8 rows: 64 layers + lm_head | 75.7 ms (≈15.4 GB: ≈204 GB/s against ≈238 peak) |

Each draft step streams the MTP head again (FP8 projections + MLP: 423 MB), and the draft lm_head (NVFP4, 69,632 rows:
≈200 MB). `Bf16Linear.to_lowbit` (`engine/spec/mtp.py`) now gives the head INT6 / INT5 / NVFP4 decode copies
(round-to-nearest from BF16, block-16 e4m3 scales) on the existing skinny GEMM paths. `COLINFER_MTP_FORMAT` selects
one.

| Drafter weights | k=3 cycle | k=7 cycle | Tokens per cycle k=3 / k=7 (`tools/eval_drafter.py`) |
|---|---|---|---|
| FP8 (before) | 91.6 ms | 103.8 ms | 2.78 / 3.45 |
| INT6 | 90.6 ms | 101.5 ms | 2.77 / 3.44 |
| INT5 | | | 2.77 / 3.43 |
| **NVFP4 (default)** | **89.1 ms** | **98.4 ms** | **2.77 / 3.41** |
| NVFP4, draft vocabulary 32k | 88.1 ms | 95.8 ms | 2.70 / 3.30 |
| NVFP4, draft vocabulary 16k | 87.8 ms | 94.3 ms | 2.60 / 3.12 |

The cycle times are with the FP8 target projections. The draft vocabulary stays 64k: a smaller one loses as much
acceptance as it saves time. Drafts change the speed, never the outputs: `tests/spec_check.py` and
`tests/scheduler_check.py` pass.

We ran the 40-request comparison of section 16 again (server defaults, tok/s):

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| FP8 drafter | 50.7 | 30.0 | 33.7 | 34.5 | 34.8 |
| **NVFP4 drafter** | **53.9** | **31.3** | **36.3** | **36.5** | **36.8** |
| SGLang + DFlash2 | 41.2 | 25.2 | 28.3 | 28.4 | 29.0 |

The gap between a k=7 cycle (≈93 ms now) and its byte floor (≈70 ms at 238 GB/s) has three parts:

- The verify pass streams at ≈86% of peak. The long-K down projection and the small output projections are the weaker
  shapes.
- The GDN recurrence, attention and norms are ≈4-5 ms of serial work per cycle.
- The drafter still streams ≈0.44 GB per step.

## 18. Decode cycle: the GDN recurrence, commit overlap, draft early exit

Where the time of a k=7 cycle goes now (`bench/traces/cycle_k7_8k_v2`, INT6 / INT5 weights, 8k). We charge each kernel
with the time that it alone adds to the timeline:

| Part | Time |
|---|---|
| Verify weight GEMMs (15.5 GB) | 69.5 ms (≈223 GB/s) |
| Verify GDN recurrence / attention / norms / small GEMVs | 1.45 / 1.36 / 0.34 / 0.44 ms (+0.5 ms elementwise) |
| Drafting (catch-up row + 6 chained steps) and what the GDN commit adds | 16.0 ms (14.8 ms of GEMMs) |

The weight GEMMs stream within 2-5% of the rate of a pure-streaming kernel with the same access pattern (229-236 GB/s,
with the NVFP4 scale planes). `cp.async.bulk` into a shared-memory ring gives +1-3%, and contiguous 4 KB tiles give
236-241. In isolation, the sum of the GEMM times is within 1.7 ms of their time inside the cycle. Thus a TMA rewrite of
the skinny GEMM would save ≈2 ms per cycle. It would save more only with a tile-contiguous copy of the NVFP4 MLP weights
(+9.6 GB).

- **GDN verify / commit kernel** (`csrc/gdn_step.cu` `k_delta_multi`). The kernel computes the q / k norms, β and the
  decay of all T tokens first (warp t: token t, the same sums in the same order). It computes the gated RMSNorm after
  the loop, from per-step column partials. Thus a step is the two passes over the state and one `__syncthreads` (two
  alternating column-partial buffers).
  - With 8 tokens: verify 30.7 → 22.5 µs per layer, commit 42.7 → 36.4 µs.
  - Both are now near the bytes of the fp32 state: 17.5 µs fixed for verify (3 MB read, 13 µs at peak) + 0.5 µs per
    token.
  - A column-split variant (4-block clusters, 192 blocks) was no faster.
- **Commit in parallel with drafting.** The cycle graph runs the GDN commit on a second branch, because the draft steps
  never read the GDN state (`COLINFER_COMMIT_OVERLAP`). k=7 cycle 91.0 → 90.4 ms.
- **GDN glue in verify.** The conv / delta-rule kernels take row strides. Thus `mixed`, `z`, `b` and `a` stay views of
  the projection outputs, and four copy kernels per layer are gone. The b / a GEMV runs on a parallel branch next to the
  qkv / z weight stream, as plain decode already did. The serial work per GDN layer went from ≈15 to ≈3 µs. k=7 cycle
  90.4 → 89.4 ms, k=3 81.3 → 81.1 ms.
- **Draft early exit** (`COLINFER_DRAFT_STOP`, default 0.1).
  - After each draft step, the cycle keeps the product of the drafter probabilities of its drafts so far (softmax over
    the draft vocabulary).
  - When this product is below the threshold for all active slots, a device flag makes the skinny GEMMs of the remaining
    steps return at once with a zero output. (`ops().skinny_skip`: the launches that the graph captures while the flag
    is set contain the flag pointer.)
  - The verify rejects the junk drafts, so the outputs do not change (`tests/spec_check.py --drafter mtp --k 7`,
    `tests/scheduler_check.py`).
  - A skipped step costs ≈0.35 ms instead of ≈1.8 ms (a k=7 cycle with all steps skipped: 79.8 ms, full: 90.8 ms).
  - Simulated on 3,080 recorded k=7 cycles: 35.3 → 36.6 tok/s at 0.2. With the probability of a single step as the
    criterion: 36.1. An oracle that stops right after the first rejected draft: 38.8.
- **Tried and dropped:** an L2 prefetch (`cp.async.bulk.prefetch.L2`, on a parallel branch) of the first 4-12 MB of the
  GDN output projection, during the serial recurrence of the layer. In isolation, 8 MB decreased that GEMM by 27 µs. In
  the cycle, the k=7 time increased by 0.5-1 ms, because the prefetch slows the state reads of the delta kernel, which
  are on the critical path.

The 40 requests of section 16 (server defaults, tok/s):

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| Section 17 | 53.9 | 31.3 | 36.3 | 36.5 | 36.8 |
| GDN kernel + commit overlap | 54.5 | 31.7 | 36.6 | 36.8 | 37.2 |
| + draft early exit at 0.1 (**default**) | 54.8 | 32.6 | 36.8 | 38.6 | **38.1** |
| early exit at 0.15 / 0.2 / 0.3 | | | | | 38.1 / 37.9 / 37.1 |
| k=7 every cycle (no k=3), early exit 0 / 0.1 | | | | | 37.3 / 38.3 |

The run-to-run noise is about ±0.4 tok/s. At width 1, a draft of 7 in each cycle with the early exit is as good as the
adaptive k=3 / 7 choice. The adaptive choice stays, because it also covers three slots, where 3 × 8 rows would need two
weight passes. `COLINFER_K_OPTIONS` overrides the draft lengths that a cycle picks from.

## 19. Low-rank draft head, and what a TMA weight stream would give

**Draft head.** Each draft step streamed the draft lm head: 64k static + 4k prompt rows of the NVFP4 lm_head of the
target, ≈200 MB, ≈0.86 ms. This was about half of the step. The weights of the head are not low-rank. Rank 1024 holds
42% of their energy, and the argmax of the full head is in the top 64 of a rank-1024 SVD approximation only 75% of the
time. But the drafter outputs g are low-rank: their top 1024 principal directions U hold 89% of the energy.

Thus the engine scores (g U)(W U)^T instead of g W^T. This takes two NVFP4 skinny GEMMs (U^T 1024 x 5120 and
W U 69,632 x 1024, ≈43 MB). Then it rescores the top candidates of the approximation exactly against the real NVFP4 rows
(`ops().rescore_nvfp4`, `csrc/sampling.cu`). This gives the argmax of the full head almost always:

| Rank | Full head's argmax in the top 16 / 64 / 256 (held-out drafter outputs, fine-tuned head) |
|---|---|
| 512 | 0.899 / 0.971 / 0.992 |
| **1024** | 0.968 / 0.994 / **0.9996** |
| 2048 | 0.995 / 0.9999 / 1.000 |

- `tools/lowrank_draft_head.py` collects the drafter outputs from real k=7 cycles (40 prompts of the
  `tools/drafter_data.py` mix, with a seed that the evals do not use). It fits U and saves it to
  `~/.cache/colinfer/drafter/draft_head_pca.safetensors`. The engine uses the file when it is present.
  `COLINFER_DRAFT_LOWRANK` (`auto` | `0` | a path) and `COLINFER_DRAFT_CANDS` (default 256) control it.
- The engine keeps W U for the whole target vocabulary (143 MB). Thus the prompt rows of a request are a gather, as for
  the draft head itself. The server startup takes 1 s more.
- The probability for the early exit uses the exact logit of the chosen draft over the logsumexp of the approximation.
- Acceptance (`tools/eval_drafter.py`, tokens per cycle k=3 / k=7): full head 2.77 / 3.41. Low-rank with 64 candidates
  2.76 / 3.39, **with 256: 2.77 / 3.41**.
- Cycle (`bench/decode_bench.py`, 8k): k=7 89.8 → **86.0 ms**, k=3 81.4 → 79.1 ms (width 2: 97.3 → 93.7, 85.6 → 83.9).

The 40 requests of section 16 (server defaults, tok/s, `bench/request_mix_bench.py`):

| | Code | Prose | Q&A | Structured | All |
|---|---|---|---|---|---|
| Section 17 (before this round) | 53.9 | 31.3 | 36.3 | 36.5 | 36.8 |
| Section 18 (GDN kernel, commit overlap, early exit) | 54.8 | 32.6 | 36.8 | 38.6 | 38.1 |
| **Low-rank draft head (default)** | **57.6** | **33.3** | **38.4** | **39.6** | **39.3** |
| (early exit at 0.15 instead of 0.1) | 57.5 | 33.5 | 38.5 | 39.8 | 39.5 |
| SGLang + DFlash2 (section 16) | 41.2 | 25.2 | 28.3 | 28.4 | 29.0 |

**TMA weight streaming, measured.**

- A `cp.async.bulk` variant of the skinny GEMM streamed at half the rate of the register version (65-130 GB/s). It used
  a 2-3 stage shared-memory ring per warp, lane r copied the 256 bytes of weight row r and its 32 scale bytes, and each
  stage had one mbarrier. Its outputs were bit-identical.
- The request count limits the copy engine. 256-byte row pieces stop at ≈192 GB/s even without the scales (≈115 with
  them).
- 2D tensor-map loads (two 128-byte x 16-row boxes per chunk, 128B swizzle, plus a 32 x 16 scale box) match the register
  path without scales (222-235 GB/s). With the scales they lose (212-224 vs 229-236).
- Only a tile-contiguous layout streams faster through bulk copies: a 16-row x 512-k chunk with its scales as one 4.6 KB
  block reaches 232-241 GB/s. This is ≤3% of the weight GEMMs, ≈2 ms per cycle.
- For the MLP (60% of the bytes), this layout needs a second copy of the weights (+9.6 GB), because the CUTLASS prefill
  reads the checkpoint layout. We did not continue.

## Next

Where the 85 ms of a k=7 cycle go now (`bench/traces/cycle_k7_8k_v3`):

- verify 73.5 ms: 69.5 ms of weight GEMMs at ≈223 GB/s, and ≈3 ms of GDN recurrence / attention / norms
- drafting 11.6 ms: ≈1.66 ms a step, ≈280 MB of weights a step

The remaining work:

- The own weights of the drafter (MTP layer, NVFP4, ≈240 MB a step, of which its MLP is 150 MB). Fewer bits or a smaller
  MLP cost only acceptance, never correctness. But they need a new training (`tools/train_drafter.py`) to keep the
  acceptance.
- A better early-exit rule. The probability product reaches 36.6 of the 38.8 tok/s of an oracle in the k=7 simulation
  (section 18). A small learned predictor over a few drafter features may close part of that gap.
- A better drafter for prose. There, the acceptance is about 0.45-0.50 per token. Sections 7, 12 and 13 show that
  features, two chains and a second layer do not help. More data is what is left.
- A tile-contiguous copy of the decode weights for bulk-copy streaming: ≤3% of the weight GEMMs (section 19).
- The fused draft-selection reductions (topk, logsumexp: ≈0.35 ms a cycle).
- A scheme that keeps the precision of Q in layers 23-51 (their outputs move 2-4% with e4m3 Q). This would remove most
  of the +0.25-0.3% perplexity of the FP8 prefill attention.

# Architecture

This page describes how colin-inference-engine runs `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark. It describes the
engine as it is now. The dated logs in `docs/history/` (`phase*`, `baseline.md`, `results.md`) record how the engine got
here. They give the measurements behind each decision and the alternatives that the project tried and removed.

## 1. The machine and the model

**GB10 (DGX Spark):**

- 48 SMs of sm_121 (the Blackwell consumer ISA: `mma.sync` tensor cores, no WGMMA / tcgen05)
- 99 KB of shared memory per block and 24 MB of L2
- 128 GB of unified LPDDR5x memory, which streams at about **238 GB/s** in practice

Decode reads each weight once per step, so the decode speed is the weight bytes divided by the bandwidth. Prefill is
compute-bound on the tensor cores.

**Qwen3.8-27B:**

- 64 layers: 48 Gated DeltaNet linear-attention layers and 16 gated full-attention layers, interleaved 3:1
- hidden size 5120, a SwiGLU MLP with intermediate size 17408, and a vocabulary of 248,320
- an untied lm_head and one MTP block (one extra decoder layer for multi-token prediction)

The layers:

- *Attention:* 24 query heads over 4 KV heads (GQA), with head dim 256. RoPE applies to the first 64 dims only
  (θ = 10⁷). q and k have an RMSNorm. An output gate multiplies the output by sigmoid(gate): `q_proj` gives q and one
  gate per head.
- *Gated DeltaNet (GDN):* 16 key heads and 48 value heads of 128 dims. A depthwise causal conv (kernel 4) runs over the
  q / k / v channels, and the layer L2-normalizes q and k. Each value head has a recurrent fp32 state S [128 × 128]. The
  gated delta rule updates S at each token: `S ← exp(g) S;  S += k ((v − Sᵀk) β)ᵀ;  o = Sᵀq`. A gated RMSNorm with
  silu(z) follows.
- *Norms:* zero-centered RMSNorm, y = x / rms(x) · (1 + w).

`engine/model/qwen35.py` implements all of this in plain PyTorch, op for op like `modeling_qwen3_5.py` in transformers.
The server does not use it. It is the reference for the tests of each kernel.

## 2. Weights

| Tensor | Stored as (checkpoint) | Decode streams | Prefill reads |
|---|---|---|---|
| MLP gate / up / down (64 layers, 9.6 GB) | NVFP4 (e2m1 + e4m3 scale per 16 + fp32 global), 4.5 bits | the same NVFP4 | the same, CUTLASS-swizzled scales |
| lm_head (0.7 GB) | NVFP4 | NVFP4 | NVFP4 (only for all-token logits) |
| attention q / k / v / o (16 layers) | FP8 e4m3, per-tensor scale | **INT6 copy** (1.36 GB), else the FP8 | FP8 (W8A8) |
| GDN in_proj_qkv / in_proj_z / out_proj (48 layers) | FP8 | **INT5 copy** (3.8 GB), else the FP8 | FP8 (W8A8) |
| GDN in_proj_b / in_proj_a, conv, norms, embeddings | BF16 | BF16 | BF16 |
| MTP block | BF16 | **NVFP4 copy** (round-to-nearest) | BF16 (cuBLAS) |

`tools/int6_requant.py` makes the INT6 / INT5 copies from the BF16 checkpoint `Qwen/Qwen3.8-27B`. Their weight error is
lower than that of the checkpoint's own FP8 (2.24% vs 2.67% relative), at 6.5 / 5.5 bits per weight. Each weight is a
signed integer q with an e4m3 scale per 16 weights and an fp32 global scale. The file keeps the low bits in an
NVFP4-layout nibble plane and the high bits in a second plane (`engine/weights/quantize.py`). Against the FP8 weights,
the copies move WikiText perplexity by −0.21% and Python-code perplexity by +0.23%, and decode is ~9.5% faster.

The engine stacks projections that share an input (q / k / v, and in_proj_qkv / in_proj_z) into one matrix, so they cost
one weight pass (`StackedFp8Linear`). In the copies, the stacked projections share one global scale.

The GPU holds 20.2 GB of checkpoint weights and 5.2 GB of INT copies. A decode step streams ≈15.5 GB.

## 3. Decode

`engine/model/fast.py` turns the reference module tree into the decode model (`load_fast_model`, `to_fast`):

- Quantized linears become kernel modules.
- Attention, GDN and norms become kernel layers.
- Positions move to the device (`FastState.pos_t`).
- The KV cache is fp8 (e4m3, unit scale, saturating: 32 KB per token).

`DecodeGraph` captures one decode step (embedding → 64 layers → lm_head → sampler) as a CUDA graph. The host only writes
the input tokens and replays the graph.

The table shows the kernels of each layer, in order. The engine launches each kernel with programmatic dependent
launch. Thus the next GEMM starts to stream its weights while the previous kernel finishes.

| Full-attention layer | GDN layer |
|---|---|
| RMSNorm (`norm.cu`) | RMSNorm |
| q / k / v projection, one INT6 weight pass (`skinny.cu`) | q / k / v / z projection, one INT5 pass. The b / a gates (bf16 GEMV, `gemv.cu`) run on a parallel branch. |
| `attn_prologue`: q / k RMSNorm, partial RoPE, k / v written to the fp8 cache at `pos_t` | `gdn_conv`: causal conv + SiLU over the conv window. `gdn_conv_commit` advances the window. |
| `attn_decode`: multi-row tensor-core attention, output gate fused (`attn_decode.cu`) | `gdn_delta`: L2 norms, gated delta rule on the fp32 state, gated RMSNorm (`gdn_step.cu`) |
| o_proj + residual | out_proj + residual |
| RMSNorm, SwiGLU (gate and up in one NVFP4 pass), down + residual | the same MLP |

**The skinny GEMM** (`csrc/skinny.cu`) is the central kernel of the engine. All decode linears run on it, in NVFP4,
INT6 / INT5 and FP8. A warp owns 16 weight rows and streams them in 256-byte runs per row, one chunk ahead in registers.
A small shared-memory scratch transposes each chunk into `mma.sync.m16n8k16` fragment order. The kernel then dequantizes
the chunk exactly into bf16. (e2m1 × e4m3 fits in bf16, and a 0x4300 | c bit trick builds the INT codes.)

The activations are the other operand of the mma, so 16 rows cost about the same as 1 row. Thus only the weight bytes
set the speed of the kernel. Split-K by shape fills the 48 SMs. In isolation, the kernel streams at 222-236 GB/s. This
is within 2-5% of a pure-streaming kernel with the same access pattern.

**Attention** (`attn_decode.cu`) serves all query rows of a slot from one pass over the KV of that slot. A slot has
24 heads × T new tokens, up to 48 rows. K stays raw fp8 in shared memory, and `mma.sync` f16 tiles compute QKᵀ and PV. Each KV head
has 12 key-tile stripes, which fill the SMs. A combine kernel folds the stripes and applies the output gate. A verify of
8 rows reads the cache once, at the cost of plain decode.

**GDN** (`gdn_step.cu`) keeps the 64 KB fp32 state of each value head in registers across all T tokens. One block of 512
threads serves each head and slot. The kernel computes all values that do not depend on the state (norms, β, decay)
before the sequential chain, and the gated RMSNorm after it. Thus a token step is two register passes and one
`__syncthreads`. The same kernels do plain decode (T = 1, state advanced), speculative verify (T rows, state not
changed) and commit (state advanced by the accepted count). At T = 8, a layer costs ≈17.5 µs to read the 3 MB of state,
plus ≈0.5 µs per token.

At 8k context, plain decode takes **74 ms per token** (13.5 tok/s). This is the ≈15.5 GB stream at ≈210 GB/s.

## 4. Speculative decoding

The code is in `engine/spec/mtp.py`. The decode of each request runs as speculative cycles. One CUDA graph covers a full
cycle (`MtpCycle`):

1. **Verify** the k + 1 rows [y, d₁ … d_k] (the last token and k drafts) on the target. This gives logits and post-norm
   hidden states for each row, and writes KV for all rows. It does not change the GDN state (`FastQwen35.verify`).
2. **Accept** the leading drafts that are equal to what the target itself gives at each row. For greedy requests this
   is the argmax, and for sampled requests it is the sample drawn with a uniform keyed by (seed, position)
   (`engine/spec/accept.py`). The target's token replaces the first draft that does not match. If all drafts match, the
   last row gives a bonus token. Thus a cycle emits 1 … k + 1 tokens, and **the output is exactly the output of plain
   decoding**, whatever the drafts were. The cycle cuts the accepted length after a stop token.
3. **Commit** the accepted prefix: advance the GDN conv windows and states, and move the positions. Later cycles
   overwrite the KV past the accepted length. The commit runs on a parallel graph branch, because the draft steps never
   read the GDN state.
4. **Draft** the k tokens of the next cycle with the MTP block. The block runs one catch-up row for each newly committed
   position, with the true hidden states of the target. Then it runs k − 1 chained steps, which feed the block its own
   output. Its KV cache is fp8 as well.

The engine streams the drafter weights again at each draft step, so the drafter must be cheap:

- Its linears run on NVFP4 copies on the skinny GEMM.
- The draft head scores a static vocabulary of the 64k most frequent tokens (`draft_vocab.npy`) plus up to 4k tokens of
  the current prompts. It does not score all 248k rows.
- With `draft_head_pca.safetensors`, the draft head is a **low-rank approximation** (g U)(W U)ᵀ. U holds the top 1024
  principal directions of the drafter outputs (≈43 MB instead of ≈200 MB). The engine rescores the top 256 candidates
  of this approximation exactly against the real NVFP4 rows (`rescore_nvfp4`). The argmax of the full head is among
  them for 99.96% of drafts, and the measured acceptance does not change.
- **Early exit**: the product of the drafter probabilities can drop below 0.1 for all active slots. Then the remaining
  draft steps skip their GEMMs. A device flag that the skinny GEMM reads (`skinny_skip`) makes them return at
  once: ≈0.35 ms instead of ≈1.8 ms per step. The next verify rejects the junk drafts of these steps.

**Draft length.** The scheduler picks k = 3 or 7 for each cycle, for the most expected tokens per second. With the
running per-token acceptance a of a slot, a cycle gives (1 − a^{k+1}) / (1 − a) expected tokens. The scheduler divides
this by the cycle time measured at startup, plus the KV reads that grow with the context. The verify rows must fit one
skinny pass (width × (k + 1) ≤ 16), so three decoding slots use k = 3.

**Suffix-match drafts** (`engine/spec/suffix.py`, `--suffix-drafts`, default 8). The history of a request is its prompt
and the reply so far. When the last 8 or more tokens of the history occurred earlier in it, the next cycle uses other
drafts. It verifies the tokens that followed the earlier occurrence instead of the MTP drafts.

A request that decodes alone can verify 15 such drafts, because its 16 rows still fit one weight pass. The cycle then
takes 85 ms instead of 80 ms. The drafter state depends only on accepted tokens, so either source can feed any cycle. Code-edit
replies that repeat their input decode ~65% faster.

At 8k context, a k = 7 cycle takes **86 ms**. The verify takes 73.5 ms (69.5 ms of weight GEMMs), and the drafts take
≈11.6 ms. With the fine-tuned drafter, a k = 7 cycle accepts 3.41 tokens on average (code 5.55, prose 2.77).

## 5. Prefill

`engine/model/prefill.py` runs prefill for one slot at a time, in chunks of 2,048 tokens. Larger chunks make the FP8
GEMMs slower per token. Each layer runs over the full chunk on the tensor cores:

- **MLP (W4A4):** The fused add + RMSNorm kernel also quantizes the activations to NVFP4, with the static input scale of
  the checkpoint. Then CUTLASS SM120 block-scaled GEMMs (`gemm_nvfp4.cu`) run in three steps. The gate GEMM runs
  first. The up GEMM follows, with silu(gate) · up and its NVFP4 quantization fused into the epilogue. The down GEMM
  runs last, with the residual in the epilogue.
- **FP8 projections (W8A8):** The same fused norm kernel quantizes the activations to e4m3 (static input scale). Then
  cuBLASLt runs the GEMM through `torch._scaled_mm`, with row-wise weight scales.
- **Attention:** The decode prologue runs on all T rows and writes the fp8 KV. Then FlashInfer FA2 runs over a bf16 copy
  of the cached prefix. Past 16k tokens of context, `attn_prefill.cu` runs instead: Q Kᵀ on FP8 tensor cores over the
  cache as stored. This makes prefill 6% faster at 64k and 11% faster at 128k. It adds 0.25-0.3% to the perplexity of
  those chunks, because it rounds Q to e4m3.
- **GDN:** One kernel runs the causal conv, SiLU and the q / k L2 norm, and continues the conv window. Then the chunked
  gated delta rule runs (`gdn_prefill.cu`, 2.35× FLA). A WY kernel per 64-token chunk builds the triangular inverse. A
  chunk kernel per value head and 64-wide V slice keeps the state in registers. A gated RMSNorm and the FP8 out_proj
  follow.

After prefill, the state is exactly what decode expects. With speculation, `Mtp.prefill` writes the KV rows of the
drafter for each chunk right after the chunk. Prefill times: 2k 0.54 s (3.8k tok/s), 8k 2.29 s, 32k 10.9 s, 126k ≈66 s.

## 6. Serving

The code is in `engine/runtime/scheduler.py` and `engine/server/api.py`.

- **Slots.** One batched state holds `--slots` (3) slots of `--max-seq-len` (262,144) tokens with contiguous KV. At
  startup, the engine captures CUDA graphs for each contiguous slot range × {greedy, sampled} × draft length. A step
  replays the smallest range that covers the decoding slots and masks off the other slots (`state.active`). Each kernel
  computes the rows of a slot the same way at any width, so batching never changes an output.
- **Engine loop.** Each step admits queued requests and prefills at most one chunk of one prompt. Then it runs one
  decode cycle for the decoding slots. Thus a long prompt slows the other requests for one chunk at a time, but never
  stops them.
- **Prefix checkpoints.** The engine keeps a ring of GDN-state snapshots (conv + recurrent, 154 MB each, 32 by default).
  It takes a snapshot at the end of each prompt and each reply. It also takes one at the end of the first message, one
  before the last message, and one every 8,192 prompt tokens. A request restores the longest checkpoint that is a proper prefix of its
  prompt, and prefills the rest. Attention KV can resume at any length, but GDN state can resume only where a snapshot
  exists. If the slot of the checkpoint is busy, the engine copies its KV prefix to a free slot.
- **Server.** One engine thread owns the GPU. Asyncio handlers render the chat template, tokenize and stream. The
  engine thread detokenizes and parses each token as the engine emits it: reasoning, content, Qwen XML tool calls and
  stop strings (`engine/server/chat.py`). The Anthropic Messages API (`/v1/messages`, for Claude Code) uses the same
  path: `engine/server/anthropic.py` converts each request to chat messages and each reply to content blocks. A CUDA
  error fails all requests and stops the process, and systemd restarts it. Thus the server checks every request value
  that reaches the engine before it submits the request, and an error in the output parsing of one request fails only
  that request.
- **Memory.** At startup, the engine uses ≈61 GB. The weights use 20.2 GB and the INT copies 5.2 GB. The KV caches use
  3 × 262k × 32 KB = 25.8 GB. The checkpoint ring uses 4.9 GB, and the drafter KV and copies use ≈2.5 GB.
- **Startup** takes ≈25 s. The engine loads the weights and runs a numerical self-test of each matmul path
  (`engine/selftest.py`), which stops the start on a wrong kernel. Then it captures the graphs, calibrates the cycle
  times and runs a warm-up.

## 7. Numerics and the identity guarantee

Speculation and batching are safe only if they cannot change an output. The design of the engine makes sure of this:

- **Row-invariant kernels.** The bits of a row do not depend on how many other rows share the launch. Thus plain
  decode, verify, the drafter and any batch width agree bit for bit:
  - The skinny GEMM uses a fixed k order in each mma step, and a split-K partition that depends only on the matrix
    shape. Rows beyond M are zeros that never touch real rows.
  - Attention processes keys in fixed tiles, assigned to fixed stripes, and folds them in a fixed order. A tile that is
    fully past the length of a row is an exact no-op.
  - The GDN kernels use the same arithmetic for T = 1 and T = k + 1.
- **Position-keyed sampling.** The engine draws the token at position t by inverse CDF from the processed distribution
  (temperature → min-p → top-k → top-p). It uses one uniform u = Philox(seed, t) (`csrc/sampling.cu`). Thus the tokens of
  a request depend only on (seed, prompt). Speculative acceptance samples the target at each verified row with the
  uniform of that row. So speculation reproduces plain sampling token for token.
- **Rounding points** follow the reference model (bf16 where transformers rounds to bf16). The extensions compile with
  `--fmad=false`, so no compiler contraction changes them.

A perplexity gate against the checkpoint controls the quality (`tests/perplexity.py`, WikiText and Python code). Each
change from the numerics of the checkpoint (the INT copies, FP8 prefill attention) stays within 0.5%.

## 8. Validation

- `pytest tests/` (≈1 min) tests each kernel against the PyTorch reference or a dequantized fp32 product. It also tests
  row invariance and the bit identity of verify and sequential decode. Other tests cover the prefill ops, the sampler
  and acceptance statistics, chat parsing and the unroll of the drafter trainer. `test_scheduler.py` runs the scheduler
  on a stub model whose next token depends on the whole history in its state (checkpoints, KV prefix copies, eviction,
  aborts), and `test_api.py` runs the HTTP endpoints on a fake engine thread (validation, streamed vs whole replies,
  errors).
- `tests/golden.py` runs 21 recorded requests at once (greedy, seeded-sampled, a 20k-token prompt) on the server's engine
  (`engine/runtime/build.py`, which the server, the checks and the benchmarks share). They must reproduce
  their tokens and per-token logprobs bit for bit: with suffix-match drafts, without them (`--suffix 0`) and without
  speculation (`check --plain`).
- `tests/scheduler_check.py` compares the outputs of the scheduler with the uncached single-request outputs. It covers
  single, concurrent, mixed (greedy next to sampled and a long prefill), multi-turn and shared-prefix requests.
- Other end-to-end checks: `tests/perplexity.py`, `tests/passkey.py` (long-context retrieval), `tests/parity_hf.py` (the
  reference vs transformers), `tests/server_check.py` (the HTTP API).
- Benchmarks: `bench/perf.py` (the repeatable baseline: prefill, TTFT, decode, memory), `bench/decode_bench.py`,
  `bench/prefill_bench.py`, `bench/request_mix_bench.py` (end to end), `bench/skinny_bench.py`, `bench/attn_bench.py`,
  `bench/trace_summary.py` (nsys).

## 9. Where the time goes, and what is left

A k = 7 cycle takes 86 ms at 8k context:

- ≈69.5 ms of verify weight GEMMs at ≈223 GB/s. The floor for 15.5 GB at 238 GB/s is 65 ms.
- ≈3 ms of GDN recurrence, attention and norms.
- ≈11.6 ms of drafting (≈280 MB of drafter weights per step).

Most of the remaining gains are in the drafter: its own weights, a better stop rule and better acceptance on prose. A
tile-contiguous weight copy that bulk copies can stream could give a few percent more (docs/history/phase6_progress.md,
sections 18-19).

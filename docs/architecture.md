# Architecture

This page describes how colin-inference-engine runs `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark, and in section 10,
`nvidia/Qwen3.8-Flash-Next-NVFP4`. It describes the engine as it is now. The dated logs in `docs/history/`
(`PLAN.md`, `phase*`, `baseline.md`, `results.md`, `qwen4exp_plan.md`) record how the engine got here. They give the
measurements behind each decision and the alternatives that the project tried and removed.

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

Sections 1 to 9 describe the 27B, the default model of the server. Section 10 describes the MoE family
Qwen3.8-Flash-Next, which the server also serves (`--model nvidia/Qwen3.8-Flash-Next-NVFP4`).

## 2. Weights

| Tensor | Stored as (checkpoint) | Decode streams | Prefill reads |
|---|---|---|---|
| MLP gate / up / down (64 layers, 9.6 GB) | NVFP4 (e2m1 + e4m3 scale per 16 + fp32 global), 4.5 bits | the same NVFP4 | the same, CUTLASS-swizzled scales |
| lm_head (0.7 GB) | NVFP4 | NVFP4 | NVFP4 (only for all-token logits) |
| attention q / k / v / o (16 layers) | FP8 e4m3, per-tensor scale | **INT6 copy** (1.36 GB), else the FP8 | FP8 (W8A8) |
| GDN in_proj_qkv / in_proj_z / out_proj (48 layers) | FP8 | **INT5 copy** (3.8 GB), else the FP8 | FP8 (W8A8) |
| GDN in_proj_b / in_proj_a, conv, norms, embeddings | BF16 | BF16 | BF16 |
| MTP block | BF16 | **NVFP4 copy** (round-to-nearest) | BF16 (cuBLAS) |

`tools/int6_requant.py` makes the INT6 / INT5 copies from the BF16 checkpoint `Qwen/Qwen3.8-27B`, and `tools/prepare.py` puts them in
the prepared directory `~/.cache/colinfer/prepared/<snapshot>/`, which the engine reads at startup (`engine/runtime/prepared.py`). Their weight error is
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

`DecodeGraph` (`engine/runtime/decode.py`) captures one decode step (embedding → 64 layers → lm_head → sampler) as a
CUDA graph. The host only writes the input tokens and replays the graph.

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
8 rows reads the cache once, at the cost of plain decode. The tensor-core core is in `attn_tc.cuh`, and it also runs
GQA 24/2 (12 query heads per KV head, 24 stripes) for the qwen4_exp family (section 10).

**GDN** (`gdn_step.cu`) keeps the 64 KB fp32 state of each value head in registers across all T tokens. One block of 512
threads serves each head and slot. The kernel computes all values that do not depend on the state (norms, β, decay)
before the sequential chain, and the gated RMSNorm after it. Thus a token step is two register passes and one
`__syncthreads`. The same kernels do plain decode (T = 1, state advanced), speculative verify (T rows, state not
changed) and commit (state advanced by the accepted count). At T = 8, a layer costs ≈17.5 µs to read the 3 MB of state,
plus ≈0.5 µs per token. A `gate_act` flag selects the output gate of the gated RMSNorm: silu(z) for the 27B, sigmoid(z)
for the qwen4_exp family (section 10).

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
- With the low-rank basis of the prepared directory (`draft_head_pca.safetensors`), the draft head is a **low-rank approximation** (g U)(W U)ᵀ. U holds the top 1024
  principal directions of the drafter outputs (≈43 MB instead of ≈200 MB). The engine rescores the top 256 candidates
  of this approximation exactly against the real NVFP4 rows (`rescore_nvfp4`). The argmax of the full head is among
  them for 99.96% of drafts, and the measured acceptance does not change.
- **Early exit**: the product of the drafter probabilities can drop below 0.1 for all active slots. Then the remaining
  draft steps skip their GEMMs. A device flag that the skinny GEMM reads (`skinny_skip`) makes them return at
  once: ≈0.35 ms instead of ≈1.8 ms per step. The next verify rejects the junk drafts of these steps.

**Draft length.** The scheduler picks k = 3 or 7 for each cycle, for the most expected tokens per second. With the
running per-token acceptance a of a slot, a cycle gives (1 − a^{k+1}) / (1 − a) expected tokens. The scheduler divides
this by the cycle time measured at startup, plus the KV reads that grow with the context. The verify rows must fit one
skinny pass (k + 1 ≤ 16).

**Suffix-match drafts** (`engine/spec/suffix.py`, repeats of 8 or more tokens). The history of a request is its prompt
and the reply so far. When the last 8 or more tokens of the history occurred earlier in it, the next cycle uses other
drafts. It verifies the tokens that followed the earlier occurrence instead of the MTP drafts.

A request that decodes alone can verify 15 such drafts, because its 16 rows still fit one weight pass. The cycle then
takes 85 ms instead of 80 ms. The drafter state depends only on accepted tokens, so either source can feed any cycle. Code-edit
replies that repeat their input decode ~65% faster. The index of the history (a chain of the positions of each 3-token
key) grows on the host while the GPU runs each prefill chunk, and stays with the slot while its tokens are a prefix of
the slot's history, so the next turn extends it instead of building it again.

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

The code is in `engine/runtime/scheduler.py`, `engine/server/worker.py` (the engine thread) and `engine/server/api.py`.

- **One request at a time.** The engine runs one request at a time; the others wait in a FIFO queue. One state holds
  `--slots` (3) slots of `--max-seq-len` (262,144) tokens with contiguous KV. A slot keeps the history of its last
  conversation, so a side request (a title, a subagent) goes to another slot and does not overwrite the cache of the
  main conversation. At startup, the engine captures CUDA graphs for each slot × {greedy, sampled} × draft length.
- **Engine loop.** Each step admits the next queued request when none runs. Then it runs one prefill chunk or one
  decode cycle of the running request.
- **Prefix checkpoints.** The engine keeps a ring of GDN-state snapshots (conv + recurrent, 154 MB each, 32 by default).
  It takes a snapshot at the end of each prompt and each reply. It also takes one at the end of the first message, one
  before the last message, and one every 8,192 prompt tokens. After a last message of 512 tokens or more, it takes one
  after that message too, for clients that send the reply back changed (for example, without its reasoning). A request restores the longest checkpoint that is a proper prefix of its
  prompt, and prefills the rest. Attention KV can resume at any length, but GDN state can resume only where a snapshot
  exists. If the request shares only that prefix with its slot's history, and the rest of that history is 1,024 tokens
  or more and longer than another slot's, the request runs in that other slot with a copy of the KV prefix, so the
  longer conversation stays cached. Each checkpoint stays a prefix of its slot's history: a request that rewrites a
  slot drops the checkpoints past the part it keeps. Thus admission compares the prompt once with each slot's history
  (vectorized), not with each checkpoint: ~10 ms at 90k tokens instead of ~35 ms.
- **Conversation cache on disk** (`engine/runtime/sessions.py`). When a request ends, a writer thread saves the slot's
  reply-end and message-boundary checkpoints of 2,048 tokens or more to the NVMe. A *checkpoint file* holds the GDN
  state, the drafter's input hidden state and the KV rows of the last partial block. A *block file* holds 1,024 KV rows
  of the target and of the drafter (the drafter's rows shifted by one, since its row i reads token i + 1). The hash
  chain of the tokens before the end of a block names it, so the bytes of a block depend only on its name, and shared
  prefixes are stored once. A request restores from disk when the disk holds a prefix longer than the best memory
  checkpoint: the blocks and the checkpoint go back into a slot with direct reads (~12 GB/s) and pinned copies
  (~59 GB/s). If a slot still holds the KV of that prefix, only the checkpoint file is read. The writer reads the KV
  of the full blocks from the slot itself, so admission waits for a pending save before it rewrites rows that the save
  still has to read, and the GDN state from the checkpoint ring, whose buffer is not reused while the save is
  pending. A save copies its partial block into a fixed buffer when it is queued, so the next turn of the
  conversation, which keeps the history up to the last full block, does not wait for the write. A budget (64 GB)
  evicts the least recently used files.
- **Images** (`engine/model/vision.py`). The server resizes and patches each image as the checkpoint's image processor
  does (`engine/server/images.py`), and gives it a run of placeholder ids past the vocabulary: one per 2 x 2 group of
  16-pixel patches, derived from a hash of the image's bytes and its grid (from its size, without processing it).
  Thus the memory and disk caches tell images apart, and a resent image costs a decode (~30 ms) until the caches
  miss: only then are its patches made (resize, normalize, patch: up to ~0.5 s) and the vision tower run. A prefill
  chunk that holds image tokens embeds them as the image pad token, then replaces their rows with the output of the
  vision tower (transformers' Qwen3_5VisionModel, BF16, 0.92 GB, loaded on the first image;
  run once per image and only when its tokens are prefilled).
  Positions follow multimodal RoPE (`get_rope_index` of transformers): the tokens of an image that starts at s
  rotate by (t, h, w) = (s', s' + row, s' + col), with s' the rotary position of s, and the text after it continues
  at s' + max(rows, cols). The attention prologue takes each prefill row's (t, h, w) (`rope_pos`); frequency i
  rotates by h when i % 3 == 1, by w when i % 3 == 2 and i < 30, else by t. Decode adds a per-slot delta
  (`FastState.rope_delta`) to the cache position, since a reply holds no images. The drafter rotates the same way.
- **Constrained decoding** (`engine/runtime/grammar.py`, llguidance). A JSON schema or forced tool calls (Qwen's XML
  format, with the function's parameter names and values typed by their schema) become a grammar, with free reasoning
  before it when the prompt opens a `<think>` block. Every graph applies a per-slot token bitmask to the logits that
  it selects from (all ones for other requests, so their bits do not change). Before a cycle, the host fills the mask
  of each verify row from the matcher advanced by the drafts before that row. A draft that the grammar refuses
  cannot equal the masked target token of its row, so acceptance stops there, and the cycle emits exactly what plain
  constrained decoding would. The masks of a cycle take ~0.2 ms.
- **Server.** One engine thread owns the GPU. Asyncio handlers render the chat template, tokenize and stream. The
  engine thread detokenizes and parses each token as the engine emits it: reasoning, content, Qwen XML tool calls and
  stop strings (`engine/server/chat.py`). The Anthropic Messages API (`/v1/messages`, for Claude Code) uses the same
  path: `engine/server/anthropic.py` converts each request to chat messages and each reply to content blocks. A CUDA
  error fails all requests and stops the process, and systemd restarts it. Thus the server checks every request value
  that reaches the engine before it submits the request, and an error in the output parsing of one request fails only
  that request.
- **Memory.** At startup, the engine uses ≈88 GB. The weights use 20.2 GB and the INT copies 5.2 GB. The KV caches use
  3 × 524k × 32 KB = 51.5 GB. The checkpoint ring uses 4.9 GB, and the drafter KV and its NVFP4 weights ≈4 GB. The
  CUDA graphs share one memory pool (0.4 GB). The vision tower adds 0.92 GB with the first image. The conversation
  cache uses 0.6 GB of pinned host memory for its file buffers and 0.14 GB of fixed buffers on the GPU.
- **Startup** takes ≈14 s. The engine reads the weights straight from the NVMe into the GPU (`engine/weights/direct.py`:
  O_DIRECT into pinned buffers, ~9 GB/s, nothing left in the page cache), takes the drafter's quantized parts from
  `~/.cache/colinfer/drafter/quant-<key>.safetensors` (quantized once per drafter file) and runs a numerical self-test
  of each matmul path (`engine/selftest.py`), which stops the start on a wrong kernel. Then it captures the graphs
  into one memory pool, calibrates the cycle times and runs a warm-up. The vision tower loads with the first image.

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

`tests/check.sh` runs the checks below in order (ruff, pytest, golden.py with and without speculation,
scheduler_check.py) and stops at the first failure; `--quick` runs ruff and pytest only.

- `pytest tests/` (≈1 min) tests each kernel against the PyTorch reference or a dequantized fp32 product. It also tests
  row invariance and the bit identity of verify and sequential decode. Other tests cover the prefill ops, the sampler
  and acceptance statistics, chat parsing and the unroll of the drafter trainer. `test_scheduler.py` runs the scheduler
  on a stub model whose next token depends on the whole history in its state (checkpoints, eviction, aborts), and
  `test_sessions.py` runs it with the disk cache (eviction, restart, shared blocks, the budget, damaged files, a save
  still pending when its slot is reused). `test_vision.py` checks the image positions against `get_rope_index` of
  transformers, the prologue at (t, h, w) positions against the reference rotary, and the scheduler with images on the
  stub model; `test_model_small.py` checks the reference model with an image's positions and embeddings against
  transformers. `test_grammar.py` checks the grammars on the checkpoint's tokenizer (schemas, forced and typed tool
  calls, the masks of the verify rows) and the scheduler under a mask. `test_api.py` runs the HTTP endpoints on a fake engine thread (validation, streamed vs whole replies,
  errors). `test_model_small.py` compares the reference model with the Qwen3.5 model of transformers at tiny scale.
- `tests/golden.py` runs 21 recorded requests at once (greedy, seeded-sampled, a 20k-token prompt) on the server's engine
  (`engine/runtime/build.py`, which the server, the checks and the benchmarks share). They must reproduce
  their tokens and per-token logprobs bit for bit: with suffix-match drafts, without them (`--suffix 0`) and without
  speculation (`check --plain`).
- `tests/scheduler_check.py` compares the outputs of the scheduler with the uncached single-request outputs. It covers
  single and queued requests (greedy around sampled and a long prefill), multi-turn and shared-prefix requests, a
  restore from disk after a restart (the same output as a restore from memory), an image (the model reads the text
  in it, MTP gives the tokens of plain decode, the next turn reuses the image) and constrained decoding (a schema-valid
  JSON reply identical to plain decode, a forced tool call, JSON after free reasoning).
- Other end-to-end checks: `tests/perplexity.py`, `tests/passkey.py` (long-context retrieval), `tests/server_check.py`
  (the HTTP API).
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

## 10. The qwen4_exp family

The engine also serves the MoE model `Qwen/Qwen3.8-Flash-Next` (`model_type` `qwen4_exp`), from the checkpoint
`nvidia/Qwen3.8-Flash-Next-NVFP4` (132.7 GB): `python -m engine.server --model nvidia/Qwen3.8-Flash-Next-NVFP4`.
`engine/model/family.py` picks the family from the `model_type` of `config.json`. A `Family` gives the load, the state,
the prefill, the defaults, the speculation (`Speculation`) and the memory of its model, and the build, the scheduler
and the server take them from there. The 27B path stays byte-identical: before and after the work, the session files of
`tests/scheduler_check.py` have the same hashes. This section describes the served engine. The numbers are from
2026-10-24, on the real checkpoint. `docs/history/qwen4exp_plan.md` is the reference for the design, the decisions,
the milestones and the dated status log (section 9).

### 10.1 The model

The model has 125B parameters, 6B of them active per token. It has 48 layers of hidden size 2560: 12 groups of three
Gated DeltaNet layers and one attention layer. The vocabulary (248,320) and the chat template are the 27B's. It has five
mechanisms that the 27B does not have:

- *MoE:* each layer has 512 SwiGLU experts (intermediate size 640), top-10, plus one shared expert with a sigmoid gate.
- *Hyper-connections:* the residual stream of a row is 4 × 2560 wide. A gated residual before each sublayer replaces
  the input and post norms: a grouped norm of the 4 streams, a low-rank mix into the input of the sublayer, and 4
  injection weights for its output. There is no final norm.
- *Qwen Sparse Attention (QSA):* GQA 24/2 with head dim 256. An indexer (4 query heads, 1 key head of 128) scores
  4-token blocks. Each query attends to its top 512 blocks plus the tail, the tokens of its own incomplete block. Thus
  a row sees all tokens up to position 2050.
- *The PLE n-gram table* at the second layer: 16 hashed bigram and trigram heads over a table of 320,001,536 rows of
  160 values, a gate per stream and a dilated depthwise conv.
- *The MTP block:* one QSA + MoE decoder layer with hyper-connections and FP8 experts.

The GDN layers have the dimensions of the 27B. Their output gate is sigmoid(z), not silu(z).

The checkpoint stores the routed experts in NVFP4, one tensor per expert and projection, and all other weights of the
model in BF16 (attention, GDN, shared experts, routers, hyper-connections, embeddings, lm_head, the vision tower). One
file of 53.7 GB (`model-fp8-mtp-ple.safetensors`) holds the n-gram table (128 FP8 shards, 51.2 GB, one bf16 scale) and
the MTP experts in FP8. In every expert, gate and up share their global scale, and each layer and projection has one
input scale. The MoE kernels use both facts.

### 10.2 The reference, the weights and the n-gram table

**The reference model** (`engine/model/qwen4exp.py`) mirrors `modeling_qwen4_exp.py` of transformers op for op.
`Qwen4ExpConfig` subclasses `Qwen35Config`, and the GDN layers, the rotary embedding and the norms come from
`qwen35.py`, which stays unchanged. `tests/test_qwen4exp_small.py` runs a tiny model on the CPU: in fp32 the reference
is bit-identical to transformers one-shot, split, stepwise, with a cache, with eos and with an image. On the real
checkpoint, `tools/qwen4exp_reference.py` loads 77.8 GB in 34 s and gives coherent greedy text at about 1 s per token.
Its tokens are the ground truth of the golden test (`tests/data/qwen4exp_golden.json`).

**The loader** (`load_qwen4exp`, `engine/weights/qwen4exp_loader.py`) allocates the stacked experts of each layer:
`w_gu` u8 [512, 1280, 1280] (the gate rows, then the up rows), `w_dn` u8 [512, 2560, 320], their e4m3 block scales,
and `gs` and `in_scale` f32 [512, 3]. The direct reader writes the bytes of each expert straight into its slot
(`read_tensors(..., into=)`), so no expert is copied a second time. The loader skips the vision tower, the MTP block
and the n-gram shards. It checks the hash buffers of the checkpoint against the values of `engine/weights/ple.py`.
`tools/verify_qwen4exp.py` checks the index, the dtypes and shapes, the hash buffers, the table rows and the expert
scales without the GPU. The server loads the weights in about 34 s.

**The n-gram table** stays on disk. The host computes the 16 row ids of a token from the token and the two tokens before
it (`ngram_rows`); an eos starts a new context. `NgramTable` reads each 160-byte row with an O_DIRECT read of the 4 KB
page or pages that hold it, so the page cache does not keep 4 KB for each row. The C++ reader `ops().ple_gather`
(`csrc/ple_io.cpp`, a host thread pool of preads) reads about 170k cold rows per second at 128 rows per call, and 370k
to 410k at 4096. `tools/ple_hotset.py` measured the hot set: an LRU row cache of 0.5 to 5 GB hits 51% to 68% of the rows
on WikiText and 83% to 90% on TinyStories.

**The feeder** (`PleFeeder`, `engine/weights/ple.py`) serves the rows to the engine. It holds a set-associative row
cache of 8 ways in C++ (`csrc/ple_io.cpp`), of `--ngram-cache-gb` (default 1.0 GB, 6.25M rows, 1.1 GB of host memory).
Before each replay of a decode step or a cycle, the host fills the graph's row buffer (`feed_buffer`) with the rows of
its tokens (the family's `before_replay` hook). A cold 2048-token prefill chunk takes about 78 ms of row reads, and a
warm decode step about 0.03 ms. The hit rate was 50% on the golden prompts and 94% on the passkey runs.

### 10.3 Decode

`engine/model/fast_qwen4exp.py` (`FastQwen4Exp`, `FastState4`) is the decode path: the reference with each op on the
kernels of the engine. A decode step, or a verify of T rows, is one CUDA graph. The routed experts stay NVFP4, and the
other matrices stream through a BF16 format of the skinny GEMM (`Bf16Linear`, `StackedBf16Linear`): the exact path of
the checkpoint. The GDN layers run the 27B's kernels with the sigmoid gate. The PLE layer runs as torch ops inside the
graph, with sums in a fixed order.

**The MoE decode kernels** (`csrc/moe.cu`, `KernelMoE` in `engine/model/moe_kernels.py`). `moe_route` takes the top
10 experts of each row (an fp32 softmax, then score descending and index ascending). `moe_gate_up` and `moe_down` run
the expert GEMMs, and `moe_combine` sums the expert outputs and the shared expert. The grid covers the 10 M pair slots
(row, k). The warp of the first pair of an expert owns that expert: it streams the expert's weights once, with the
weight-streaming pattern of the skinny GEMM, for all the rows that chose it. Each pair keeps an fp32 partial, and the
combine sums the partials in slot order. The router and the shared expert run on the BF16 format of the skinny GEMM
(`skinny_core.cuh` holds the helpers that `skinny.cu` and `moe.cu` share). The kernels stream 208 GB/s at 10 distinct
experts and 238 GB/s at 75. A full MoE layer takes 196 µs at 1 row, 935 µs at 8 rows and 1635 µs at 16 rows: 9.4 / 45
/ 78 ms over 48 layers.

**The QSA kernels** (`csrc/qsa.cu`, `engine/model/qsa_kernels.py`). `qsa_prologue` norms and ropes the indexer
queries and writes the raw key of each new row. `qsa_block_keys` makes the key of each completed block: the fp32 mean
of 4 raw keys, k_layernorm, then RoPE at the first position of the block. `qsa_score` scores the visible blocks of
each row, and `qsa_topk` keeps the top 512 in ascending index order. `qsa_attn` runs the decode attention of
`attn_tc.cuh` in its list mode over the selected keys. The selection equals a stable sort and `torch.topk` with forced
ties, up to 65536 blocks. `qsa_attn` over 2051 keys takes 9.3 µs at T = 1 and 72 µs at T = 8, at any context length.
`qsa_score` takes about 24 to 50 µs at 65k blocks. `qsa_topk` takes about 70 µs at 65k blocks (about 0.85 ms per
token over 12 layers at 262k), and it is the first kernel to make faster.

**The hyper-connection kernels** (`csrc/hc.cu`, `KernelGatedResidual` in `engine/model/hc_kernels.py`). `hc_norm` is
the grouped form of `norm.cu`. `hc_mix` streams W_up (about 200 GB/s, 33 µs at M = 1) and computes the sigmoid mix and
the mean of the 4 streams. `hc_post` adds the injected output of the sublayer to each stream. The kernels follow the
rounding points of transformers: `hc_post` and the grouped norm are bit-exact against it. A full gated residual takes
14 / 40 / 74 µs at M = 1 / 8 / 16. The down | inject product runs on `bf16_gemv`, because the bf16 matmul of torch is
not row-invariant. A move to the skinny GEMM gave nothing: `bf16_gemv` matches it at M ≤ 8.

**The measured step.** At 1k context, a decode step takes 46.0 to 46.6 ms (21.4 to 21.8 tok/s) on the exact path, and a
verify of 8 rows takes 63 to 64 ms. The profile of the step: the BF16 skinny GEMMs 61% (36.5 ms), the experts 18%, the
hyper-connections 11% and their GEMV 7%, GDN 2%, attention and the indexer under 1%. With the decode copies (section
10.7), the step takes 39.4 ms and the verify of 8 rows 56.1 ms. The kernels fix hidden 2560, 4 streams, the low-rank
width 320 and 512 blocks, and the slot length must be a multiple of 16.

**The host's row read.** The host reads the n-gram rows of a step after the step before it ends, so the GPU waits for
the read. New text brings new n-grams, so most rows of a step miss the row cache. A miss is an O_DIRECT read of about
0.2 ms from the drive. The gather first ran 15 misses on 2 threads: 5.4 ms per plain step and 10.4 ms per cycle, and
plain decode ran at 19.6 tok/s against the 25.4 tok/s of the graph. The gather now runs one thread per miss, up to 128
(`gather_threads` in `engine/weights/ple.py`): 16 cold rows take 0.5 ms, and a cold chunk of 4096 rows takes 6.5 ms
instead of 36 ms. Plain decode then runs at the speed of the graph: 25.0 to 25.2 tok/s through the scheduler.

### 10.4 Prefill

`engine/model/prefill_qwen4exp.py` runs a prompt in chunks of 4096 tokens, each layer over the whole chunk, on the
state of the decode model. The BF16 matrices run on cuBLAS. The MoE runs on the grouped GEMM, the GDN layers on the
27B's prefill kernels, and the hyper-connections on `csrc/hc_prefill.cu` (167 ms per chunk instead of 763 ms in torch
ops). The last token's mixer and lm_head run on the decode modules.

**The MoE prefill** (`csrc/moe_prefill.cu`, `KernelMoE.prefill`) sorts the (token, slot) pairs by expert. Two grouped
NVFP4 GEMMs then run, gate | up and down, with the CUTLASS 4.8 SM120 block-scaled ptr-array kernel: one group per
expert, compiled and run on sm_121a. A gather kernel quantizes the rows of each group, a SwiGLU kernel quantizes the
input of the down GEMM, and a combine sums in slot order. The device writes the group arguments, so nothing waits for
the host. A layer takes 8.8 ms per 2048-token chunk (11.3 ms at 4096, 16.6 ms at 8192): a MoE ceiling of 4.8k / 7.5k /
10.3k tok/s. The raster swizzle of the scheduler matters: without it, the grouped GEMM runs 2.3 to 3 times slower.

**The sparse attention of prefill** (`csrc/qsa_prefill.cu`). The indexer scores all rows of a chunk in one launch, with
the bits of the decode scores (0.13 / 0.75 / 4.6 ms for 2048 rows at prefixes 0 / 16k / 128k). The union kernel
attends each group of 4 rows over the union of their selected blocks. On real text, that union is 3100 to 3700 keys,
and a layer takes 12.8 / 15.0 ms at prefixes 8k / 64k. Attention with the selection as a mask takes 21 / 108 ms, so
the union is the default path.

**The feeder** reads the rows of a chunk from the tokens and the slot's two tokens before it (`prev2`). Its worker reads
the next chunk's rows while the GPU runs the current one: within a prefill call, and between the scheduler's calls,
which give each call the tokens of the next one (`next_ids`). A 6,375-token prompt in two chunks prefills at 2,657
tok/s, against 1,624 tok/s before the prefetch and the gather threads.

**The chunk** is 4096 tokens. On the real checkpoint it is 6 to 13% faster than 2048: 2760 to 2810 tok/s at prefixes of
0 to 8k, and about 2500 tok/s at 16k to 60k with cold rows. On random weights (`bench/prefill_chunk_bench.py`), 8192
adds 8% more, but it holds the lane for 3 s. A call per chunk gives the same bits as one call.

The MoE prefill quantizes its activations to NVFP4, and the decode MoE does not. A small input change can flip one of
the top 10 of 512 experts. Thus the last-token logits of prefill and decode differ by 10 to 15% in relative L2. Both
paths give the same tokens on the golden prompts and on a 3000-token prompt, and both pass the perplexity gate
(section 10.9).

### 10.5 Speculation

`engine/spec/mtp4.py` (`Mtp4`, `Mtp4State`, `Mtp4Cycle`) is the drafter of the family, with the entry points of the
27B's `engine/spec/mtp.py`. The wiring comes from the source of vLLM:

- The input of a draft row is the 4-stream hidden [4 × 2560] before the final mixer: the target's on the first step,
  the drafter's own on the chained steps.
- The projected embedding enters every stream as a plain add. One norm runs over the whole 10240-wide hidden, then the
  shared `fc_hidden` runs on each stream.
- The decoder layer is a QSA + MoE layer with its own KV and indexer, and a per-stream norm in the final mixer. The
  target's lm_head is the draft head, over the 64k draft vocabulary: a BF16 copy of those rows (0.36 GB per draft
  step), or the low-rank head of the prepared directory (below).
- The drafter selects its sparse blocks on every row with the target's kernels. vLLM reuses the selection of step 0;
  both are exact under the accept rule.

The FP8 block experts of the MTP block become NVFP4 copies on `KernelMoE` (1.41 GB). The first load requantizes them in
7.5 s. Later loads read them in 0.3 to 0.9 s from `~/.cache/colinfer/drafter/quant4-*.safetensors`.

A cycle verifies [y, d1..dk] in one graph, accepts with `engine/spec/accept.py`, commits, and then drafts: one catch-up
row per verified position, and k - 1 chained steps. The host fills the n-gram rows of the k + 1 rows before each
replay. The scheduler picks k from 1, 2 or 3 per cycle by the acceptance of the request, and a request starts at
k = 2. Its cost model (`qwen4exp_cycle_cost` in `engine/model/family.py`) is a least-squares fit at 1k context of the
served configuration (the decode copies and the low-rank head): cycle_ms = 24.2 + 0.64 × distinct experts + 1.06 × k,
with about 11.2 + 5.45 k distinct experts per layer in a verify. (The exact path with the full head gave 41.4 + 0.577 ×
distinct + 2.92 × k.) The model adds about 110 µs per attention layer launch at 262,144 tokens for the block selection;
at 262,000 tokens the cycles took 1.2 to 2.2 ms more than at 1k, against the model's 1.4 to 1.65 ms. k = 4 adds
nothing. Suffix-match drafts verify up to 15 copied tokens, as for the 27B.

**The low-rank draft head** is the 27B's design on a BF16 lm_head. `tools/lowrank_draft_head.py --model` collects
12,369 drafter outputs from real cycles and keeps their top 512 principal directions U. The head then scores (g U)(W
U)^T on two NVFP4 GEMMs (about 21 MB) and rescores its top 256 candidates exactly against the BF16 rows of lm_head
(`ops().rescore_bf16`, `csrc/sampling.cu`, one warp per candidate). The full head's draft is among those candidates for
99.8% of the outputs at rank 512 and for all of them at rank 1024. The drafter then holds no BF16 copy of the head rows.

**The draft early exit** skips the GEMMs of the chained steps when the product of the draft probabilities falls below
`DRAFT_STOP` (0.1, the 27B's value). With k of 3 at most, it fires in 0 to 7% of the cycles, and a threshold of 0.03
gives the same tok/s: the threshold stays.

Measured in the served engine (`tests/scheduler_check_qwen4exp.py`, 2026-10-24, with the decode copies and the
low-rank head), against plain decode in the same process:

| Text | Plain decode | Speculation | Speed-up | Tokens per cycle |
|---|---|---|---|---|
| Code | 33.9 tok/s | 68.6 tok/s | 2.02x | 3.19 |
| Prose | 33.9 tok/s | 47.2 tok/s | 1.39x | 2.07 |

On the exact path with the full head, before the gather fix (section 10.3), the same checks gave 39.1 against 21.3
tok/s on code and 27.2 against 20.3 tok/s on prose. On the GPU alone (`tests/golden_qwen4exp_spec.py --prepared`),
k = 3 gives 60.1 tok/s on the golden prompts (1.74x the 29.0 ms plain step) and 57.7 tok/s on WikiText (1.68x).

### 10.6 The serving state

The hidden state of a row between layers is the 4-stream x4 [4 × 2560], not one 2560 vector. `FastState4` holds the
27B's state (fp8 K / V per attention layer, the GDN conv and recurrent state) and, per slot:

- the raw indexer keys `ik` bf16 [L, 128] and the block keys `ikb` bf16 [L / 4, 128] of each attention layer
- `rope_pos` int32 [L, 3], the rotary position of each cached row
- `ple_conv` bf16 [10240, 9], the last 9 inputs of the dilated conv of the PLE layer

The scheduler, the checkpoint ring, `_copy_kv`, the slot reset and the conversation cache see the state only through
two hooks. `kv_tensors()` gives the per-position tensors (time on dim 2; `ikb` has one row per 4 positions), and
`snap_tensors()` gives the recurrent state of a slot. The scheduler also takes the family's `prefill_fn`, `prepare_fn`,
`feed_buffer` and `before_replay(slot, tokens)`, which fills the n-gram rows right before each replay. The ring takes
16 snapshots (1.85 GB), and it needs no new tensor: the drafter's input, the 10240-wide hidden of the last position,
fits its buffers.

`Mtp4State` is the drafter's state, with the same hooks. The conversation cache (`engine/runtime/sessions.py`) saves
the drafter's `kv_tensors()` with the target's. The drafter's rows lag the target's by one, so the files save only the
complete block-key groups. The session fingerprint names the n-gram table file, the drafter and the decode copies.

Measured: a 4676-token second turn reuses 4655 tokens of its prefix (0.13 s to the first token, against 1.75 s from
scratch). A second slot keeps the first conversation cached. A 7398-token conversation that left both slots restores
from disk in 0.15 s with the drafter's rows, with the output of a restore from memory.

### 10.7 The decode copies

The exact path streams 9.95 GB per decode token, and 8.6 GB of it are BF16 matrices outside the routed experts.
`tools/requant_qwen4exp.py` (run by `tools/prepare.py --model nvidia/Qwen3.8-Flash-Next-NVFP4`, about 30 s on the GPU)
writes smaller copies of them into the prepared directory. `attach_decode_copies4` points the decode path at them when
the directory has them (`--decode-weights int`, the default); the prefill keeps the BF16 weights. The perplexity gate
decided each format: 0.5% against the exact decode path's 2.3274, on the 8 golden WikiText windows.

| Class | Format: ratio | Decision |
|---|---|---|
| attention: q / k / v / index and o_proj | INT6 1.0033 | INT6 |
| lm_head | INT6 0.9998; NVFP4 1.0089 | INT6 |
| hyper-connection W_up | INT8 per row 1.0012 (a new mode of `hc_mix`: 19.8 µs instead of 33.4 µs) | INT8 |
| hyper-connection down / inject | INT6 1.0032, but not with the others | BF16 |
| GDN out_proj | INT6 0.9992, but not with the others | INT6 by GPTQ |
| GDN in_proj | INT6 1.0103; INT5 for both GDN classes 1.0112 | INT6 by GPTQ |
| shared experts | INT6 1.0111 | INT6 by GPTQ |

With round to nearest, all classes together give 1.0218 and fail, and the three that pass together give 1.0014 (1.0043
on 8 held-out windows). The cause of the failures: a small change before the top-10-of-512 routing changes which experts
run, and the gate's own noise is about 0.25%. With those three copies, the decode step falls from 46.6 to 39.4 ms
(21.4 to 25.4 tok/s, +18%), and the bytes per token from 9.95 to 8.15 GB. With all six classes, the step takes
29.1 ms at 1k context (34.3 tok/s on the GPU) and the verify of 8 rows 51.0 ms; the bytes per token fall to about 5.5
GB (computed from the formats; the plan's estimate was 4.7 GB).

`tools/gptq_qwen4exp.py` makes calibrated copies of the GDN projections and the shared experts' gate | up. It prefills
48 windows of 2048 tokens of text that the gate does not use (WikiText train, Python functions, TinyStories), and
collects the second moment H = X^T X of each input: a torch function mode sees the prefill's matmuls with those weights.
GPTQ then quantizes each matrix one column at a time and moves the error of each column onto the columns not yet
quantized, weighted by the inverse of H. The layout is the engine's INT6, so the decode path reads the files unchanged.
In act order (the columns by descending H diagonal, with static group scales), the output error on the calibration
inputs falls 2 to 6 times against round to nearest. All six classes together give 1.0029 on the gate windows, 1.0016
on the held-out windows and 1.0023 on all 16 (the three round-to-nearest classes alone: 1.0027). In column order the
same set gave 1.0058 and failed. The gate of `tests/golden_qwen4exp.py --decode-copies`, token by token, gives the
same 1.0029 (2.3341 against 2.3274), and 1.0003 against the reference. The six classes take 3.2 GB on the GPU, beside
the BF16 weights that the prefill reads.

The INT6 lm_head writes fp32 logits: `skinny_int` has an fp32 output, as the NVFP4 and BF16 formats have. With bf16
logits, two golden prompts diverged at exact ties of those logits. With fp32 logits, both give the reference's 32
tokens again, and only the known near-tie of the first prompt remains (a reference margin of 0.125).

### 10.8 Memory

Bytes in GB (1e9), at the defaults: 2 slots of 262,144 tokens, 16 checkpoints, the drafter, the decode copies and a
1 GB row cache.

| Item | GB |
|---|---|
| Weights: routed experts NVFP4 with scales, and the BF16 rest | 77.8 (78.5 on the GPU after the load) |
| Decode copies, six classes | 3.2 |
| Drafter: weights and the low-rank draft head | 1.69 (1.95 with the full head; about 2.8 with its state and its cycle graphs) |
| Slot state, 2 slots: fp8 KV, indexer and block keys, GDN and PLE state | 8.7 (and 0.71 for the drafter's rows) |
| Checkpoint ring, 16 snapshots | 1.85 |
| n-gram row cache (host memory) | 1.1 |
| `QWEN4EXP_FIXED_BYTES`: everything except the slot state and the row cache | 93.2 |

The fixed bytes are the measured 87e9 (the fall of `MemAvailable` of a server without the drafter, less its state and
row cache), the drafter's 3e9 and the copies' 3.2e9. With the drafter and the six classes of copies, `MemAvailable`
falls by 101.9 GB at the end of the build (91.4 GB of it besides the state and the row cache), and the server is ready
with 94.3 GB allocated and 97.1 GB reserved. The memory check (`engine/runtime/build.py`) thus asks for about 110 GB
with the 6 GB growth reserve. The first long prompts and an image add about 4.4 GB, inside the reserve. The torch
allocator cap is per family: 104 GB here, 95 GB for the 27B. In the plan's budget, three slots do not fit with the
drafter.

### 10.9 Exactness

The evidence, in order:

1. The reference is bit-identical to transformers in fp32 on a tiny model (`tests/test_qwen4exp_small.py`).
2. Every decode kernel is row-invariant: the pair-slot MoE grid has no split-K and no atomics, and its combine sums in
   slot order. A verify of 8 rows equals 8 single steps, bit for bit (`tests/test_fast_qwen4exp.py`,
   `tests/test_mtp4.py`). Thus decode equals verify (section 7).
3. The decode path and the prefill give the reference's 32 greedy tokens on the three golden prompts
   (`tests/golden_qwen4exp.py`). The one divergence of the decode path is a near-tie: the reference's bf16 margin is
   0.125, and its own fp32 logits pick the engine's token. The rule is now a margin below 0.2.
4. Speculation gives the tokens of plain decode for k = 1 to 4, greedy and sampled (`tests/golden_qwen4exp_spec.py`),
   and in the served engine (256 of 256 tokens on code and prose, `tests/scheduler_check_qwen4exp.py`).
5. The perplexity gate (`tests/perplexity.py --engine qwen4exp`, 8 windows of 2048 WikiText tokens): the reference
   gives 2.3335, the decode path 2.3274 (ratio 0.9974), the prefill 2.3438 (ratio 1.0044). One 16k window gives 2.2186.
   The six classes of decode copies give 1.0029 against the exact decode path (1.0016 on 8 held-out windows).
6. Passkey retrieval (`tests/passkey_qwen4exp.py`): 9 of 9 at 16k, 32k and 64k. With YaRN x2 (one slot of 524,288
   tokens): 9 of 9 at 16k, 288k and 461k (prefill 2,400 tok/s at 461k).
7. Images: the reference with an image is bit-identical to transformers on the tiny model. On the real checkpoint, the
   served engine gives the reference's 32 greedy tokens for a page of text in an image (768 image tokens,
   `tools/qwen4exp_reference.py --image`, `tests/data/qwen4exp_golden_image.json`), with speculation as without.
8. YaRN: the reference with YaRN equals transformers' `yarn` rope type on the tiny model, at 70 tokens over 32 native
   positions (`tests/test_qwen4exp_small.py`). Without the fold into the indexer's norms, the logits differ by 0.24.

The kernel tests compare against the reference with a tolerance: exactness is required between the paths of the
engine, not against the rounding of transformers. The top-k orders by score descending, then index ascending, with
-0.0 read as +0.0, in every path. Ties are common: a block whose 4 dot products are all 0 or less scores exactly 0.

### 10.10 Design decisions

- **One swizzled copy of the expert scales.** The decode kernels read the CUTLASS swizzled layout that the prefill GEMM
  needs. A second, row-major copy would cost 7.5 GB, and a swizzle at prefill time would add 22% to prefill.
- **Always-sparse decode attention.** Decode and verify always run `qsa_attn`. A short row gets the dense list (all its
  positions). A per-launch choice of kernel would make the bits of a row depend on its neighbours. There is no dense
  fallback.
- **The n-gram table stays on disk.** 51 GB does not fit beside the experts. The bounded row cache of the feeder holds
  the hot rows.
- **Images as for the 27B.** The vision tower of the checkpoint has the 27B's architecture with a 2560-wide output, so
  `engine/model/vision.py` runs it unchanged. The image rows hash as the image pad token in the n-gram rows, as
  transformers passes `ple_input_ids`. The prefill records the (t, h, w) position of each row before it makes the block
  keys, and the decode rows rotate by their position plus the slot's `rope_delta`.
- **YaRN in four norms per layer.** Past the 262,144 native positions, `apply_yarn4` (`engine/model/qwen4exp.py`)
  scales the rope frequencies by the ratio, as for the 27B. It folds YaRN's attention factor into the rotated dims of
  four norms per attention layer: the attention's q / k norms and the indexer's. transformers gives the indexer the
  attention's scaled cos and sin, and the block keys take the factor through `k_layernorm`, before their rotation. The
  drafter folds its own norms. The load refuses a slot of more than 4 times the native context, the model card's
  1,000,000 tokens.

### 10.11 What remains

- a decode copy of the hyper-connections' down | inject product (0.64 GB per token). Its round-to-nearest INT6 copy
  passes the gate alone, but not with the other copies. A GPTQ copy needs the second moments of its 10240-wide inputs:
  0.42 GB each, for 97 groups.
- a fine-tuned drafter, for the acceptance on prose (about 0.44 of the drafts, against 0.73 on code).
  `tools/train_drafter.py` trains the 27B's dense MTP head, not this MoE drafter.
- a faster `qsa_topk` at long context: one block per row. The block selection adds 1.2 to 2.2 ms to a cycle at
  262,000 tokens.
- video input, as for the 27B.

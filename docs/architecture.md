# Architecture

How colin-inference-engine runs `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark. This page describes the engine as it is;
the dated logs in `docs/history/` (`phase*`, `baseline.md`, `results.md`) record how it got here, with the
measurements behind each choice and the alternatives that were tried and dropped.

## 1. The machine and the model

**GB10 (DGX Spark):** 48 SMs of sm_121 (Blackwell consumer ISA: `mma.sync` tensor cores, no WGMMA / tcgen05), 99 KB of
shared memory per block, 24 MB of L2, 128 GB of unified LPDDR5x that streams at about **238 GB/s** in practice. Decode
reads every weight once per step, so decode speed is weight bytes over bandwidth; prefill is compute-bound on the
tensor cores.

**Qwen3.8-27B:** 64 layers (48 Gated DeltaNet linear-attention layers, 16 gated full-attention layers, 3:1 interleaved),
hidden 5120, SwiGLU MLP with intermediate 17408, vocabulary 248,320, an untied lm_head and one MTP block (a single
extra decoder layer for multi-token prediction).

- *Attention:* 24 query heads over 4 KV heads (GQA), head dim 256, partial RoPE on the first 64 dims (θ = 10⁷), q / k
  RMSNorm, and an output gate (`q_proj` emits q and a gate per head; the output is multiplied by sigmoid(gate)).
- *Gated DeltaNet (GDN):* 16 key heads and 48 value heads of 128 dims, a depthwise causal conv (kernel 4) over the q / k
  / v channels, L2-normalized q and k, and a recurrent fp32 state S [128 × 128] per value head updated per token by the
  gated delta rule `S ← exp(g) S;  S += k ((v − Sᵀk) β)ᵀ;  o = Sᵀq`, followed by a gated RMSNorm with silu(z).
- *Norms:* zero-centered RMSNorm, y = x / rms(x) · (1 + w).

`engine/model/qwen35.py` implements all of this in plain PyTorch, op for op like transformers' `modeling_qwen3_5.py`.
It is not used for serving: it is the reference that every kernel is tested against.

## 2. Weights

| Tensor | Stored as (checkpoint) | Decode streams | Prefill reads |
|---|---|---|---|
| MLP gate / up / down (64 layers, 9.6 GB) | NVFP4 (e2m1 + e4m3 scale per 16 + fp32 global), 4.5 bits | the same NVFP4 | the same, CUTLASS-swizzled scales |
| lm_head (0.7 GB) | NVFP4 | NVFP4 | NVFP4 (only for all-token logits) |
| attention q / k / v / o (16 layers) | FP8 e4m3, per-tensor scale | **INT6 copy** (1.36 GB), else the FP8 | FP8 (W8A8) |
| GDN in_proj_qkv / in_proj_z / out_proj (48 layers) | FP8 | **INT5 copy** (3.8 GB), else the FP8 | FP8 (W8A8) |
| GDN in_proj_b / in_proj_a, conv, norms, embeddings | BF16 | BF16 | BF16 |
| MTP block | BF16 | **NVFP4 copy** (round-to-nearest) | BF16 (cuBLAS) |

The INT6 / INT5 copies (`tools/int6_requant.py`, from the BF16 checkpoint `Qwen/Qwen3.8-27B`) have a lower weight error
than the checkpoint's own FP8 (2.24% vs 2.67% relative) at 6.5 / 5.5 bits per weight: signed q with an e4m3 scale per 16
weights and an fp32 global scale, stored as an NVFP4-layout nibble plane plus a high-bit plane
(`engine/weights/quantize.py`). With them, WikiText perplexity moves −0.21% and Python-code perplexity +0.23% against
the FP8 weights, and decode is ~9.5% faster. Projections that share an input (q / k / v; in_proj_qkv / in_proj_z) are
stacked into one matrix, so they cost one weight pass (`StackedFp8Linear`, a shared global scale in the copies).

On the GPU: 20.2 GB of checkpoint weights + 5.2 GB of INT copies. A decode step streams ≈15.5 GB.

## 3. Decode

`engine/model/fast.py` turns the reference module tree into the decode model (`load_fast_model`, `to_fast`): quantized
linears become kernel modules, attention / GDN / norms become kernel layers, positions move to the device
(`FastState.pos_t`), and the KV cache is fp8 (e4m3, unit scale, saturating: 32 KB per token). `DecodeGraph` captures
one decode step (embedding → 64 layers → lm_head → sampler) as a CUDA graph; the host only writes the input tokens and
replays.

Per layer, in order (every kernel launched with programmatic dependent launch, so the next GEMM starts streaming its
weights while the previous kernel finishes):

| Full-attention layer | GDN layer |
|---|---|
| RMSNorm (`norm.cu`) | RMSNorm |
| q / k / v projection, one INT6 weight pass (`skinny.cu`) | q / k / v / z projection, one INT5 pass; the b / a gates (bf16 GEMV, `gemv.cu`) on a parallel branch |
| `attn_prologue`: q / k RMSNorm, partial RoPE, k / v written to the fp8 cache at `pos_t` | `gdn_conv`: causal conv + SiLU over the conv window; `gdn_conv_commit` advances the window |
| `attn_decode`: multi-row tensor-core attention, output gate fused (`attn_decode.cu`) | `gdn_delta`: L2 norms, gated delta rule on the fp32 state, gated RMSNorm (`gdn_step.cu`) |
| o_proj + residual | out_proj + residual |
| RMSNorm, SwiGLU (gate and up in one NVFP4 pass), down + residual | the same MLP |

**The skinny GEMM** (`csrc/skinny.cu`) is the engine's central kernel: every decode-time linear runs on it, in NVFP4,
INT6 / INT5 and FP8. A warp owns 16 weight rows and streams them in 256-byte runs per row (one chunk ahead in
registers); the chunk is transposed through a small shared-memory scratch into `mma.sync.m16n8k16` fragment order and
dequantized exactly into bf16 (e2m1 × e4m3 fits in bf16; INT codes are built with a 0x4300 | c bit trick). The
activations ride along as the mma's other operand, so 16 rows cost about what 1 row costs: the kernel is bound by
weight bytes alone. Split-K by shape fills the 48 SMs. It streams at 222-236 GB/s in isolation, within 2-5% of a
pure-streaming kernel with the same access pattern.

**Attention** (`attn_decode.cu`) serves all query rows of a slot (24 heads × T new tokens, up to 48 rows) from one pass
over that slot's KV: K stays raw fp8 in shared memory, `mma.sync` f16 tiles compute QKᵀ and PV, 12 key-tile stripes per
KV head fill the SMs, and a combine kernel folds the stripes and applies the output gate. A verify of 8 rows reads the
cache once, at the cost of plain decode.

**GDN** (`gdn_step.cu`) keeps each value head's 64 KB fp32 state in registers across all T tokens (one block of 512
threads per head and slot), computes everything that does not depend on the state (norms, β, decay) before the
sequential chain and the gated RMSNorm after it, so a token step is two register passes and one `__syncthreads`. The
same kernels do plain decode (T = 1, state advanced), speculative verify (T rows, state untouched) and commit (state
advanced by the accepted count). At T = 8 a layer costs ≈17.5 µs (reading the 3 MB of state) + ≈0.5 µs per token.

Plain decode at 8k context: **74 ms per token** (13.5 tok/s), i.e. the ≈15.5 GB stream at ≈210 GB/s.

## 4. Speculative decoding

`engine/spec/mtp.py`. Each request's decode runs as speculative cycles; one CUDA graph covers a whole cycle (`MtpCycle`):

1. **Verify** the k + 1 rows [y, d₁ … d_k] (the last token and k drafts) on the target: logits and post-norm hidden states
   for every row, KV written for all rows, GDN state untouched (`FastQwen35.verify`).
2. **Accept** the leading drafts that equal what the target itself emits at each row: its argmax (greedy), or its sample
   drawn with a uniform keyed by (seed, position) (`engine/spec/accept.py`). The first mismatch is replaced by the
   target's token and, if all match, the last row gives a bonus token, so a cycle emits 1 … k + 1 tokens and **the output
   is exactly the output of plain decoding**, whatever the drafts were. The accepted length is cut after a stop token.
3. **Commit** the accepted prefix: GDN conv windows and states advanced, positions moved (KV past the accepted length is
   simply overwritten later). It runs on a parallel graph branch, since drafting never reads the GDN state.
4. **Draft** the next cycle's k tokens with the MTP block: a catch-up row per newly committed position with the target's
   true hidden states, then k − 1 chained steps that feed the block its own output. Its KV cache is fp8 as well.

Making the drafter cheap (it is re-streamed every draft step):

- its linears run on NVFP4 copies on the skinny GEMM;
- the draft head scores a static vocabulary of the 64k most frequent tokens (`draft_vocab.npy`) plus up to 4k tokens of
  the current prompts, not 248k rows;
- with `draft_head_pca.safetensors`, the draft head is a **low-rank approximation** (g U)(W U)ᵀ (U: the top 1024
  principal directions of the drafter's outputs, ≈43 MB instead of ≈200 MB) whose top 256 candidates are rescored
  exactly against the real NVFP4 rows (`rescore_nvfp4`): the full head's argmax is among them for 99.96% of drafts, and
  measured acceptance is unchanged;
- **early exit**: once the product of the drafter's probabilities of the drafts so far drops below 0.1 for every active
  slot, the remaining draft steps' GEMMs return at once (a device flag read by the skinny GEMM, `skinny_skip`): ≈0.35 ms
  instead of ≈1.8 ms per step. Their junk drafts are rejected by the next verify.

**Draft length.** The scheduler picks k = 3 or 7 per cycle for the most expected tokens per second: each slot's running
per-token acceptance a gives (1 − a^{k+1}) / (1 − a) expected tokens, over the cycle time measured at startup plus the KV
reads that grow with context. Verify rows must fit one skinny pass (width × (k + 1) ≤ 16), so three decoding slots use k = 3.

A k = 7 cycle at 8k context takes **86 ms**: verify 73.5 ms (69.5 ms of weight GEMMs), drafting ≈11.6 ms. With the
fine-tuned drafter a k = 7 cycle accepts 3.41 tokens on average (code 5.55, prose 2.77).

## 5. Prefill

`engine/model/prefill.py`, one slot at a time, in chunks of 2,048 tokens (larger chunks make the FP8 GEMMs slower per
token), each layer over the whole chunk on tensor cores:

- **MLP (W4A4):** the fused add + RMSNorm kernel also quantizes the activations to NVFP4 with the checkpoint's static
  input scale; CUTLASS SM120 block-scaled GEMMs (`gemm_nvfp4.cu`) run gate, then up with silu(gate) · up and its NVFP4
  quantization fused into the epilogue, then down with the residual in the epilogue.
- **FP8 projections (W8A8):** activations quantized to e4m3 (static input scale) in the same fused norm kernel;
  cuBLASLt via `torch._scaled_mm` with row-wise weight scales.
- **Attention:** the decode prologue on all T rows (writing the fp8 KV), then FlashInfer FA2 over a bf16 copy of the
  cached prefix; past 16k tokens of context `attn_prefill.cu` instead (Q Kᵀ on FP8 tensor cores over the cache as
  stored: 6% faster prefill at 64k, 11% at 128k; perplexity +0.25-0.3% for those chunks from rounding Q to e4m3).
- **GDN:** causal conv + SiLU + q / k L2 norm in one kernel continuing the conv window, then the chunked gated delta rule
  (`gdn_prefill.cu`: a WY kernel per 64-token chunk builds the triangular inverse, a chunk kernel per value head and
  64-wide V slice carries the state in registers, 2.35× FLA), gated RMSNorm, FP8 out_proj.

The state afterwards is exactly what decode expects. With speculation the drafter's KV rows for the chunk are written
right after it (`Mtp.prefill`). Times: 2k 0.54 s (3.8k tok/s), 8k 2.29 s, 32k 10.9 s, 126k ≈66 s.

## 6. Serving

`engine/runtime/scheduler.py` and `engine/server/api.py`.

- **Slots.** One batched state holds `--slots` (3) slots of `--max-seq-len` (262,144) tokens with contiguous KV. CUDA
  graphs are captured at startup for every contiguous slot range × {greedy, sampled} × draft length; a step replays
  the smallest range covering the decoding slots, with the others masked off (`state.active`). Every kernel computes a
  slot's rows the same way at any width, so batching never changes an output.
- **Engine loop.** Each step admits queued requests, prefills at most one chunk of one prompt, then runs one decode
  cycle for the decoding slots: a long prompt slows the others for one chunk at a time, never stops them.
- **Prefix checkpoints.** A ring of GDN-state snapshots (conv + recurrent, 154 MB each, 32 by default) is taken at the
  end of each prompt and reply, at the end of the first message and before the last one, and every 8,192 prompt tokens.
  A request restores the longest checkpoint that is a proper prefix of its prompt (attention KV is resumable at any
  length; GDN state only where a snapshot exists) and prefills the rest; if that slot is busy its KV prefix is copied.
- **Server.** One engine thread owns the GPU; asyncio handlers render the chat template, tokenize and stream. The
  engine thread detokenizes and parses each token as it is emitted (reasoning / content / Qwen XML tool calls, stop
  strings: `engine/server/chat.py`). A CUDA error fails all requests and exits (systemd restarts the process).
- **Memory** at startup: ≈61 GB (weights 20.2 GB, INT copies 5.2 GB, KV 3 × 262k × 32 KB = 25.8 GB, checkpoint ring
  4.9 GB, drafter KV and copies ≈2.5 GB).
- **Startup** ≈25 s: weights, a numerical self-test of every matmul path (`engine/selftest.py`: refuses to start on a
  wrong kernel), graph capture with cycle-time calibration, warm-up.

## 7. Numerics and the identity guarantee

Speculation and batching are only safe if they cannot change an output. The engine guarantees it by construction:

- **Row-invariant kernels.** The skinny GEMM uses a fixed k order inside each mma step and a split-K partition that
  depends only on the matrix shape; rows beyond M are zeros that never touch real rows. Attention processes keys in fixed
  tiles assigned to fixed stripes and folds them in a fixed order; a tile entirely past a row's length is an exact
  no-op. The GDN kernels run the same arithmetic for T = 1 and T = k + 1. So a row's bits do not depend on how many
  other rows share the launch, and plain decode, verify, the drafter and any batch width agree bit for bit.
- **Position-keyed sampling.** The token at position t is drawn by inverse CDF from the processed distribution
  (temperature → min-p → top-k → top-p) with one uniform u = Philox(seed, t) (`csrc/sampling.cu`). A request's tokens
  depend only on (seed, prompt); speculative acceptance samples the target at each verified row with that row's
  uniform, so it reproduces plain sampling token for token.
- **Rounding points** follow the reference model (bf16 where transformers rounds to bf16); extensions compile with
  `--fmad=false` so no compiler contraction changes them.

Quality is gated by perplexity against the checkpoint (`tests/perplexity.py`, WikiText and Python code): every
departure from the checkpoint's own numerics (the INT copies, FP8 prefill attention) stays within 0.5%.

## 8. Validation

- `pytest tests/` (≈1 min): each kernel against the PyTorch reference or a dequantized fp32 product; row-invariance and
  verify = sequential-decode bit identity; prefill ops; sampler and acceptance statistics; chat parsing; the drafter
  trainer's unroll.
- `tests/golden.py`: 21 recorded requests (greedy, seeded-sampled, a 20k-token prompt), all at once, must reproduce
  their tokens and per-token logprobs bit for bit, with speculation and without (`check --plain`).
- `tests/scheduler_check.py`: the scheduler's outputs for single, concurrent, mixed (greedy next to sampled and a long
  prefill), multi-turn and shared-prefix requests must equal the uncached single-request outputs.
- `tests/perplexity.py`, `tests/passkey.py` (long-context retrieval), `tests/parity_hf.py` (the reference vs
  transformers), `tests/server_check.py` (the HTTP API).
- Benchmarks: `bench/perf.py` (the repeatable baseline: prefill, TTFT, decode, memory), `bench/decode_bench.py`,
  `bench/prefill_bench.py`, `bench/request_mix_bench.py` (end to end), `bench/skinny_bench.py`, `bench/attn_bench.py`,
  `bench/trace_summary.py` (nsys).

## 9. Where the time goes, and what is left

A k = 7 cycle (86 ms at 8k): ≈69.5 ms of verify weight GEMMs at ≈223 GB/s (the 15.5 GB floor at 238 GB/s is 65 ms),
≈3 ms of GDN recurrence / attention / norms, ≈11.6 ms of drafting (≈280 MB of drafter weights per step). The remaining
levers are mostly on the drafter (its own weights, a better stopping rule, better prose acceptance) and, for a few
percent, a tile-contiguous weight copy that bulk copies could stream (docs/history/phase6_progress.md, sections 18-19).

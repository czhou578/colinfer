# colin-inference-engine

colinfer is a single-user inference engine for two models of the Qwen3.8 family on one **DGX Spark** (GB10,
sm_121): **Qwen3.8-27B** (`nvidia/Qwen3.8-27B-NVFP4`), the default, and the MoE model **Qwen3.8-Flash-Next**
(`nvidia/Qwen3.8-Flash-Next-NVFP4`, [below](#qwen38-flash-next)). One server process serves one of them. The server
has an OpenAI-compatible and an Anthropic-compatible HTTP API, so Claude Code can use it directly. With the 27B, it
reads images (screenshots, diagrams) with the checkpoint's own vision tower. It is for one person, and it runs one
request at a time.

This is a research engine for one machine, not a general inference server. It has no paged KV cache and no multi-GPU
support. The kernels compile for sm_121a only. The design notes and the dated logs in `docs/` record each measurement
behind the design.

The memory bandwidth of this chip sets the decode speed. Each decode step of the 27B reads all the weights from LPDDR5x
memory at ~238 GB/s, so plain decode cannot go faster than ~14 tok/s. All the speed above this floor comes from
speculative decoding. The table compares the 27B with SGLang and vLLM:

| | colinfer | SGLang 0.5.21 | vLLM 0.25.1 |
|---|---|---|---|
| 40-request mix, greedy, 256 tokens, wall tok/s (`bench/request_mix_bench.py`) | **39.4** (code 57.7, prose 33.3, Q&A 38.5, structured 39.6) | 29.0 (with DFlash2) | not measured |
| Plain decode, no speculation | **13.5 tok/s** (74 ms / token at 8k context) | 12.3 tok/s | 12.3 tok/s |
| Prefill, 2k / 8k / 32k prompt | **0.54 s / 2.29 s / 10.9 s** (3.8k / 3.6k / 3.0k tok/s) | ≈1.3 s / ≈4.9 s / not measured (≈1.5k / ≈1.7k tok/s) | 0.81 s / 3.9 s / 18.0 s |

The SGLang and vLLM decode and prefill figures are the phase-0 baselines on the same checkpoint and machine
(`docs/history/baseline.md` section 4). These runs used FP8 KV and no prefix cache, and they measured decode on a short
prose prompt. The SGLang prefill figures come from its own per-batch log. The SGLang mix figure is SGLang with the
DFlash2 drafter on the same 40 requests (`docs/history/phase6_progress.md` section 16).

Outputs are **token-identical with speculation on or off**, for greedy and for seeded sampling
(`tests/golden.py`, `tests/scheduler_check.py`). Each quantization choice beyond the checkpoint's own must pass a
perplexity gate of ≤ 0.5% against the checkpoint (WikiText and Python code).

## Requirements

- A DGX Spark: NVIDIA GB10 (compute capability 12.1) with 128 GB of unified memory. Other GPUs are not supported.
- The CUDA 13.0 toolkit (nvcc) and a 580-series driver (tested with 580.173.02), on aarch64 Linux.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the pinned dependencies.
- The checkpoint `nvidia/Qwen3.8-27B-NVFP4` (21.9 GB) from Hugging Face. The optional INT decode copies (below) also
  need the BF16 checkpoint `Qwen/Qwen3.8-27B` (55.6 GB).
- For the 27B: about 105 GB of free unified memory (in units of 10^9 bytes, as `--mem-cap-gb`) while the server runs.
  Stop other GPU work first: an over-commit of the shared memory can power off the machine instead of an out-of-memory
  error. The server does not start with less.
- Up to 64 GB of disk for the conversation cache (`--session-cache-gb`, under `~/.cache/colinfer/sessions`).
- Optional, for the second model: the checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` (132.7 GB). Its server takes
  about 100 GB of unified memory at startup with the drafter. The memory check asks for about 108 GB of `MemAvailable`
  (the 6 GB growth reserve included), and the server does not start with less.

## Quick start

```bash
git clone --recursive https://github.com/czhou578/colinfer.git && cd colinfer   # CUTLASS is a submodule
uv sync                                                  # Python 3.12, torch 2.14 (cu130), FlashInfer 0.7 (sm121a)
uv run hf download nvidia/Qwen3.8-27B-NVFP4             # into the Hugging Face cache
uv run python -m engine.server                           # 127.0.0.1:8000, 3 slots x 524,288 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

To serve Qwen3.8-Flash-Next instead of the 27B, start the server with its checkpoint:

```bash
uv run hf download nvidia/Qwen3.8-Flash-Next-NVFP4      # 132.7 GB
uv run python -m engine.server --model nvidia/Qwen3.8-Flash-Next-NVFP4   # 2 slots x 262,144 tokens, MTP speculation
```

If you cloned without `--recursive`, run `git submodule update --init` first. The CUDA kernels in `csrc/` compile when
the engine first uses them, in about a minute (`engine/kernels/__init__.py`). The engine keeps the build in
`build/torch_ext`.

The server has no authentication. It listens on 127.0.0.1 by default. Do not expose it to a network that you do not
trust.

One prepare step makes decode faster. For the 27B, it does not change the output of the model. For
Qwen3.8-Flash-Next, the copies change the bits of decode within the perplexity gate. Run the step once per model,
offline, with the server stopped. It gathers the files below into `~/.cache/colinfer/prepared/<snapshot>/` with a
`manifest.json`, and the engine loads that directory when it exists (`engine/runtime/prepared.py`):

```bash
uv run hf download Qwen/Qwen3.8-27B                      # the BF16 checkpoint (55.6 GB), for the INT copies; or pass --skip-int
uv run python tools/prepare.py                           # a few minutes on the GPU
uv run python tools/prepare.py --model nvidia/Qwen3.8-Flash-Next-NVFP4   # Qwen3.8-Flash-Next: about 30 s on the GPU
```

| File | Made by | Effect |
|---|---|---|
| INT6 / INT5 decode copies of the FP8 attention and GDN projections | `tools/int6_requant.py`, run by `prepare.py` | ~9.5% faster decode, perplexity within 0.25% |
| a fine-tuned MTP drafter, when one exists | `tools/drafter_data.py`, then `tools/train_drafter.py --extract` / `--train`; `prepare.py` takes the file | more accepted drafts per cycle |
| a low-rank draft head | `tools/lowrank_draft_head.py`, run by `prepare.py` on that drafter | the same drafts for ~1/5 of the draft head's bytes (k=7 cycle 89.8 → 86.0 ms) |
| Qwen3.8-Flash-Next: INT6 copies of the attention projections and of lm_head, an INT8 per-row copy of the hyper-connections' W_up | `tools/requant_qwen4exp.py`, run by `prepare.py --model nvidia/Qwen3.8-Flash-Next-NVFP4` (it reads the checkpoint's BF16 matrices) | decode step 46.6 → 39.4 ms (+18%), perplexity ratio 1.0014 against the exact decode path, +1.35 GB on the GPU |

The first start quantizes the drafter and keeps the result in `~/.cache/colinfer/drafter/quant-<key>.safetensors`
(~3 s off every later start).

`docs/server.md` describes the flags, the API and the behavior of the server. `deploy/colinfer.service` runs the server
under systemd. `docs/claude_code.md` tells you how to use the model from Claude Code, in the VS Code extension or the
CLI: a local gateway (`engine/server/gateway.py`, `deploy/colinfer-gateway.service`) puts Claude and Qwen in one model
picker. `docs/night_shift.md` runs a list of Claude Code tasks against the server at night, each on its own branch
(`tools/night_shift.py`).
`docs/review.md` reviews a finished feature with `/reviewing-features` in Claude Code: parallel finders for slop, security
and races, refuters, fixes and capped re-review rounds (`REVIEW.md`, `.claude/`, `tools/review.py`). The harness is
generic: `python3 tools/review.py export <repo>` copies it into another repository, and `REVIEW.md` tells it what colinfer is.

## How it works

`docs/architecture.md` gives the technical description. `docs/history/` holds the dated logs that explain each design
decision. In short, for the 27B (the last bullet gives what differs for Qwen3.8-Flash-Next):

- **Decode** reads the quantized weights once per step with a tensor-core *skinny GEMM* (`csrc/skinny.cu`). This kernel
  multiplies up to 16 rows for the cost of one row. Thus a verify of 8 speculative tokens costs about the same as one
  token. The 48 Gated DeltaNet layers (linear attention) and the 16 attention layers have their own kernels. One CUDA
  graph runs a full decode step or a full speculative cycle.
- **Speculation** uses the MTP head of the checkpoint as a drafter. Each cycle drafts k = 3 or 7 tokens, from the
  acceptance rate of each request. Acceptance is exact: the engine keeps a draft only if it is equal to the token that
  the target model itself gives. When a reply repeats earlier text, a cycle can verify up to 15 copied tokens instead.
- **Prefill** processes prompts in 2,048-token chunks. It uses CUTLASS NVFP4 and cuBLASLt FP8 tensor-core GEMMs, a CUDA
  kernel for the chunked delta rule, and FlashInfer or FP8 attention.
- **Serving** runs one request at a time and keeps three fixed slots with contiguous KV caches: the last three
  conversations stay cached. A ring of prefix checkpoints (GDN state snapshots) lets multi-turn chats and shared system
  prompts skip a second prefill. A cache on the NVMe keeps the KV and GDN snapshots of past conversations: a
  conversation that lost its slot, or any conversation after a restart, restores from disk (a 21k-token conversation:
  0.2 s to the first token instead of 7.2 s).
- **Constrained decoding** (llguidance): JSON schemas (`response_format`) and forced tool calls (`tool_choice`
  required / a named tool, also on the Anthropic API) mask the logits of each verify row, so speculation stays exact.
- **Committee mode**: a request that names the model `<name>-committee` is answered by three members of the model in
  different roles, one after another, and a reconcile pass writes the reply (`docs/server.md`). About 4x the time of
  one answer, for one review or design question at a time.
- **Images** run through the checkpoint's vision tower (a 27-block ViT, BF16, loaded on the first image) at prefill. Their tokens get multimodal
  RoPE positions, and the prefix caches tell images apart by a hash of their bytes, so a conversation that sends the
  same screenshot again does not encode it again.
- **Qwen3.8-Flash-Next** has its own decode model and prefill path on the same machinery. Its 512 experts per layer run
  on pair-slot MoE kernels at decode (`csrc/moe.cu`) and on a CUTLASS grouped NVFP4 GEMM at prefill
  (`csrc/moe_prefill.cu`). Each attention row attends to its 512 selected blocks of 4 tokens and its own tail
  (`csrc/qsa.cu`). A 4-stream residual (hyper-connections, `csrc/hc.cu`) replaces the norms. The 51 GB n-gram table
  stays on the disk, and the host reads its rows through a row cache. Decode runs the exact BF16 path of the
  checkpoint. When the prepared directory has them, requantized copies replace three classes of BF16 matrices.

## Qwen3.8-Flash-Next

`Qwen/Qwen3.8-Flash-Next` (`model_type` `qwen4_exp`) is a MoE model with 125B parameters, 6B of them active per
token. It has 48 layers (36 Gated DeltaNet, 12 with Qwen Sparse Attention), 512 experts per layer (top-10 and one shared
expert), a 4-stream residual (hyper-connections) and a 51 GB n-gram embedding table. The engine serves it from the
checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4`:

```bash
uv run python -m engine.server --model nvidia/Qwen3.8-Flash-Next-NVFP4
```

The family has its own defaults: 2 slots of 262,144 tokens (the native context), 16 checkpoints, 4,096-token prefill
chunks, a 1 GB host row cache for the n-gram table (`--ngram-cache-gb`) and a `--mem-cap-gb` of 101. Speculation is on:
the checkpoint's MTP block drafts 1 to 3 tokens per cycle, by the acceptance of each request, and suffix-match drafts
work as for the 27B. The model reads text only: an image in a request becomes a short text note. The chat template is
the 27B's, so `reasoning_effort` and the tool calls work the same way. `docs/server.md`, section
"Qwen3.8-Flash-Next", gives the flags and the behavior.

Measured on 2026-10-24 on this machine, at the defaults. The public vLLM build of this checkpoint decodes 16.9 tok/s
on the same machine.

| | colinfer |
|---|---|
| Plain decode, no speculation, through the server | **19 to 20 tok/s** (the graph step alone: 46.6 ms; the rest is the host's read of the n-gram rows) |
| Plain decode with the decode copies, graph step | **39.4 ms** (25.4 tok/s, +18%) |
| Decode with speculation, short context | code **39.1 tok/s** (1.84x, 3.23 tokens per cycle), prose **27.2 tok/s** (1.34x); chat completions on the server 28 to 39 tok/s |
| Prefill, 4,096-token chunks | **2760 to 2810 tok/s** at prefixes of 0 to 8k; about 2500 tok/s at 16k to 60k |
| Time to first token: short prompt / next turn (prefix reuse) / restore from disk | 0.1 to 0.3 s / **0.13 s** (1.75 s from scratch for 4,676 tokens) / **0.15 s** (7,398 tokens) |
| Passkey retrieval | 9 of 9 at 16k, 32k and 64k |
| Startup / memory at the ready line | 45 s / 91.3 GB allocated; `MemAvailable` falls by about 100 GB |

The speculation and server figures come from the exact path, before the decode copies existed. The prefill reads the
BF16 weights with or without the copies. The speed of the server with the copies is not measured yet.

How the engine proves the output exact:

- The plain-PyTorch reference (`engine/model/qwen4exp.py`) is bit-identical to the `Qwen4ExpForCausalLM` of
  transformers in fp32 on a tiny model (`tests/test_qwen4exp_small.py`): one-shot, split, stepwise, with a cache, with
  eos, with an image.
- The kernel path gives the reference's 32 greedy tokens on the three golden prompts (`tests/golden_qwen4exp.py`,
  `tests/data/qwen4exp_golden.json`), also with the prompts through the prefill. A divergence passes only at a near-tie
  of the reference (a bf16 margin below 0.2). With the decode copies, two golden prompts diverge at exact ties of the
  bf16 logits of the INT6 lm_head.
- Speculation gives the tokens of plain decode on every check, greedy and sampled (`tests/golden_qwen4exp_spec.py`;
  `tests/scheduler_check_qwen4exp.py`: 256 of 256 tokens on code and prose).
- Every decode kernel is row-invariant: a verify of 8 rows equals 8 single steps, bit for bit.
- The perplexity gate (0.5%, 8 WikiText windows of 2,048 tokens): the prefill gives 2.3438 against the reference's
  2.3335 (ratio 1.0044), and the decode path gives 2.3274 (ratio 0.9974). The decode copies give a ratio of 1.0014
  against the exact decode path. The last-token logits of prefill and decode differ by 10 to 15%, because the MoE
  prefill quantizes its activations to NVFP4, but both paths give the same tokens.

What the engine does not have for this family:

- images (the loader skips the vision tower; the 27B serves images as before)
- YaRN past 262,144 tokens
- the low-rank draft head of the 27B
- decode copies of the GDN projections and of the shared experts: their round-to-nearest copies break the perplexity
  gate, so they stay BF16 until a calibrated quantizer exists
- a prefetch of the n-gram rows between the prefill chunks

`docs/history/qwen4exp_plan.md` holds the design, the memory budget, the milestones and the dated status log.

## Repository

| Path | |
|---|---|
| `engine/model/qwen35.py` | the dense model (Qwen3.8-27B) in plain PyTorch: the correctness reference for its kernel tests |
| `engine/model/fast.py` | the decode path: kernel modules, decode state, the INT decode copies |
| `engine/model/prefill.py` | the prefill path |
| `engine/model/vision.py` | image input: the vision tower, the image tokens and their multimodal RoPE positions |
| `engine/spec/` | the MTP drafter and the speculative cycle (`mtp.py`), suffix-match drafts (`suffix.py`), exact acceptance for sampling (`accept.py`) |
| `engine/runtime/` | the request scheduler (slots, checkpoints, draft length), the conversation cache on disk (`sessions.py`), constrained decoding (`grammar.py`), the engine build (`build.py`), the plain decode graph (`decode.py`), the sampler, metrics |
| `engine/server/` | the HTTP server (`api.py`), its engine thread (`worker.py`), the OpenAI formats (`openai.py`), the Anthropic Messages API (`anthropic.py`), the chat template and output parsing (`chat.py`), images in requests (`images.py`) |
| `engine/weights/` | checkpoint loading and dequantization (`loader.py`), weight quantizers (`quantize.py`), the fake NVFP4 activation quantization that the kernel tests compare against (`quant_emul.py`) |
| `csrc/` | CUDA kernels and their Torch bindings (`bindings.cpp`). CUTLASS is a submodule in `csrc/third_party`. |
| `tests/` | unit tests (`pytest tests/`) and end-to-end checks (`golden.py`, `scheduler_check.py`, `perplexity.py`, ...) |
| `tools/` | offline artifacts (decode copies, drafter training, low-rank head), checkpoint utilities, the night shift (`night_shift.py`: Claude Code tasks against the server while the machine idles), the bookkeeping of the review harness (`review.py`) |
| `bench/` | decode / prefill / attention / GEMM / end-to-end benchmarks, nsys trace summaries |
| `docs/` | `architecture.md` (the current design), `server.md` (how to run the server), `claude_code.md` (how to connect Claude Code), `night_shift.md` (Claude Code tasks at night), `review.md` (the review harness), `checkpoints.md` (the checkpoints on this machine). `history/` holds the dated logs and measurements of the project, and `PLAN.md`, the original plan of 2026-09-30. |

The files of Qwen3.8-Flash-Next:

| Path | |
|---|---|
| `engine/model/family.py` | the model family of a checkpoint by its `model_type` (`qwen3_5`, `qwen4_exp`): the load, the state, the defaults, the speculation and the memory of each; `check_served` refuses a family that the engine cannot serve |
| `engine/model/qwen4exp.py` | Qwen3.8-Flash-Next in plain PyTorch: the correctness reference of the family |
| `engine/model/fast_qwen4exp.py` | the decode path: kernel modules, the decode state (`FastState4`), the decode copies (`attach_decode_copies4`) |
| `engine/model/prefill_qwen4exp.py` | the prefill path: the layer functions and the chunk loop |
| `engine/model/moe_kernels.py`, `qsa_kernels.py`, `hc_kernels.py` | the Python side of the new kernels (`KernelMoE`, the QSA wrappers, `KernelGatedResidual`), their torch references and their self-tests |
| `engine/spec/mtp4.py` | the MTP drafter of the family and its speculative cycle (`Mtp4`, `Mtp4State`, `Mtp4Cycle`) |
| `engine/weights/qwen4exp_loader.py` | `load_qwen4exp`: the checkpoint into the reference model, with the experts of each layer stacked |
| `engine/weights/ple.py` | the n-gram hash (`ngram_rows`), the row reader of the table on disk (`NgramTable`), and the feeder with its host row cache (`PleFeeder`) |
| `csrc/moe.cu`, `csrc/moe_prefill.cu` | the MoE decode kernels (route, gate-up, down, combine); the grouped NVFP4 GEMM of prefill (CUTLASS) with its gather, SwiGLU and combine kernels |
| `csrc/qsa.cu`, `csrc/qsa_prefill.cu` | Qwen Sparse Attention: the indexer prologue, the block keys, the scores, the top-k and the sparse decode attention; the indexer scores of a prefill chunk and the attention of each group of 4 rows over the union of their blocks |
| `csrc/hc.cu`, `csrc/hc_prefill.cu` | the hyper-connection kernels: the low-rank mix and the injection (the grouped norm is in `norm.cu`); the mix of a prefill chunk |
| `csrc/attn_tc.cuh`, `csrc/skinny_core.cuh` | shared with the 27B: the tensor-core core of the decode attention (dense and list modes); the dequantization helpers of the skinny GEMM |
| `csrc/ple_io.cpp` | the host row reader of the n-gram table (a thread pool of preads, `ops().ple_gather`) and the row cache of `PleFeeder` |
| `csrc/moe_bind.cpp`, `qsa_bind.cpp`, `hc_bind.cpp` | the Torch bindings of the new kernels; `bindings.cpp` calls `bind_moe`, `bind_qsa` and `bind_hc` |
| `tools/requant_qwen4exp.py` | the requantized decode copies, run by `prepare.py` (the formats and their perplexity ratios are in its docstring) |
| `tools/qwen4exp_reference.py` | greedy text from the reference on the real checkpoint (77.8 GB on the GPU: stop the server first) |
| `tools/verify_qwen4exp.py` | checks the checkpoint's headers, n-gram buffers, rows and scales without the GPU |
| `tools/ple_hotset.py` | the hot set of the n-gram table on real text, and the hit rate of a row cache |
| `bench/moe_bench.py`, `bench/prefill_chunk_bench.py` | the MoE decode stream rate, the full MoE layer and the MoE prefill per chunk; the prefill chunk by block on random weights |
| `tests/test_qwen4exp_small.py`, `test_qwen4exp_loader.py`, `test_ple.py`, `test_ple_io.py`, `test_ple_feeder.py`, `test_family.py` | the reference against transformers on a tiny model; the loader, the row readers and the feeder on synthetic checkpoints (`tests/qwen4exp_ckpt.py`); the family dispatch |
| `tests/test_moe.py`, `test_moe_prefill.py`, `test_qsa.py`, `test_qsa_prefill.py`, `test_hc.py` | the new kernels against their references; the bit identity of each row of the decode kernels for any row count |
| `tests/test_fast_qwen4exp.py`, `test_prefill_qwen4exp.py`, `test_mtp4.py`, `test_requant_qwen4exp.py` | the decode path, the prefill, the drafter and the decode copies on small random models |
| `tests/golden_qwen4exp.py`, `golden_qwen4exp_spec.py`, `scheduler_check_qwen4exp.py`, `passkey_qwen4exp.py` | the end-to-end checks on the real checkpoint: the golden tokens (`tests/data/qwen4exp_golden.json`), speculation against plain decode, the scheduler as the server builds it, long-context retrieval |

## Testing

```bash
tests/check.sh                                           # ruff, pytest, golden.py (both modes), scheduler_check.py, in order; stops at the first failure (~8 min)
tests/check.sh --quick                                   # ruff and pytest only (~1 min)
uv run pytest tests/ -q                                  # kernels against PyTorch references, bit-identity properties, the scheduler and the HTTP layer on stubs (~1 min)
uv run python tests/golden.py check                      # recorded tokens and logprobs, bit for bit (the default server)
uv run python tests/golden.py check --plain              # the same, without speculation
uv run python tests/scheduler_check.py                   # queueing, sampling, prefix reuse: outputs unchanged
uv run python tests/perplexity.py --engine prefill --ckpt nvidia/Qwen3.8-27B-NVFP4   # quality
```

The end-to-end checks of Qwen3.8-Flash-Next load about 80 GB. Stop the server first:

```bash
uv run python tests/scheduler_check_qwen4exp.py          # golden tokens, speculation = plain decode, the caches
uv run python tests/perplexity.py --engine qwen4exp --ckpt nvidia/Qwen3.8-Flash-Next-NVFP4   # quality through the prefill
```

## License

Apache-2.0 (`LICENSE`). Copyright 2026 czhou578.

Third-party material:

- CUTLASS (`csrc/third_party/cutlass`, a git submodule) is BSD-3-Clause, from NVIDIA.
- The model weights are not part of this repository. They have their own licenses: see the model cards of
  `Qwen/Qwen3.8-27B`, `nvidia/Qwen3.8-27B-NVFP4`, `Qwen/Qwen3.8-Flash-Next` and `nvidia/Qwen3.8-Flash-Next-NVFP4`.
- `tests/golden/prompts.json` holds the token ids of text excerpts from WikiText-103 (CC BY-SA 3.0), TinyStories and the
  Python source of open-source packages. These excerpts keep their original licenses. `engine/spec/draft_vocab.npy` is a
  list of token ids, ranked by their frequency in similar text.

## Acknowledgments

- The Qwen team (Qwen3.8-27B, Qwen3.8-Flash-Next), and NVIDIA (the NVFP4 checkpoints, the 27B's made with ModelOpt, and
  CUTLASS).
- FlashInfer, for the prefill attention and the sampling kernels that the engine uses.
- flash-linear-attention, the reference for the chunked Gated DeltaNet kernels.
- vLLM and SGLang, the baselines. The row semantics of the MTP drafter follow the Qwen3.5 MTP proposer of vLLM. The
  wiring of the Qwen3.8-Flash-Next drafter follows the source of vLLM.
- EAGLE-3 (the training-time test of the drafter fine-tune), DFlash, prompt-lookup and suffix decoding, Marlin and
  FlashDecoding, whose ideas the docs credit where they apply.

Issues and pull requests are welcome. This is a personal project, so there is no guarantee of support.

# colin-inference-engine

colinfer is a single-user inference engine for **Qwen3.8-27B** (`nvidia/Qwen3.8-27B-NVFP4`) on one **DGX Spark** (GB10,
sm_121). It serves the model through an OpenAI-compatible and an Anthropic-compatible HTTP API, so Claude Code can use
it directly, and it reads images (screenshots, diagrams) with the checkpoint's own vision tower. It is for one person,
and it runs one request at a time.

This is a research engine for one machine and one served model, not a general inference server. It has no paged KV
cache and no multi-GPU support. The kernels compile for sm_121a only. The design notes and the dated logs in `docs/`
record each measurement behind the design. Support for a second model, the MoE model Qwen3.8-Flash-Next, is in
progress on this branch, and the server does not serve it yet ([below](#qwen38-flash-next-in-progress)).

The memory bandwidth of this chip sets the decode speed. Each decode step reads all the weights from LPDDR5x memory at
~238 GB/s, so plain decode cannot go faster than ~14 tok/s. All the speed above this floor comes from speculative
decoding.

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
- About 105 GB of free unified memory (in units of 10^9 bytes, as `--mem-cap-gb`) while the server runs. Stop other
  GPU work first: an over-commit of the shared memory can power off the machine instead of an out-of-memory error. The
  server does not start with less.
- Up to 64 GB of disk for the conversation cache (`--session-cache-gb`, under `~/.cache/colinfer/sessions`).
- Optional, only for the work in progress: the checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` (132.7 GB). Only
  `tools/qwen4exp_reference.py` and `tools/verify_qwen4exp.py` read it. The server does not serve it.

## Quick start

```bash
git clone --recursive https://github.com/czhou578/colinfer.git && cd colinfer   # CUTLASS is a submodule
uv sync                                                  # Python 3.12, torch 2.14 (cu130), FlashInfer 0.7 (sm121a)
uv run hf download nvidia/Qwen3.8-27B-NVFP4             # into the Hugging Face cache
uv run python -m engine.server                           # 127.0.0.1:8000, 3 slots x 524,288 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

If you cloned without `--recursive`, run `git submodule update --init` first. The CUDA kernels in `csrc/` compile when
the engine first uses them, in about a minute (`engine/kernels/__init__.py`). The engine keeps the build in
`build/torch_ext`.

The server has no authentication. It listens on 127.0.0.1 by default. Do not expose it to a network that you do not
trust.

One prepare step makes decode faster. It does not change the output of the model. Run it once, offline, with the
server stopped. It gathers the files below into `~/.cache/colinfer/prepared/<snapshot>/` with a `manifest.json`, and
the engine loads that directory when it exists (`engine/runtime/prepared.py`):

```bash
uv run hf download Qwen/Qwen3.8-27B                      # the BF16 checkpoint (55.6 GB), for the INT copies; or pass --skip-int
uv run python tools/prepare.py                           # a few minutes on the GPU
```

| File | Made by | Effect |
|---|---|---|
| INT6 / INT5 decode copies of the FP8 attention and GDN projections | `tools/int6_requant.py`, run by `prepare.py` | ~9.5% faster decode, perplexity within 0.25% |
| a fine-tuned MTP drafter, when one exists | `tools/drafter_data.py`, then `tools/train_drafter.py --extract` / `--train`; `prepare.py` takes the file | more accepted drafts per cycle |
| a low-rank draft head | `tools/lowrank_draft_head.py`, run by `prepare.py` on that drafter | the same drafts for ~1/5 of the draft head's bytes (k=7 cycle 89.8 → 86.0 ms) |

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
decision. In short:

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

## Qwen3.8-Flash-Next (in progress)

The branch `moe` adds a second model family: `Qwen/Qwen3.8-Flash-Next` (`model_type` `qwen4_exp`), from the
checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4`. It is a MoE model with 125B parameters, 6B of them active per token. It
has 48 layers (36 Gated DeltaNet, 12 with Qwen Sparse Attention), 512 experts (top-10 and one shared expert), a
4-stream residual (hyper-connections) and a 51 GB n-gram embedding table. The server serves it with exact MTP
speculation: `python -m engine.server --model nvidia/Qwen3.8-Flash-Next-NVFP4` (`docs/server.md`, section
"Qwen3.8-Flash-Next").

What exists and is tested:

- **The reference model** in plain PyTorch (`engine/model/qwen4exp.py`). On a tiny model in fp32, it is bit-identical to
  the `Qwen4ExpForCausalLM` of transformers: one-shot, split, stepwise, with a cache, with eos, with an image.
- **The loader** (`engine/weights/qwen4exp_loader.py`) stacks the 512 experts of each layer at load time. On the real
  checkpoint, the reference loads 77.8 GB in 34 s and gives coherent greedy text at about 1 s per token
  (`tools/qwen4exp_reference.py`, the ground truth of the coming golden test).
- **The n-gram table** (51.2 GB FP8) stays on disk. `engine/weights/ple.py` hashes the ids and reads the rows on
  demand. `tools/verify_qwen4exp.py` checks the checkpoint without the GPU. `tools/ple_hotset.py` measures the hot set:
  an LRU row cache of 0.5 to 5 GB hits 51-68% of the rows on WikiText and 83-90% on TinyStories.
- **The kernels**, each tested against a PyTorch reference:
  - MoE decode (`csrc/moe.cu`) streams 208 GB/s at 10 distinct experts and 238 GB/s at 75. A full MoE layer takes
    196 µs at 1 row, 935 µs at 8 rows and 1635 µs at 16 rows.
  - MoE prefill (`csrc/moe_prefill.cu`, a CUTLASS grouped NVFP4 GEMM) takes 8.8 ms per layer for a 2,048-token chunk
    (11.3 ms at 4,096, 16.6 ms at 8,192): a MoE ceiling of 4.8k / 7.5k / 10.3k tok/s.
  - Sparse attention (`csrc/qsa.cu`): the block selection equals a stable sort and `torch.topk`, ties included. The
    attention over 2051 keys takes 9.3 µs at T = 1 and 72 µs at T = 8, at any context length.
  - Hyper-connections (`csrc/hc.cu`): a full gated residual takes 14 / 40 / 74 µs at 1 / 8 / 16 rows.
  - The GDN kernels take a sigmoid output gate, and the decode attention runs GQA 24/2. The 27B keeps its bits.
  - The host row reader of the n-gram table (`csrc/ple_io.cpp`) reads about 170k cold rows per second at 128 rows per
    call.
- **The startup self-test** covers 22 paths, the new kernels included. memcheck, racecheck (filtered to the engine's
  kernels) and synccheck are clean on the tests of the new kernels.

The serving path (milestone M4) works end to end. `tests/scheduler_check_qwen4exp.py` builds the engine as the server
does (2 slots x 262,144 tokens, 16 checkpoints, MTP speculation of 1 to 3 drafts), and every check passes:

- The greedy tokens of the three golden prompts equal the reference's, with the prompts through the prefill and with
  speculation. Speculation gives the tokens of plain decode on every check.
- A next turn restores its checkpoint: 0.13 s to the first token, against 1.75 s for the prefill of 4,676 tokens.
- A second conversation runs in the other slot, and the first one keeps its cache.
- A conversation that left both slots restores from the disk cache in 0.15 s (the drafter's rows included), with the
  output of a restore from memory.
- A prompt of 6,375 tokens (two prefill chunks) finds a code from its first chunk.

The server answers on `/v1/chat/completions` (streaming, `reasoning_effort`), `/v1/messages` and `/metrics`. It is ready
45 s after the start, and `MemAvailable` falls by 100 GB. Measured at short context: decode with speculation runs at
39.1 tok/s on code and 27.2 tok/s on prose, against 21.3 and 20.3 tok/s for plain decode (1.84x and 1.34x); a
streaming chat completion decodes at 28 to 32 tok/s. The public vLLM build decodes 16.9 tok/s on this machine.

What does not exist yet in the serving path: images, YaRN past 262,144 tokens, the requantized decode copies, and the
low-rank draft head of the 27B.

The plan expects about 49 tok/s of plain decode with requantized decode copies (under the perplexity gate). That
figure is an estimate from the bytes per token, not a measurement. `docs/history/qwen4exp_plan.md` holds the design, the memory
budget, the milestones and the dated status log.

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

The files of Qwen3.8-Flash-Next (in progress):

| Path | |
|---|---|
| `engine/model/family.py` | the model family of a checkpoint by its `model_type`; `check_served` refuses a family that the engine cannot serve yet |
| `engine/model/qwen4exp.py` | Qwen3.8-Flash-Next in plain PyTorch: the correctness reference of the family |
| `engine/model/moe_kernels.py`, `qsa_kernels.py`, `hc_kernels.py` | the Python side of the new kernels (`KernelMoE`, the QSA wrappers, `KernelGatedResidual`), their torch references and their self-tests |
| `engine/weights/qwen4exp_loader.py` | `load_qwen4exp`: the checkpoint into the reference model, with the experts of each layer stacked |
| `engine/weights/ple.py` | the n-gram hash (`ngram_rows`) and the row reader of the table on disk (`NgramTable`) |
| `csrc/moe.cu`, `csrc/moe_prefill.cu` | the MoE decode kernels (route, gate-up, down, combine); the grouped NVFP4 GEMM of prefill (CUTLASS) with its gather, SwiGLU and combine kernels |
| `csrc/qsa.cu` | Qwen Sparse Attention: the indexer prologue, the block keys, the scores, the top-k and the sparse decode attention |
| `csrc/hc.cu` | the hyper-connection kernels: the low-rank mix and the injection (the grouped norm is in `norm.cu`) |
| `csrc/attn_tc.cuh`, `csrc/skinny_core.cuh` | shared with the 27B: the tensor-core core of the decode attention (dense and list modes); the dequantization helpers of the skinny GEMM |
| `csrc/ple_io.cpp` | the host row reader of the n-gram table: a thread pool of preads (`ops().ple_gather`) |
| `csrc/moe_bind.cpp`, `qsa_bind.cpp`, `hc_bind.cpp` | the Torch bindings of the new kernels; `bindings.cpp` calls `bind_moe`, `bind_qsa` and `bind_hc` |
| `tools/qwen4exp_reference.py` | greedy text from the reference on the real checkpoint (77.8 GB on the GPU: stop the server first) |
| `tools/verify_qwen4exp.py` | checks the checkpoint's headers, n-gram buffers, rows and scales without the GPU |
| `tools/ple_hotset.py` | the hot set of the n-gram table on real text, and the hit rate of a row cache |
| `bench/moe_bench.py` | the MoE decode stream rate, the full MoE layer, and the MoE prefill per chunk |
| `tests/test_qwen4exp_small.py`, `test_qwen4exp_loader.py`, `test_ple.py`, `test_ple_io.py`, `test_family.py` | the reference against transformers on a tiny model; the loader and the row readers on synthetic checkpoints (`tests/qwen4exp_ckpt.py`); the refusal |
| `tests/test_moe.py`, `test_moe_prefill.py`, `test_qsa.py`, `test_hc.py` | the new kernels against their references; the bit identity of each row of the decode kernels for any row count |

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

## License

Apache-2.0 (`LICENSE`). Copyright 2026 czhou578.

Third-party material:

- CUTLASS (`csrc/third_party/cutlass`, a git submodule) is BSD-3-Clause, from NVIDIA.
- The model weights are not part of this repository. They have their own licenses: see the model cards of
  `Qwen/Qwen3.8-27B` and `nvidia/Qwen3.8-27B-NVFP4`.
- `tests/golden/prompts.json` holds the token ids of text excerpts from WikiText-103 (CC BY-SA 3.0), TinyStories and the
  Python source of open-source packages. These excerpts keep their original licenses. `engine/spec/draft_vocab.npy` is a
  list of token ids, ranked by their frequency in similar text.

## Acknowledgments

- The Qwen team (Qwen3.8-27B), and NVIDIA (the NVFP4 checkpoint, made with ModelOpt, and CUTLASS).
- FlashInfer, for the prefill attention and the sampling kernels that the engine uses.
- flash-linear-attention, the reference for the chunked Gated DeltaNet kernels.
- vLLM and SGLang, the baselines. The row semantics of the MTP drafter follow the Qwen3.5 MTP proposer of vLLM.
- EAGLE-3 (the training-time test of the drafter fine-tune), DFlash, prompt-lookup and suffix decoding, Marlin and
  FlashDecoding, whose ideas the docs credit where they apply.

Issues and pull requests are welcome. This is a personal project, so there is no guarantee of support.

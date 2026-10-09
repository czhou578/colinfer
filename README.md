# colin-inference-engine

colinfer is a single-user inference engine for **Qwen3.8-27B** (`nvidia/Qwen3.8-27B-NVFP4`) on one **DGX Spark** (GB10,
sm_121). It serves the model through an OpenAI-compatible and an Anthropic-compatible HTTP API, so Claude Code can use
it directly. It is for one person, and it runs one request at a time.

This is a research engine for one machine and one model, not a general inference server. It has no paged KV cache, no
multi-GPU support and no other models. The kernels compile for sm_121a only. The design notes and the dated logs in
`docs/` record each measurement behind the design.

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
- About 65 GB of free unified memory while the server runs. Stop other GPU work first: an over-commit of the shared
  memory can power off the machine instead of an out-of-memory error.

## Quick start

```bash
git clone --recursive https://github.com/czhou578/colinfer.git && cd colinfer   # CUTLASS is a submodule
uv sync                                                  # Python 3.12, torch 2.14 (cu130), FlashInfer 0.7 (sm121a)
uv run hf download nvidia/Qwen3.8-27B-NVFP4             # into the Hugging Face cache
uv run python -m engine.server                           # 127.0.0.1:8000, 3 slots x 262,144 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

If you cloned without `--recursive`, run `git submodule update --init` first. The CUDA kernels in `csrc/` compile when
the engine first uses them, in about a minute (`engine/kernels/__init__.py`). The engine keeps the build in
`build/torch_ext`.

The server has no authentication. It listens on 127.0.0.1 by default. Do not expose it to a network that you do not
trust.

Three optional files make decode faster. They do not change the output of the model. You make each file once, offline,
and the engine uses it when it is present:

| File | Made by | Effect |
|---|---|---|
| `~/.cache/colinfer/requant/<snapshot>/attn_gdn_int{6_self_attn,5_linear_attn}.safetensors` | `tools/int6_requant.py --bits 6 --filter self_attn`, `--bits 5 --filter linear_attn` (needs the BF16 checkpoint `Qwen/Qwen3.8-27B`) | INT6 / INT5 decode copies of the FP8 attention and GDN projections: ~9.5% faster decode, perplexity within 0.25% |
| `~/.cache/colinfer/drafter/mtp_ft.safetensors` | `tools/drafter_data.py`, then `tools/train_drafter.py --extract` / `--train` | a fine-tuned MTP drafter: more accepted drafts per cycle |
| `~/.cache/colinfer/drafter/draft_head_pca.safetensors` | `tools/lowrank_draft_head.py` | a low-rank draft head: the same drafts for ~1/5 of the draft head's bytes (k=7 cycle 89.8 → 86.0 ms) |

`docs/server.md` describes the flags, the API and the behavior of the server. `deploy/colinfer.service` runs the server
under systemd. `docs/claude_code.md` tells you how to use the model from the Claude Code CLI.

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
  prompts skip a second prefill.

## Repository

| Path | |
|---|---|
| `engine/model/qwen35.py` | the model in plain PyTorch: the correctness reference for all kernel tests |
| `engine/model/fast.py` | the decode path: kernel modules, decode state, `DecodeGraph`, the INT decode copies |
| `engine/model/prefill.py` | the prefill path |
| `engine/spec/` | the MTP drafter and the speculative cycle (`mtp.py`), suffix-match drafts (`suffix.py`), exact acceptance for sampling (`accept.py`) |
| `engine/runtime/` | the request scheduler (slots, checkpoints, draft length), the engine build (`build.py`), the sampler, metrics |
| `engine/server/` | the HTTP server (`api.py`), the Anthropic Messages API (`anthropic.py`), the chat template and output parsing (`chat.py`) |
| `engine/weights/` | checkpoint loading and dequantization (`loader.py`), weight quantizers (`quantize.py`), the fake NVFP4 activation quantization that the kernel tests compare against (`quant_emul.py`) |
| `csrc/` | CUDA kernels and their Torch bindings (`bindings.cpp`). CUTLASS is a submodule in `csrc/third_party`. |
| `tests/` | unit tests (`pytest tests/`) and end-to-end checks (`golden.py`, `scheduler_check.py`, `perplexity.py`, ...) |
| `tools/` | offline artifacts (decode copies, drafter training, low-rank head) and checkpoint utilities |
| `bench/` | decode / prefill / attention / GEMM / end-to-end benchmarks, nsys trace summaries |
| `docs/` | `architecture.md` (the current design), `server.md` (how to run the server), `claude_code.md` (how to connect Claude Code), `checkpoints.md` (the checkpoints on this machine). `history/` holds the dated logs and measurements of the project. |
| `PLAN.md` | the original plan and targets |

## Testing

```bash
uv run pytest tests/ -q                                  # kernels against PyTorch references, bit-identity properties, the scheduler and the HTTP layer on stubs (~1 min)
uv run python tests/golden.py check                      # recorded tokens and logprobs, bit for bit (the default server)
uv run python tests/golden.py check --plain              # the same, without speculation
uv run python tests/scheduler_check.py                   # batching, sampling, prefix reuse: outputs unchanged
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

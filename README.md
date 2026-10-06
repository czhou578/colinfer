# colin-inference-engine

A single-user inference engine for **Qwen3.8-27B** (`nvidia/Qwen3.8-27B-NVFP4`) on one **DGX Spark** (GB10, sm_121),
served through an OpenAI-compatible HTTP API. It is built for one person making up to three concurrent requests, and for
the speed this chip can physically deliver: decode is bound by LPDDR5x bandwidth (~238 GB/s), so everything above the
~14 tok/s weight-streaming floor comes from speculative decoding.

| | colinfer | SGLang 0.5.21 | vLLM 0.25.1 |
|---|---|---|---|
| 40-request mix, greedy, 256 tokens, wall tok/s (`bench/request_mix_bench.py`) | **39.4** (code 57.7, prose 33.3, Q&A 38.5, structured 39.6) | 29.0 (with DFlash2) | not measured |
| Plain decode, no speculation | **13.5 tok/s** (74 ms / token at 8k context) | 12.3 tok/s | 12.3 tok/s |
| Prefill, 2k / 8k / 32k prompt | **0.54 s / 2.29 s / 10.9 s** (3.8k / 3.6k / 3.0k tok/s) | ≈1.3 s / ≈4.9 s / not measured (≈1.5k / ≈1.7k tok/s) | 0.81 s / 3.9 s / 18.0 s |

The SGLang and vLLM decode and prefill figures are the phase-0 baselines on the same checkpoint and machine
(`docs/history/baseline.md` section 4: FP8 KV, prefix cache off; decode on a short prose prompt; SGLang's prefill from its
own per-batch log). The SGLang mix figure is SGLang with the DFlash2 drafter on the identical 40 requests
(`docs/history/phase6_progress.md` section 16).

Outputs are **token-identical with speculation on or off**, greedy or seeded-sampled, at any batch width
(`tests/spec_check.py`, `tests/scheduler_check.py`). Every quantization choice beyond the checkpoint's own passes a
perplexity gate of ≤ 0.5% against the checkpoint (WikiText and Python code).

## Quick start

```bash
uv sync                                                  # Python 3.12, torch 2.14 (cu130), FlashInfer 0.7 (sm121a)
uv run python -m engine.server                           # 127.0.0.1:8000, 3 slots x 262,144 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

The CUDA kernels in `csrc/` compile on first use (`engine/kernels/__init__.py`, cached in `build/torch_ext`). The checkpoint
must be in the local Hugging Face cache. Three optional files make decoding faster without changing what the model says
(each is made once, offline, and used when present):

| File | Made by | Effect |
|---|---|---|
| `~/.cache/colinfer/requant/<snapshot>/attn_gdn_int{6_self_attn,5_linear_attn}.safetensors` | `tools/int6_requant.py --bits 6 --filter self_attn`, `--bits 5 --filter linear_attn` (needs the BF16 checkpoint `Qwen/Qwen3.8-27B`) | INT6 / INT5 decode copies of the FP8 attention / GDN projections: ~9.5% faster decode, perplexity within 0.25% |
| `~/.cache/colinfer/drafter/mtp_ft.safetensors` | `tools/drafter_data.py`, then `tools/train_drafter.py --extract` / `--train` | a fine-tuned MTP drafter: more accepted drafts per cycle |
| `~/.cache/colinfer/drafter/draft_head_pca.safetensors` | `tools/lowrank_draft_head.py` | a low-rank draft head: the same drafts for ~1/5 of the draft head's bytes (k=7 cycle 89.8 → 86.0 ms) |

`docs/server.md` covers the server's flags, API and behavior; `deploy/colinfer.service` runs it under systemd.

## How it works

`docs/architecture.md` is the technical description; `docs/history/` holds the dated logs behind each design choice. In short:

- **Decode** streams the quantized weights once per step with a tensor-core *skinny GEMM* (`csrc/skinny.cu`) that
  multiplies up to 16 rows for the price of one, so verifying 8 speculative tokens costs about what one token costs.
  The 48 Gated DeltaNet layers (linear attention) and 16 attention layers run on dedicated kernels; a whole decode step
  or speculative cycle is one CUDA graph.
- **Speculation** uses the checkpoint's MTP head as a drafter (k = 3 or 7 drafts per cycle, chosen from each request's
  acceptance), with exact acceptance: a draft is kept only if it equals the token the target itself emits.
- **Prefill** runs prompts in 2,048-token chunks on CUTLASS NVFP4 and cuBLASLt FP8 tensor-core GEMMs, a CUDA chunked
  delta-rule kernel and FlashInfer / FP8 attention.
- **Serving** keeps three fixed slots with contiguous KV caches and a ring of prefix checkpoints (GDN state snapshots), so
  multi-turn chats and shared system prompts are not prefilled twice.

## Repository

| Path | |
|---|---|
| `engine/model/qwen35.py` | the model in plain PyTorch: the correctness reference every kernel is tested against |
| `engine/model/fast.py` | the decode path: kernel modules, decode state, `DecodeGraph`, the INT decode copies |
| `engine/model/prefill.py` | the prefill path |
| `engine/spec/` | the MTP drafter and speculative cycle (`mtp.py`), sampling-exact acceptance (`accept.py`) |
| `engine/runtime/` | the request scheduler (slots, checkpoints, draft length), sampler, metrics, the reference generator |
| `engine/server/` | the OpenAI-compatible server (`api.py`), chat template and output parsing (`chat.py`) |
| `engine/weights/` | checkpoint loading and dequantization (`loader.py`), weight quantizers (`quantize.py`), numerics emulation |
| `csrc/` | CUDA kernels and their Torch bindings (`bindings.cpp`); CUTLASS is a submodule in `csrc/third_party` |
| `tests/` | unit tests (`pytest tests/`) and end-to-end checks (`spec_check.py`, `scheduler_check.py`, `perplexity.py`, ...) |
| `tools/` | offline artifacts (decode copies, drafter training, low-rank head) and checkpoint utilities |
| `bench/` | decode / prefill / attention / GEMM / end-to-end benchmarks, nsys trace summaries |
| `docs/` | `architecture.md` (current design), `server.md` (running the server), `checkpoints.md` (the checkpoints on this machine); `history/`: the project's dated logs and measurements |
| `PLAN.md` | the original plan and targets |

## Testing

```bash
uv run pytest tests/ -q                                  # kernels against PyTorch references, bit-identity properties (~1 min)
uv run python tests/spec_check.py --k 7                  # speculation on vs off, token-identical
uv run python tests/scheduler_check.py                   # batching, sampling, prefix reuse: outputs unchanged
uv run python tests/perplexity.py --engine prefill --ckpt nvidia/Qwen3.8-27B-NVFP4   # quality
```

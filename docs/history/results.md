# Results: Phase 5, daily driver (2026-10-04)

This page covers the engine as an OpenAI-compatible server for `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark (GB10). It compares the engine
with the Phase 0 public-stack baselines (`docs/history/baseline.md`) under the same harnesses. Usage is in `docs/server.md`.

**Measurement setup:**

- **Engine:** `python -m engine.server` with its defaults:
  - 3 slots × 262,144 tokens;
  - FP8 KV cache;
  - MTP speculation, k=3 (k=1 when three slots decode together);
  - thinking at the template default (on).
- **vLLM baseline:**
  - vLLM 0.25.1, from `models/qwen3.8_27b_nvidia_nvfp4.yml`;
  - FP8 KV cache;
  - prefix caching off.
- **SGLang baseline:** SGLang 0.5.21 with NEXTN (MTP) speculation, 3 steps, 4 draft tokens.
- **Harness:**
  - `~/Projects/model-benchmarks` (`core_runner.py`), YAMLs `models/colinfer_qwen3.8_27b{,_nospec}.yml`.
  - Run directories are under `results/colinfer-Qwen3.8-27B-NVFP4*`. llama-benchy and full tool-calling outputs are under `results/phase5/`.
- **No prefix reuse in benchmarks:** for every benchmark the engine ran with `--no-prefix-caching`, or llama-benchy's `--no-cache`. The baselines also ran without a prefix cache, so prefill numbers are raw.

## Targets (PLAN.md 2.5 and docs/history/baseline.md section 5)

| Target | Result | |
|---|---|---|
| Base decode, no speculation, ≥ 12.5 tok/s | **12.6-12.7 tok/s** (vLLM 12.3) | met |
| Speculative decode on a code/chat mix at T=0, ≥ 35 tok/s | code edit 41.1, JSON 41.4, code generation 36.5, prose 24.4; **mean 35.9** (`tests/scheduler_check.py`) | met (Phase 4 left this at 32.4) |
| Prefill, 2k prompt, ≥ 2,500 tok/s (TTFT ≤ 0.8 s) | **3,130 tok/s**, TTFT 0.77 s (llama-benchy); 2,690 / 0.78 s in the harness | met |
| Prefill, 32k prompt, ≥ 2,000 tok/s (TTFT ≤ 16 s) | **2,390 tok/s**, TTFT 13.7 s (vLLM 1,800 / 18.0 s) | met |
| Multi-turn TTFT with a prefix hit ≤ 150 ms + new tokens | **0.11-0.12 s** for 8k-19k-token conversations (`server_check`, `scheduler_check`) | met |
| Context 262k by Phase 5 | 3 slots × 262,144; passkey retrieved from a **260,649-token** prompt through the HTTP API (TTFT 272 s) | met |
| 3 concurrent requests: per-request decode ≥ 0.8× single | Spec: 17.4 per request vs 35.8 alone (0.49×), but the aggregate is 48.8 tok/s. Plain decode: 10.5 vs 12.6 (0.83×) | not met with speculation (see "Not done") |
| Tool-call quality within noise of vLLM on the same checkpoint | full 85-task suite: **44 vs 46** at the harness's 256-token limit; **53 vs 53** at 2048 tokens (6 tasks differ, 3 each way) | met |
| Startup ≤ 60 s (≤ 30 s in docs/history/baseline.md) | **21 s** to serving with a warm page cache, **31 s** cold (checkpoint evicted from the page cache); vLLM: 140 s | met (cold: 1 s over the 30 s stretch) |
| Memory cap enforced | 57.4 GB allocated at startup, all of it preallocated; torch allocator capped at 80 GB (`--mem-cap-gb`) | met |

## Harness: engine vs Phase 0 baselines

### Prefill: TTFT median (s) / prefill tok/s, single request, no prefix reuse

| Prompt tokens | Engine | vLLM |
|---|---|---|
| 512 | 0.212 / 2,655 | 0.250 / 2,150 |
| 2,048 | 0.778 / 2,694 | 0.809 / 2,533 |
| 8,192 | 2.930 / 2,812 | 3.904 / 2,040 |
| 16,384 | 6.170 / 2,664 | 8.146 / 2,010 |
| 32,768 | 13.72 / 2,391 | 18.03 / 1,802 |
| 65,536 | 33.38 / 1,970 | 43.32 / 1,515 |

The SGLang harness runs have no TTFT figures (its stream gives no first-token timing; `docs/history/baseline.md` section 4).

### Decode: average tok/s, single stream, the harness's prose prompt (thinking on, greedy)

| Output tokens | Engine, MTP | Engine, no spec | vLLM | vLLM + MTP | SGLang + MTP |
|---|---|---|---|---|---|
| 512 | **27.1** | 12.6 | 12.4 | 24.8 | 24.8 |
| 1,024 | **25.4** | 12.7 | 12.3 | 22.2 | 22.2 |
| 2,048 | **26.5** | 12.7 | 12.3 | 21.7 | 22.3 |

### Concurrency: aggregate tok/s (256-token greedy outputs, 8 requests per level)

| Streams | Engine, MTP | Engine, no spec | vLLM | SGLang + MTP |
|---|---|---|---|---|
| 1 | **41.3** | 12.6 | 12.2 | 34.1 |
| 2 | **49.1** | 24.1 | 23.3 | 44.1 |
| 3 | 46.6 | 31.5 | 30.7 | **70.5** |
| 4 | 53.2 | 31.3 | 45.0 | **76.3** |

The engine has 3 slots, so a 4th request waits for one. SGLang wins at 3-4 streams because it verifies 4 drafts × 4 requests (16 rows) on tensor cores for about the cost of one row. The engine's verify step uses the CUDA-core weight-streaming GEMV, which becomes compute-bound past about 4 rows (+45% at 8 rows). The fix is the skinny tensor-core GEMM below.

### llama-benchy (pp 2048, tg 32/128/512 with `--exact-tg`, depth 0/4k/16k, 3 runs, no prefix cache)

| Test | Engine (MTP) t/s | vLLM t/s | Engine e2e TTFT (ms) | vLLM e2e TTFT (ms) |
|---|---|---|---|---|
| pp2048 | 3,107-3,264 | 2,444-2,955 | 741-770 | 809-971 |
| pp2048 @ d4096 | 2,945-2,960 | 2,130-2,176 | 2,187-2,197 | 2,940-3,002 |
| pp2048 @ d16384 | 2,655-2,673 | 2,021-2,023 | 7,006-7,054 | 9,225-9,236 |
| tg32 / tg128 / tg512 | 31.0 / 26.9 / 25.1 | 12.5 / 12.5 / 12.5 | | |
| tg @ d4096 | 25.6 / 27.2 / 25.8 | 12.4 / 12.4 / 12.4 | | |
| tg @ d16384 | 26.4 / 25.3 / 23.5 | 12.2 / 12.2 / 12.2 | | |

With `--exact-tg` (`ignore_eos`), generation continues past the end of the answer into text the drafter predicts less well. Ordinary replies run 30-40 tok/s.

## Quality: the 85-task tool-calling suite (`benchmarks/tool_calling.py --task-set full`, greedy, thinking on)

| max_tokens | Engine | vLLM |
|---|---|---|
| 256 (the harness default) | 44 / 85 | 46 / 85 |
| 2048 | **53 / 85** | **53 / 85** |

At 2048 tokens the two stacks score the same. Six tasks differ, three in each direction, which is the noise of two greedy
paths through slightly different numerics.

At 256 tokens, 28 of the engine's 85 requests run out of tokens while still thinking at xhigh effort. At that limit the score mostly measures whether a reply fits. Four tasks differ between the two stacks, and all four are differences in what the model generated, not parsing errors:

- The engine (W4A16: 4-bit weights, BF16 activations) and vLLM (W4A4: 4-bit activations too) follow different greedy paths, with differently long reasoning.
- On two schema tasks the engine's reasoning ran longer and hit the limit inside the tool call.
- On one arithmetic task the engine answered directly instead of calling the calculator.

Replayed with 2048 tokens, both schema tasks pass. The engine's numerics are the more accurate ones for this checkpoint: W4A16 perplexity is 1.75% lower than W4A4 (Phase 1, `docs/history/phase1_results.md`).

One parser difference was found and fixed. vLLM's parser keeps a tool call cut off by the token limit, with its completed parameters, and our server now does the same. Unlike vLLM, the engine still reports `finish_reason: "length"` for it, so a client can see the call was truncated.

## End-to-end checks

- **`tests/scheduler_check.py`** (scheduler, real model):
  - Greedy output from the MTP scheduler is token-identical to plain decode in all of these cases:
    - each request alone;
    - four requests at once (3 slots plus 1 queued);
    - next to a sampled request and a 5000-token chunked prefill.
  - A seeded request at T=0.8 emits exactly what plain decode samples, alone and next to one or two other requests (with three requests decoding, the cycle drafts only k=1). `tests/spec_check.py --drafter mtp --temperature 0.7` also passes with identical output on all four prompts, at 41 / 41 / 36 / 23 tok/s.
  - Turn 2 of a conversation restores the end of turn 1: 7,956 of 7,978 prompt tokens reused, TTFT 0.11 s.
  - A second conversation with the same 8k-token system prompt, started while the first is still decoding, copies that prefix's KV into another slot: TTFT 0.115 s instead of 2.87 s, with the same greedy output.
- **`tests/server_check.py`** (HTTP):
  - tool calls, non-streamed and streamed, including two parallel calls;
  - a tool-result round trip;
  - seeded completions with logprobs;
  - a 4th concurrent request queued;
  - a client disconnect freeing its slot;
  - 400 errors;
  - multi-turn reuse (19,385 of 19,405 tokens cached, TTFT 0.12 s vs 7.3 s);
  - `/metrics`.
- **`tests/passkey.py --url`:** a pass key at 90% depth of a 260,649-token prompt, through the HTTP API.
- **`pytest tests`:** 128 tests pass. New:
  - `test_chat.py` (templating, detokenizer, reasoning / tool parsing, stop strings);
  - position-keyed sampling tests (inverse-CDF marginal; drawing k+1 verify rows at once gives the same tokens as k+1 single-row steps);
  - a test of the seeded-uniform kernel.

## Startup and memory

`python -m engine.server` on a warm page cache:

| Step | Time |
|---|---|
| Kernels (cached extension) | 0.0 s |
| Weights + MTP head | 5.3-5.9 s |
| Self-test, 12 speculative graphs, checkpoint ring | 9.5-9.7 s |
| Warm-up | 4.8 s |
| Serving | 21.2-21.8 s from the command |
| Cold page cache (weights 15.3 s from NVMe) | 31.1 s |

Memory: 57.4 GB allocated once at startup, under an 80 GB allocator cap:

| Item | Size |
|---|---|
| Weights | 21 GB |
| KV cache, 3 × 262k tokens × 32 KB | 25.8 GB |
| Drafter KV | 3.2 GB |
| Checkpoint ring, 32 × 154 MB | 4.9 GB |
| Graphs and workspace | the rest |

## What was built (Phase 5)

- **Scheduler (`engine/runtime/scheduler.py`):** replaces the Phase 3 `Engine`.
  - **Batched speculation.** One CUDA graph per contiguous slot range and per greedy/sampled mode (12 graphs) runs a whole MTP cycle for every decoding slot: verify, acceptance, commit and drafting. Draft length depends on how many slots decode (`k_for`) so verify fits one GEMV pass.
  - **Batch-invariant greedy output.** Every kernel computes a slot's rows the same way at any width, which is what makes multi-slot greedy output identical to single-slot.
  - **Stop tokens on the GPU.** The cycle cuts the accepted length after an accepted stop token, so a slot's history ends exactly at the reply's last token.
  - **Prefix reuse:**
    - A preallocated checkpoint ring, with checkpoints at prompt end, generation end, every 8192 tokens, and two chat-message boundaries (end of the system message, start of the last message).
    - Cross-slot KV-prefix copy when the matching checkpoint's slot is busy.
    - `cache_salt` partitioning.
  - **Per-token output:** stop tokens, `min_tokens`, logprobs.
- **Drafter (`engine/spec/mtp.py`):** `MtpCycle` generalized to B slots, greedy and sampled slots mixed in one graph.
- **Position-keyed sampling (`engine/spec/accept.py`, `engine/runtime/sampler.py`, `csrc/sampling.cu`).** This replaces Phase 4's rejection sampling.
  - Each token is drawn by inverse CDF from the processed distribution, using one uniform from Philox(seed, token position).
  - Verification draws the target's sample at every row and accepts the drafts that equal it.
  - For a deterministic drafter this has the same acceptance rate as rejection sampling, p(draft).
  - The output is exactly the plain-sampling output, so seeded requests are reproducible across batch width, draft length and `--spec`.
- **Server (`engine/server/`):**
  - FastAPI over uvicorn, with one engine thread that owns the GPU.
  - Detokenization and parsing happen in the engine thread, so a stop string ends a request in the same step.
  - Chat template with tools and thinking; reasoning / tool-call parsing at the token level.
  - SSE streaming; usage and timings; vLLM-style `request_metrics`.
  - Prometheus `/metrics`, `/v1/status`.
  - A client disconnect aborts the request.
- **Deployment:** systemd user unit `deploy/colinfer.service`; a CUDA failure exits the process and systemd restarts it.

## Not done / next

- **Skinny tensor-core GEMM** (NVFP4 / FP8 weights, BF16 activations, `mma.sync`, M ≤ 16). It would let three slots verify k=3 for about the price of one row, and k=5-7 on code at one slot. It is the main remaining gap: 3-4 stream aggregate throughput (47-53 vs SGLang's 70-76) and the 0.8× per-request concurrency target.
- **Not implemented, accepted and ignored:** repetition / presence / frequency penalties, and `response_format` (JSON-schema-constrained decoding).
- **Cold start (31 s) is dominated by reading 22 GB of checkpoint at about 1.5 GB/s** (15.3 s). That is under the 60 s target. A repacked single-file weight cache read with O_DIRECT could take it under 25 s, but at this startup time the plan's repack cache isn't needed.
- **The user-side exit criterion** ("you switch your own tooling to it") is the user's step: `systemctl --user enable --now colinfer` (docs/server.md), then point clients at `http://127.0.0.1:8000/v1`.

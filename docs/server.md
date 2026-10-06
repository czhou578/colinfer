# Running the server

An OpenAI-compatible server for `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark (PLAN.md Phase 5).

```bash
uv run python -m engine.server                     # 127.0.0.1:8000, 3 slots x 262,144 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

Startup takes about 21 s with a warm page cache: 6 s to load weights, 10 s for the self-test and graph capture, and
5 s for warm-up. The process then holds about 57 GB of the 121 GB unified memory until it exits.

| Flag | Default | |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | |
| `--model` | `nvidia/Qwen3.8-27B-NVFP4` | An HF repo id in the local cache, or a checkpoint directory. |
| `--served-model-name` | the `--model` value | The id that `/v1/models` reports. Requests can name any model. |
| `--slots` | 3 | Concurrent requests. Further requests wait in a FIFO queue. |
| `--max-seq-len` | 262144 | Tokens per slot (prompt plus output). KV costs 32 KB per token per slot (18 KB with `--kv fp4`). |
| `--kv` | `fp8` | KV cache format. `fp4` stores e2m1 values plus e4m3 scales per 16 dims: 0.56× the memory (3 × 262k: 14.5 GB instead of 25.8 GB), perplexity +0.2-0.3%. At 128k: plain decode 89 vs 97 ms per step, a width-3 k=3 cycle 153 vs 175 ms. |
| `--spec` | `mtp` | `none` turns off speculation and runs plain one-token decode. |
| `--k` | 7 | Longest MTP draft. Each cycle picks k=3 or 7 from measured acceptance (see below). |
| `--drafter-weights` | `auto` | MTP head weights: `auto` uses `~/.cache/colinfer/drafter/mtp_ft.safetensors` (`tools/train_drafter.py`) when it exists; `none` the checkpoint's; or a path. Drafts change speed, never outputs. |
| `--decode-weights` | `auto` | `auto`: `int` when its two files exist in `~/.cache/colinfer/requant/<snapshot>/` (`tools/int6_requant.py --bits 6 --filter self_attn` and `--bits 5 --filter linear_attn`), else `awq-attn` when `attn_gdn_nvfp4_awq_attn.safetensors` exists, else `checkpoint`. `int`: INT6 attention + INT5 GDN projections, ~9.5% faster decode, perplexity within 0.25%, +5.2 GB GPU memory. `awq-attn`: decode the 64 attention projections from AWQ NVFP4, ~3% faster, perplexity within 0.5%. `requant`: attention and GDN projections from NVFP4, about 18% faster, but code perplexity +1.5%. `checkpoint`: the FP8 originals (`docs/phase6_progress.md` section 3). |
| `--checkpoints` | 32 | Prefix-checkpoint ring, 154 MB each, allocated at startup. |
| `--no-prefix-caching` | off | Never reuse a prompt prefix (same as `--checkpoints 0`). For raw-prefill benchmarks, like vLLM's `--no-enable-prefix-caching`. |
| `--mem-cap-gb` | 80 | Hard cap on the torch allocator. Exceeding it raises an error, and the process exits and restarts. |
| `--thinking` | `auto` | Default `enable_thinking`. `auto` uses the template default, which is on. |

## API

`POST /v1/chat/completions` and `POST /v1/completions` support these request fields:

- **Sampling:**
  - `temperature`, `top_p`, `top_k`, `min_p`.
  - Unset fields take the checkpoint's `generation_config` defaults: temperature 1.0, top_k 20, top_p 0.95.
  - `temperature: 0` gives greedy decoding.
- **Length:** `max_tokens` / `max_completion_tokens` (default: the remaining context), `min_tokens`, `ignore_eos`.
- **Stopping:** `stop` (strings; matched on the decoded text and excluded from the output) and `stop_token_ids`.
- **Seeds:** with a `seed`, the tokens depend only on the seed and the prompt. That holds with speculation on or off and whatever else runs alongside, because each position's draw is keyed by (seed, position). Without a seed, each request gets a random one.
- **Logprobs:**
  - Chat: `logprobs` with `top_logprobs` (at most 20).
  - Completions: `logprobs: n`.
  - Values come from the raw model distribution, before temperature is applied.
- **Streaming:** `stream` with `stream_options.include_usage`.
- **Prompt rendering:**
  - `tools`, `tool_choice` (`none` leaves the tools out of the prompt).
  - `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, ...), plus top-level `enable_thinking` and `reasoning_effort` (`none`, `low`, `medium`, `high`).
- **Prefix reuse:** `cache_salt` (a checkpoint is reused only by requests with the same salt) and `cache_prompt: false` (no reuse for this request).
- **Extras:**
  - `return_token_ids` (vLLM extension, used by llama-benchy).
  - An `extra_body` object sent literally inside the JSON is merged in.

Accepted but ignored: `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `response_format`, `user`.
Rejected with a 400: `n > 1`, `echo`, and a prompt longer than the slot.

Responses:

- **Reasoning:** reasoning text goes in `reasoning_content`, with a copy in `reasoning`; the final answer goes in `content`.
- **Tool calls:**
  - Qwen's XML tool calls become OpenAI `tool_calls`.
  - Argument values are typed by the request's JSON schema.
  - `finish_reason` is `tool_calls`.
  - In a stream, each call arrives in one delta once it is complete.
- **Usage:** `usage` includes `prompt_tokens_details.cached_tokens`, the prompt tokens restored from a checkpoint.
- **Timings:** a `timings` object reports queue time, TTFT, prefill tokens and speed, and decode time and speed.
- **Streaming metrics:** the first content chunk of a stream carries vLLM-style `request_metrics`.

Other endpoints:

- `GET /v1/models`, whose result includes `max_model_len` and the model config.
- `GET /health`.
- `GET /metrics` (Prometheus), with:
  - step time by kind (prefill chunk, or decode cycle by batch width);
  - TTFT and queue time;
  - tokens per speculative cycle;
  - draft and accepted counters;
  - prompt, cached and generated token counters;
  - queue depth and busy slots;
  - allocator memory.
- `GET /v1/status`, showing each slot's phase and length, the checkpoints and the startup timings.

## Behavior worth knowing

- **Prefill past 16k tokens of context uses FP8 attention** (Q K^T on FP8 tensor cores, `csrc/attn_prefill.cu`): 6%
  faster at 64k, 11% at 128k, perplexity +0.25-0.3% for those chunks. `COLINFER_ATTN_FP8_PREFILL=0` turns it off,
  `COLINFER_ATTN_FP8_MIN_CTX` moves the threshold (`docs/phase6_progress.md` section 8).
- **Speculation is always on unless the server runs with `--spec none`.** Its output is token-identical to plain decode, greedy or sampled (with the same seed).
- **Draft length adapts.** (The rest of this item describes Phase 5; with the Phase 6 tensor-core verify kernel, a
  cycle verifies up to 16 rows in one weight pass and picks k=3 or 7 per cycle from each request's acceptance.)
  - A cycle verifies (k+1) rows per slot, and a weight-streaming pass handles 8 rows.
  - One or two decoding slots use k=3: 4 and 8 rows.
  - Three slots use k=1, which is 6 rows; with k=3 they would need 12 rows and a second pass over every weight.
- **Drafting stops early when it is unlikely to pay.** Once the product of the drafter's probabilities of a cycle's
  drafts falls below 0.1 for every decoding slot, the cycle's remaining draft steps skip their weight GEMMs (~0.35 ms
  instead of ~1.8 ms a step); verify rejects their junk drafts, so outputs do not change. `COLINFER_DRAFT_STOP` sets
  the threshold (0 turns it off; `docs/phase6_progress.md` section 18).
- **Low-rank draft head.** With `~/.cache/colinfer/drafter/draft_head_pca.safetensors` (`tools/lowrank_draft_head.py`),
  each draft step scores a rank-1024 approximation of the draft lm head and rescores its top 256 candidates exactly:
  the same drafts as the full head for ~1/5 of its bytes (k=7 cycle 89.8 → 86.0 ms). `COLINFER_DRAFT_LOWRANK=0` turns
  it off (`docs/phase6_progress.md` section 19).
- **Prefix checkpoints.** A request restores the longest checkpoint that is a prefix of its prompt. Checkpoints are taken:
  - at the end of each prompt and each reply;
  - at the end of the first message (a shared system prompt);
  - before the last message;
  - every 8192 prompt tokens.

  If the checkpoint's slot is busy, its KV prefix is copied into a free slot, so parallel agents that share a long system prompt each pay for it once.
- **Long prompts don't freeze other requests.** A long prompt prefills 2048 tokens per engine step, and the other slots decode between chunks, so they slow down but don't stop.
- **Failures restart the process.** A failed CUDA call fails every in-flight request with a 500 and exits the process, and systemd restarts it.

## systemd

`deploy/colinfer.service` is a user unit:

```bash
systemctl --user link ~/Projects/colin-inference-engine/deploy/colinfer.service
systemctl --user enable --now colinfer
journalctl --user -u colinfer -f
```

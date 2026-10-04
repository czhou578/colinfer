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
| `--max-seq-len` | 262144 | Tokens per slot (prompt plus output). KV costs 32 KB per token per slot. |
| `--spec` | `mtp` | `none` turns off speculation and runs plain one-token decode. |
| `--k` | 3 | MTP draft length. Three slots decoding together use k=1 (see below). |
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

- **Speculation is always on unless the server runs with `--spec none`.** Its output is token-identical to plain decode, greedy or sampled (with the same seed).
- **Draft length depends on how many slots are decoding.**
  - A cycle verifies (k+1) rows per slot, and a weight-streaming pass handles 8 rows.
  - One or two decoding slots use k=3: 4 and 8 rows.
  - Three slots use k=1, which is 6 rows; with k=3 they would need 12 rows and a second pass over every weight.
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

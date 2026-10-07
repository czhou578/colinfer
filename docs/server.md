# Running the server

The engine includes an OpenAI-compatible server for `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark. `docs/architecture.md`
describes the design.

```bash
uv run python -m engine.server                     # 127.0.0.1:8000, 3 slots x 262,144 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

With a warm page cache, startup takes about 25 s:

- 9 s to load the weights and build the copies for the drafter
- 12 s for the self-test and the graph capture
- 3 s for the warm-up

Then the process holds about 61 GB of the 121 GB unified memory until it exits.

| Flag | Default | |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | |
| `--model` | `nvidia/Qwen3.8-27B-NVFP4` | An HF repo id in the local cache, or a checkpoint directory. |
| `--served-model-name` | the `--model` value | The id that `/v1/models` reports. Requests can name any model. |
| `--slots` | 3 | Concurrent requests. Other requests wait in a FIFO queue. |
| `--max-seq-len` | 262144 | Tokens per slot (prompt plus output). The fp8 KV cache costs 32 KB per token per slot. |
| `--spec` | `mtp` | `none` turns off speculation and runs plain one-token decode. |
| `--k` | 7 | The longest MTP draft. Each cycle picks k=3 or 7 from the measured acceptance (see below). |
| `--suffix-drafts` | 8 | Suffix-match drafts of at least N matched tokens. 0 turns them off. See below. |
| `--drafter-weights` | `auto` | MTP head weights. `auto` uses `~/.cache/colinfer/drafter/mtp_ft.safetensors` (`tools/train_drafter.py`) when it exists. `none` uses the weights of the checkpoint. You can also give a path. Drafts change the speed, never the outputs. |
| `--decode-weights` | `int` | `int`: decode the attention / GDN projections from INT6 / INT5 copies when both files exist in `~/.cache/colinfer/requant/<snapshot>/` (`tools/int6_requant.py --bits 6 --filter self_attn` and `--bits 5 --filter linear_attn`). This gives ~9.5% faster decode, perplexity within 0.25%, and uses +5.2 GB of GPU memory. Without the files, or with `checkpoint`, decode reads the FP8 weights. |
| `--checkpoints` | 32 | The prefix-checkpoint ring, 154 MB each. The server allocates it at startup. |
| `--no-prefix-caching` | off | Never reuse a prompt prefix (the same as `--checkpoints 0`). Use it for raw-prefill benchmarks, like `--no-enable-prefix-caching` in vLLM. |
| `--mem-cap-gb` | 80 | A hard cap on the torch allocator. Past the cap, the allocator raises an error, and the process exits and restarts. |
| `--thinking` | `auto` | The default `enable_thinking`. `auto` uses the template default, which is on. |

## API

`POST /v1/chat/completions` and `POST /v1/completions` support these request fields:

- **Sampling:**
  - `temperature`, `top_p`, `top_k`, `min_p`.
  - Fields that you do not set take the `generation_config` defaults of the checkpoint: temperature 1.0, top_k 20,
    top_p 0.95.
  - `temperature: 0` gives greedy decoding.
- **Length:** `max_tokens` / `max_completion_tokens` (default: the remaining context), `min_tokens`, `ignore_eos`.
- **Stop:** `stop` and `stop_token_ids`. The server matches the `stop` strings on the decoded text and removes them from
  the output.
- **Seeds:** with a `seed`, the tokens depend only on the seed and the prompt. This is true with speculation on or off,
  and with any other load, because the server keys the draw at each position by (seed, position). Without a seed, each
  request gets a random seed.
- **Logprobs:**
  - Chat: `logprobs` with `top_logprobs` (at most 20).
  - Completions: `logprobs: n`.
  - The values come from the raw model distribution, before the server applies the temperature.
- **Streaming:** `stream` with `stream_options.include_usage`.
- **Prompt rendering:**
  - `tools`, `tool_choice` (`none` leaves the tools out of the prompt).
  - `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, ...), plus top-level `enable_thinking` and
    `reasoning_effort` (`none`, `low`, `medium`, `high`).
- **Prefix reuse:** `cache_salt` and `cache_prompt: false`. Only requests with the same salt reuse a checkpoint.
  `cache_prompt: false` stops reuse for this request.
- **Extras:**
  - `return_token_ids` (a vLLM extension that llama-benchy uses).
  - The server merges an `extra_body` object that a client sends literally inside the JSON.

The server accepts these fields but ignores them: `presence_penalty`, `frequency_penalty`, `repetition_penalty`,
`response_format`, `user`. It rejects these with a 400: `n > 1`, `echo`, and a prompt longer than the slot.

Responses:

- **Reasoning:** the reasoning text goes in `reasoning_content`, with a copy in `reasoning`. The final answer goes in
  `content`.
- **Tool calls:**
  - Qwen's XML tool calls become OpenAI `tool_calls`.
  - The JSON schema of the request sets the types of the argument values.
  - `finish_reason` is `tool_calls`.
  - In a stream, each call arrives in one delta when it is complete.
- **Usage:** `usage` includes `prompt_tokens_details.cached_tokens`, the prompt tokens that the server restored from a
  checkpoint.
- **Timings:** a `timings` object gives the queue time, the TTFT, the prefill tokens and speed, and the decode time and
  speed.
- **Streaming metrics:** the first content chunk of a stream carries vLLM-style `request_metrics`.

Other endpoints:

- `GET /v1/models`. The result includes `max_model_len` and the model config.
- `GET /health`.
- `GET /metrics` (Prometheus), with:
  - step time by kind (prefill chunk, or decode cycle by batch width)
  - TTFT and queue time
  - tokens per speculative cycle
  - draft and accepted counters
  - prompt, cached and generated token counters
  - queue depth and busy slots
  - allocator memory
- `GET /v1/status`. It shows the phase and length of each slot, the checkpoints and the startup timings.

## Behavior to know

- **Prefill past 16k tokens of context uses FP8 attention** (Q K^T on FP8 tensor cores, `csrc/attn_prefill.cu`). It is 6%
  faster at 64k and 11% faster at 128k, and it adds 0.25-0.3% to the perplexity of those chunks. See
  `ATTN_FP8_MIN_CTX` in `engine/model/prefill.py` and `docs/history/phase6_progress.md` section 8.
- **Speculation is always on, except when the server runs with `--spec none`.** The output is token-identical to plain
  decode, greedy or sampled (with the same seed). `--spec none` gives exactly the same tokens, only slower.
- **The draft length adapts.** A cycle verifies k+1 rows per slot in one weight pass of up to 16 rows. It picks k = 3 or 7
  for each cycle, for the most expected tokens per second, from the acceptance of each request. Code and structured
  output usually draft 7, and prose drafts 3. Three decoding slots use k = 3 (12 rows), because k = 7 needs a second
  weight pass.
- **The draft steps stop early when they are unlikely to help.** The product of the drafter probabilities of the drafts
  of a cycle can fall below 0.1 for all decoding slots. Then the remaining draft steps of the cycle skip their weight
  GEMMs (~0.35 ms instead of ~1.8 ms a step). The verify rejects their junk drafts, so the outputs do not change. See
  `DRAFT_STOP` in `engine/spec/mtp.py` and `docs/history/phase6_progress.md` section 18.
- **Low-rank draft head.** `tools/lowrank_draft_head.py` makes `~/.cache/colinfer/drafter/draft_head_pca.safetensors`.
  With this file, each draft step scores a rank-1024 approximation of the draft lm head and rescores its top 256
  candidates exactly. This gives the same drafts as the full head for ~1/5 of its bytes (k=7 cycle 89.8 → 86.0 ms).
  Without the file, the engine uses the full draft head (`docs/history/phase6_progress.md` section 19).
- **Suffix-match drafts (`--suffix-drafts N`, default 8, 0 = off).** The history of a request is its prompt and the reply
  so far. When the last N or more tokens of the history occurred earlier in it, the cycle drafts the tokens that
  followed that earlier occurrence (`engine/spec/suffix.py`). These drafts replace the MTP drafts.

  When the request decodes alone, a cycle can verify up to 15 of these tokens. 16 verify rows still fit one weight pass,
  and the cycle costs 85 ms instead of 80. With N = 8, code-editing replies that reproduce their input decode ~65% faster
  (85 → 140 tok/s). The 40-request mix does not change: 39.2 tok/s wall time with the drafts on or off. In that mix, code
  is slightly faster and Q&A is ~1% slower. The outputs do not change.
- **Prefix checkpoints.** A request restores the longest checkpoint that is a prefix of its prompt. The server takes a
  checkpoint:
  - at the end of each prompt and each reply
  - at the end of the first message (a shared system prompt)
  - before the last message
  - every 8192 prompt tokens

  If the slot of the checkpoint is busy, the server copies its KV prefix into a free slot. Thus parallel agents that
  share a long system prompt pay for it only once.
- **Long prompts do not stop other requests.** A long prompt prefills 2048 tokens per engine step, and the other slots
  decode between the chunks. The other requests become slower, but they do not stop.
- **Failures restart the process.** A failed CUDA call fails each in-flight request with a 500 and stops the process.
  Then systemd restarts it.

## systemd

`deploy/colinfer.service` is a user unit. Its paths assume that the clone is `~/Projects/colin-inference-engine`. If
your clone is elsewhere (a plain `git clone` creates `./colinfer`), change `WorkingDirectory` and `ExecStart` first.
Then, from the root of the clone:

```bash
systemctl --user link "$PWD/deploy/colinfer.service"
systemctl --user enable --now colinfer
journalctl --user -u colinfer -f
```

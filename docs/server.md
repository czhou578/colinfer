# Running the server

The engine includes a server for `nvidia/Qwen3.8-27B-NVFP4` on one DGX Spark. It has an OpenAI-compatible API and an
Anthropic-compatible API. `docs/architecture.md` describes the design.

```bash
uv run python -m engine.server                     # 127.0.0.1:8000, 3 slots x 524,288 tokens, MTP speculation
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

With a warm page cache, startup takes about 25 s:

- 9 s to load the weights and build the copies for the drafter
- 12 s for the self-test and the graph capture
- 3 s for the warm-up

Then the process holds about 88 GB of the 121 GB unified memory until it exits. Before the server creates the CUDA
context, it compares its estimate of this memory with `MemAvailable`. The estimate adds a reserve of 6 GB for the growth
after startup: the vision tower, a tokenize or an image decode, and the request bodies before the queue. At the defaults, the server needs about 101 GB. When
the estimate does not fit, the server logs the two figures and exits with status 3. The systemd unit
(`deploy/colinfer.service`) does not restart on this status. `engine/runtime/build.py` (`check_memory`) holds the
estimate, and `build_engine` makes the same check for the end-to-end checks and the benchmarks.

The family of the model (`model_type` in `config.json`, `engine/model/family.py`) gives the defaults of `--slots`,
`--max-seq-len` and `--spec`, the size of the checkpoint ring and the image input. The table shows the defaults for
the 27B (the `qwen3_5` family). The section "Qwen3.8-Flash-Next" below gives those of the `qwen4_exp` family.

| Flag | Default | |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | |
| `--model` | `nvidia/Qwen3.8-27B-NVFP4` | An HF repo id in the local cache, or a checkpoint directory. The engine serves two families (`model_type` in `config.json`): `qwen3_5`, which includes Qwen3.8-27B, and `qwen4_exp`, Qwen3.8-Flash-Next (`nvidia/Qwen3.8-Flash-Next-NVFP4`, see below). `check_served` in `engine/model/family.py` refuses a `qwen4_exp` checkpoint when the modules of its decode model or its prefill are missing (`engine/model/fast_qwen4exp.py`, `prefill_qwen4exp.py`). The server and `build_engine` call it before their memory checks. |
| `--slots` | 3 (the 27B) | Conversations whose KV stays cached. `qwen4_exp`: 2. The server runs one request at a time; the others wait in a FIFO queue. |
| `--max-seq-len` | 524288 (the 27B) | Tokens per slot (prompt plus output). `qwen4_exp`: 262144, its native context. The fp8 KV cache costs 32 KB per token per slot. The checkpoint was trained on 262,144 positions. Past them, the server scales the rope with YaRN by the ratio (x2 at the default), as Qwen documents for its models. YaRN is static: it applies to every request, and costs a little quality on short prompts. Set `--max-seq-len 262144` for the unscaled model. |
| `--spec` | `mtp` (the 27B) | `none` turns off speculation and runs plain one-token decode. `qwen4_exp`: `none`. |
| `--drafter-weights` | `auto` | MTP head weights. `auto` uses the fine-tuned head of the prepared directory (`tools/prepare.py`, `tools/train_drafter.py`) when it has one. `none` uses the weights of the checkpoint. You can also give a path. Drafts change the speed, never the outputs. The first start with a given drafter file quantizes it and keeps the result in `~/.cache/colinfer/drafter/quant-<key>.safetensors` for the next starts. |
| `--decode-weights` | `int` | `int`: decode the attention / GDN projections from the INT6 / INT5 copies of the prepared directory (`tools/prepare.py`, which runs `tools/int6_requant.py`) when it has them. This gives ~9.5% faster decode, perplexity within 0.25%, and uses +5.2 GB of GPU memory. Without the files, or with `checkpoint`, decode reads the FP8 weights. |
| `--no-prefix-caching` | off | Never reuse a prompt prefix: no checkpoint ring, no conversation cache on disk. Use it for raw-prefill benchmarks, like `--no-enable-prefix-caching` in vLLM. |
| `--session-cache-gb` | 64 | The disk budget of the conversation cache (see below). The least recently used files go first. 0 turns the cache off. |
| `--no-vision` | off | No image input. Images then become a short text note. |
| `--ngram-cache-gb` | 1.0 | `qwen4_exp` only: the host cache of the n-gram rows that the PLE layer reads from the disk, in GB of 160-byte rows (1.0: 6.25M rows, 1.1 GB of host memory with the index). The memory check counts it. 0 turns the cache off. The 27B has no n-gram table and ignores the flag. |
| `--mem-cap-gb` | 95 | A hard cap on the torch allocator, in GB of 10^9 bytes. Past the cap, the allocator raises an error, and the process exits and restarts. Torch reserves 88.6 GB at startup and 89.5 GB after the first image. The cap does not bound the CUDA context, the pinned host buffers, the Python process or the OS. Thus the cap leaves about 36 GB of the 130.7 GB of `MemTotal` for them. |
| `--thinking` | `auto` | The default `enable_thinking`. `auto` uses the template default, which is on. A `/v1/messages` request without a `thinking` field runs without reasoning, as on the Anthropic API, unless the value is `on`. |
| `--reasoning-effort` | `auto` | The reasoning effort of a request with thinking on that does not set an effort: `low`, `medium` or `xhigh`. `auto` uses the template default, `xhigh`. A request that sets an effort keeps it. |
| `--committee-size` | 3, or the roles of the file | Members of a committee reply (a request that names the model `<name>-committee`, see "Committee mode" below). 0 turns the alias off. |
| `--committee-roles` | the built-in roles | A JSON file `{"roles": [...], "reconcile": "..."}` with the members' roles and the reconcile instruction (`{n}`: the number of answers). |
| `--no-selftest` | off | Skip the startup self-test of the matmul kernels (`engine/selftest.py`). For development only: the test is what catches a miscompiled kernel. |
| `--no-warmup` | off | Skip the warm-up requests. The first real requests then pay for the first-use compilation and autotuning. |
| `--api-key` | `$COLINFER_API_KEY`, else none | The `/v1/` endpoints require this key, as `Authorization: Bearer <key>` or `x-api-key: <key>`. `/health` and `/metrics` stay open. |

These values have no flag. They are the defaults of `ServerConfig` in `engine/server/worker.py`:

- The queue holds 8 waiting requests. Past that, a request gets a 503 (`/v1/messages`: 529, `overloaded_error`), which
  Hermes Agent and Claude Code retry.
- A request body is at most 32 MB, the request limit of the Anthropic API: the JSON with the messages and the base64
  images. A larger body gets a 413 (`/v1/messages`: `request_too_large`), and the server stops reading it at the limit.
- The text of a prompt, as the chat template renders it, is at most 8 bytes of UTF-8 per token of the slot: 4,194,304
  at the default. A longer text gets the 400 `prompt is too long` before the tokenize, which holds host memory in
  proportion to the text. Text and code use 2 to 4 bytes per token, so a prompt that fits the slot fits this limit.
- At most 9 requests, the queue depth plus one, are between the body read and the queue at once. The next one gets the
  busy reply before the server reads its body. The tokenizes and the image decodes run one at a time.
- A request ends 1,800 s after it arrived, running or still in the queue, with `finish_reason` `length` (`stop_reason`
  `max_tokens`). There is no other cap on the output tokens: a request can use the room left in its slot.
- The prefix-checkpoint ring holds 32 snapshots of 154 MB. The server allocates it at startup.
- A cycle drafts up to 7 tokens with the MTP head, and uses suffix-match drafts for repeats of 8 or more tokens.
- The conversation cache lives in `~/.cache/colinfer/sessions`, one subdirectory per engine configuration (checkpoint,
  decode weights, drafter). The budget covers all of them.
- An image is resized to at most 4,194,304 pixels (2048 x 2048), one prompt token per 32 x 32 pixels, so at most 4,096
  tokens. The vision tower (0.92 GB) loads on the first image request, which is then 1-2 s slower.

## Qwen3.8-Flash-Next

```bash
./py -m engine.server --model nvidia/Qwen3.8-Flash-Next-NVFP4   # 2 slots x 262,144 tokens, no speculation
```

The `qwen4_exp` family has its own defaults: 2 slots of 262,144 tokens (its native context), 16 checkpoints of 115.6 MB
in the ring, `--spec none` and no image input. The prefill runs in chunks of 4,096 tokens. The PLE layer reads 16 rows
of the n-gram table (51.2 GB on the disk) per token. `engine/weights/ple.py` (`PleFeeder`) reads them through a host
row cache of `--ngram-cache-gb`, for the prefill chunks and for each decode step. The chat template is the 27B's, so
`reasoning_effort`, `enable_thinking` and the tool calls work the same way.

Measured on 2026-10-24 at the defaults:

- The server is ready 42 s after the start: 35 s for the weights (78.5 GB on the GPU), 2 s for the self-test and the
  graphs, 4 s for the warm-up.
- `MemAvailable` falls by 96.7 GB from the start to the ready line: 8.7 GB of slot state, 1.1 GB of n-gram row cache
  and 86.9 GB for the rest (`QWEN4EXP_FIXED_BYTES` in `engine/model/family.py`, 87 GB). Torch reserves 91.2 GB, under
  the default `--mem-cap-gb` of 95. The memory check thus asks for about 103 GB with the 6 GB reserve.
- Plain decode runs at 19 to 20 tok/s at short context (the exact BF16 path, the n-gram rows read before each step).
  A prompt of 4,641 tokens prefills in 1.7 to 1.9 s. The next turn of that conversation restores its checkpoint, and
  its time to first token is 0.13 s.
- A turn of a 7,398-token conversation that two other conversations evicted from both slots restores from the
  conversation cache on disk in 0.16 s, with the output of a restore from memory.

`tests/scheduler_check_qwen4exp.py` is the end-to-end check of the family through the scheduler (under the GPU lock).

The engine does not support these yet for this family:

- images: an image becomes a short text note, as with `--no-vision`
- speculation: the MTP drafter is milestone M3 of `docs/history/qwen4exp_plan.md`, so the server refuses `--spec mtp`
- YaRN: the server refuses a `--max-seq-len` past the 262,144 native positions
- the requantized decode copies (`--decode-weights int` reads the checkpoint's BF16 weights)

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
- **Streaming:** `stream` with `stream_options.include_usage`. While a stream has no output yet (the queue, a long
  prefill), the server sends an SSE comment line every 10 s, so an idle timeout of the client or a proxy does not end
  it.
- **Constrained decoding** (`engine/runtime/grammar.py`, with llguidance):
  - `response_format` `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": {...}}}`: the
    reply is one JSON value that matches the schema (`strict` is always on). `/v1/completions` takes it too.
  - `tool_choice` `required`, or `{"type": "function", "function": {"name": ...}}`: the reply is tool calls (to that
    function), with the parameter names of its schema and values of their types. `parallel_tool_calls: false`: one
    call.
  - With thinking on, the model reasons freely first; the constraint applies after the reasoning.
  - Speculation stays exact: each verify row is masked by the grammar after the drafts before it, so a cycle emits
    what plain constrained decoding would. The masks cost ~0.2 ms a cycle. The first constrained request builds the
    grammar tokenizer (~1.5 s); a schema compiles in a few ms. A schema or a tool name that the grammar cannot take
    gets a 400.
- **Images:** content parts `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` in user and
  tool messages (PNG, JPEG, WebP, GIF: what Pillow reads). The server accepts only base64 data URLs: it fetches
  nothing. A system message cannot hold images (the chat template rejects them with a 400).
- **Prompt rendering:**
  - `tools`, `tool_choice` (`none` leaves the tools out of the prompt; `required` and a named function force tool
    calls, see below).
  - `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, ...), plus top-level `enable_thinking` and
    `reasoning_effort` (`none`, `low`, `medium`, `high`).
  - The template accepts the efforts `xhigh` (its default), `medium` and `low`. The server changes the effort names of
    other APIs to these levels, in `chat_template_kwargs` and at the top level: `none` turns off thinking, `minimal`
    becomes `low`, and `high` and `max` become `xhigh`. At the top level, other names also become `xhigh`. In
    `chat_template_kwargs`, other names go to the template, which rejects them with a 400.
  - `chat_template_kwargs.thinking` (the name in DeepSeek templates) sets `enable_thinking` when the request does not
    set it. Thus `{"thinking": false, "reasoning_effort": "none"}`, which Hermes Agent sends for DeepSeek, turns off
    thinking.
- **Prefix reuse:** `cache_salt` and `cache_prompt: false`. Only requests with the same salt reuse a checkpoint.
  `cache_prompt: false` stops reuse for this request.
- **Extras:**
  - `return_token_ids` (a vLLM extension that llama-benchy uses).
  - The server merges an `extra_body` object that a client sends literally inside the JSON.

The server accepts the `user` field and ignores it. It rejects these with a 400:

- `n > 1`, `echo`, and a prompt longer than the slot
- `presence_penalty` or `frequency_penalty` other than 0, and `repetition_penalty` other than 1. The sampler has no
  penalties. The neutral values pass, because they do not change the output.
- a malformed request: a body that is not a JSON object, a field of the wrong type (a number field, `stop` other than
  strings, messages or tools that are not objects), token ids outside the vocabulary (a prompt of token ids,
  `stop_token_ids`), `min_p` outside [0, 1], `max_tokens` or `max_completion_tokens` below 1, and an error of the chat
  template (no user message, an unknown role).

Any other error is a bug of the server: a 500 whose message names the exception, with the traceback in the log.

Responses:

- **Reasoning:** the reasoning text goes in `reasoning_content`, with a copy in `reasoning`. The final answer goes in
  `content`.
- **Tool calls:**
  - Qwen's XML tool calls become OpenAI `tool_calls`.
  - The JSON schema of the request sets the types of the argument values.
  - `finish_reason` is `tool_calls`.
  - In a stream, each call arrives in one delta when it is complete.
- **Usage:** `usage` includes `prompt_tokens_details.cached_tokens`, the prompt tokens that the server restored from a
  checkpoint, in memory or on disk.
- **Timings:** a `timings` object gives one count and one rate per phase: the queue time, the TTFT, the prefill tokens
  and speed, the decode tokens, time and speed, and the speculation of the decode: the cycles, the draft tokens that the
  cycles verified and accepted, and the acceptance rate. A rate is `null` when the request did not reach its phase.
- **Streaming metrics:** the first content chunk of a stream carries vLLM-style `request_metrics`.

Other endpoints:

- `GET /v1/models`. The result includes `max_model_len` and the model config, and a second entry `<name>-committee`
  (committee mode, below) with its roles.
- `GET /health`.
- `GET /metrics` (Prometheus), with:
  - step time by kind (prefill chunk, or decode cycle)
  - TTFT and queue time
  - tokens per speculative cycle, and the draft and accepted token counters, by draft source (`source="mtp"`: the
    MTP head; `source="suffix"`: copies of earlier text). The acceptance rate of each drafter is
    `rate(colinfer_spec_accepted_tokens_total[5m]) / rate(colinfer_spec_draft_tokens_total[5m])` per source; a drop in
    the `mtp` rate shows the drafter doing worse on the current kind of work.
  - prompt, cached and generated token counters, and the prompt tokens restored from disk
  - queue depth and busy slots
  - allocator memory
  - the conversation cache on disk: bytes, files, saves, restores, errors
- `GET /v1/status`. It shows the phase and length of each slot, the checkpoints, the conversation cache and the startup
  timings.

### Anthropic Messages API

`POST /v1/messages` and `POST /v1/messages/count_tokens` accept the Anthropic Messages format. Claude Code uses them
(`docs/claude_code.md`). The server converts a request to the same chat messages as `/v1/chat/completions`
(`engine/server/anthropic.py`):

- **System prompt:** the `system` field becomes the first system message. The chat template accepts a system message
  only at the start. Thus a system message inside `messages` becomes a user turn in `<system-reminder>` tags. Claude
  Code sends its environment details and reminders this way.
- **Tools:** `tool_use` blocks become tool calls, and `tool_result` blocks become tool messages. A result with
  `is_error` starts with the line `[tool error]`, so the model sees that the call failed. Server tools, for example
  `web_search_20250305`, are not available. The server leaves them out of the prompt. A request that has only
  server tools gets a 400 error.
- **Thinking:**
  - `thinking.type` `enabled` or `adaptive` turns on reasoning, and `disabled` turns it off. Without `thinking`, the
    request runs without reasoning, unless the server runs with `--thinking on`.
  - `output_config.effort` sets the reasoning effort as `reasoning_effort` does on the chat endpoint: `low` and
    `medium` as they are, higher values the template default (`xhigh`). Without an effort, the request gets the
    server's `--reasoning-effort`.
  - The reply has a `thinking` block. Send it back unchanged in the next request. The template then repeats the
    reasoning of the earlier turns, and the server can reuse its prefix checkpoint.
- **Other fields:**
  - `max_tokens`, `temperature`, `top_p`, `top_k` and `stop_sequences` work as on the Anthropic API.
  - `tool_choice` `none` keeps the tools in the prompt and drops the tool calls of the reply. `any` and a named tool
    force the reply to be tool calls (to that tool), one with `disable_parallel_tool_use`, as on the chat endpoint.
  - `output_format` (also `output_config.format`) `{"type": "json_schema", "schema": {...}}` forces a JSON reply.
  - Image blocks with a base64 source go to the model, also inside a `tool_result` (Claude Code sends screenshots and
    the images that its Read tool opens this way). An image with a `url` source gets a 400: the server fetches
    nothing. Documents (PDF) become a short text note.
  - A final assistant message (prefill) gets a 400 error.
  - The server ignores `cache_control`, `metadata`, `context_management`, the thinking budget and `display`.
- **Usage:** `cache_read_input_tokens` is the prompt part that the server restored from a checkpoint (in memory or on
  disk), and `input_tokens`
  is the rest.
- **Stream:** the server sends `message_start` at once. Then it sends a `ping` event every 10 s until the first output,
  for example during a long prefill. Thus the stream watchdog of the client does not stop the request.
- **Errors:** the errors have the Anthropic format. A prompt that is too long gets
  `prompt is too long: <n> tokens > <max> maximum`. Claude Code recognizes this text and compacts the conversation.

### Committee mode

A request that names the model `<name>-committee` (or `committee`) gets one answer written from several. The server
runs one *member* request per role: the same conversation, with the role appended to its last user message (by
default an implementer, a skeptic and a reviewer; `engine/server/committee.py`). The members run one after another
through the normal queue. Then a *reconcile* request, the conversation with the members' answers appended, writes the
final answer. That answer is the reply, in the format of the endpoint (`/v1/chat/completions` or `/v1/messages`,
with or without a stream). The members' answers appear in the server log, and in the `committee` field of a chat
completion without a stream.

To use it, name the model `<name>-committee` in a request. Nothing else changes:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "qwen3.8-27b-committee", "messages": [{"role": "user", "content": "Review this design: ..."}]}'
```

The reply is a normal chat completion. Its `committee` field holds the members' answers, and the server log shows one
line per member (`[chat member 1/3 ...]`), then the reply. On the Anthropic API, the `model` field of a `/v1/messages`
request works the same way; Claude Code switches the model for one question with `/model` (`docs/claude_code.md`,
section 8). To change the members, start the server with `--committee-roles roles.json`, a file like this (`{n}` in
the reconcile instruction becomes the number of answers):

```json
{"roles": ["You are the implementer of this committee. ...",
           "You are the skeptic of this committee. ...",
           "You are the reviewer of this committee. ..."],
 "reconcile": "The {n} members of a committee answered the request above independently. ... Write the single best final answer ..."}
```

To find out whether it helps on your questions, on a running server:

```bash
uv run python tools/committee_ab.py                        # the built-in design and review questions, single vs committee
uv run python tools/committee_ab.py --prompts questions.md # your own prompts, separated by blank lines
# judge ~/.cache/colinfer/committee_ab/<run>/pairs.md without looking at the key, then:
uv run python tools/committee_ab.py --reveal ~/.cache/colinfer/committee_ab/<run>
```

- **Cost:** about 4x the time of a single answer for answers of the usual length. The members' prompts share the
  conversation's prefix, so with the prefix cache each member prefills its role text only, and the reconcile pass
  prefills the answers. A stream sends keep-alive pings while the members run, so the client's watchdog does not end it.
- **Use it for one question at a time:** a review, a design question, a plan. In an agentic session, every tool-calling
  turn would pay the 4x. Claude Code switches the model for one question with `/model` (`docs/claude_code.md`).
- **Sampling:** the request's `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, tools and thinking apply to every
  member and to the reconcile pass. With a `seed`, member *i* uses `seed + i` and the reconcile pass the seed itself,
  so a committee reply is reproducible. Constraints (`response_format`, a forced `tool_choice`) and `logprobs` apply
  to the reconcile pass only. A member's tool calls reach the reconcile pass as text.
- **Measure before you trust it:** `tools/committee_ab.py` asks a running server the same questions as a single
  answer and as a committee, and writes the pairs in a random order for a blind judgment, with the times.

## Behavior to know

- **Prefill past 16k tokens of context uses FP8 attention** (Q K^T on FP8 tensor cores, `csrc/attn_prefill.cu`). It is 6%
  faster at 64k and 11% faster at 128k, and it adds 0.25-0.3% to the perplexity of those chunks. See
  `ATTN_FP8_MIN_CTX` in `engine/model/prefill.py` and `docs/history/phase6_progress.md` section 8.
- **Speculation is always on, except when the server runs with `--spec none`.** The output is token-identical to plain
  decode, greedy or sampled (with the same seed). `--spec none` gives exactly the same tokens, only slower.
- **The draft length adapts.** A cycle verifies k+1 rows in one weight pass of up to 16 rows. It picks k = 3 or 7 for
  each cycle, for the most expected tokens per second, from the acceptance of the request. Code and structured output
  usually draft 7, and prose drafts 3.
- **The draft steps stop early when they are unlikely to help.** The product of the drafter probabilities of the drafts
  of a cycle can fall below 0.1. Then the remaining draft steps of the cycle skip their weight
  GEMMs (~0.35 ms instead of ~1.8 ms a step). The verify rejects their junk drafts, so the outputs do not change. See
  `DRAFT_STOP` in `engine/spec/mtp.py` and `docs/history/phase6_progress.md` section 18.
- **Low-rank draft head.** `tools/lowrank_draft_head.py` makes the basis, and `tools/prepare.py` puts it in the prepared directory.
  With this file, each draft step scores a rank-1024 approximation of the draft lm head and rescores its top 256
  candidates exactly. This gives the same drafts as the full head for ~1/5 of its bytes (k=7 cycle 89.8 → 86.0 ms).
  Without the file, the engine uses the full draft head (`docs/history/phase6_progress.md` section 19).
- **Suffix-match drafts (repeats of 8 or more tokens).** The history of a request is its prompt and the reply
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
  - after the last message, if it has 512 tokens or more (`REPLY_SPLIT_MIN` in `engine/runtime/scheduler.py`)
  - every 8192 prompt tokens

  The checkpoint after the last message is for clients that send a reply back in a different form. For example, Hermes
  Agent sends it back without its reasoning. Then the next prompt differs from the end-of-prompt checkpoint in the last
  token of the generation prompt, and without this checkpoint the server prefills the last message again. This
  checkpoint costs one more weight pass (~0.1 s). Claude Code sends the reasoning back, and its next prompt restores
  the end of the reply. A request that restored the end of a reply does not take this checkpoint.

  A request that shares only a prefix with a cached conversation (a title request or a subagent with the same system
  prompt) would overwrite the rest of that conversation in its slot. When that rest is 1,024 tokens or more
  (`COPY_MIN_TAIL`) and longer than what another slot holds, the request runs in that other slot instead, with a GPU
  copy of the shared prefix (milliseconds). Thus the main conversation keeps its cache. A shorter rest, such as the
  reply that Hermes Agent sends back changed, is overwritten where it is.
- **Conversation cache on disk** (`engine/runtime/sessions.py`). When a request ends, the server writes its reply-end
  and message-boundary checkpoints to the NVMe, for prefixes of 2,048 tokens or more, with the KV rows before them.
  A background thread writes them, so the next request does not wait. A new request restores from disk when the disk
  holds a longer prefix of its prompt than the memory checkpoints (by 256 tokens or more): a conversation that lost its
  slot to three others, or any conversation after a restart. The restore puts back the exact bytes that the slot held,
  so the reply is the same as after a restore from memory. Direct reads run at ~12 GB/s: a 21k-token conversation
  restores in 0.2 s to the first token, against 7.2 s of prefill (`tests/scheduler_check.py`).

  The KV rows are stored in blocks of 1,024 tokens, named by the hash of all the tokens before their end, so
  conversations that share a system prompt share its blocks on disk. A request with `cache_prompt: false` writes
  nothing, and requests with different `cache_salt` values share nothing. The files depend on the checkpoint, the
  decode weights and the drafter weights: another configuration writes to its own subdirectory. A file that is
  missing or damaged makes that request prefill instead.
- **Images** (`engine/model/vision.py`, `engine/server/images.py`). The server resizes an image as the checkpoint's
  image processor does, to multiples of 32 pixels between 65,536 and 4,194,304 pixels, and gives it one
  prompt token per 32 x 32 pixels: a 1920 x 1080 screenshot is 2,040 tokens. The vision tower runs during the prefill
  of those tokens, 0.4 s for a 1080p screenshot. A conversation that sends the same image again restores it from the
  prefix caches without running the vision tower again: the image is keyed by its bytes and its grid, which the
  server gets from the image's size, so a resent image costs a decode (~30 ms) and no processing until the caches
  miss.

  The checkpoint's static activation scales were calibrated on text. Measured against the BF16 model of transformers
  on four test images, the engine's answer tokens differ more in log-probability on images (0.18 on average) than on
  the same content given as text (0.005); most of this comes with the NVFP4 / FP8 weights themselves. NVIDIA reports
  MMMU-Pro 74.86 for this checkpoint against 75.14 in BF16. Text that is small in the image can be misread.
- **One request at a time.** A request that arrives while another runs waits in the queue, also behind a long
  prefill. The three slots keep the last three conversations cached: a side request goes to another slot, so the main
  conversation keeps its cache. The queue holds 8 requests; past that, requests are refused with a 503
  (529 on `/v1/messages`). A request ends 1,800 s after it arrived, so one runaway request cannot
  hold the engine for longer than that.
- **Failures restart the process.** A failed CUDA call fails each in-flight request with a 500 and stops the process.
  Then systemd restarts it. An error while the server formats the output of one request (the parser runs on the engine
  thread) fails only that request, with a 500.

## systemd

`deploy/colinfer.service` is a user unit. Its paths assume that the clone is `~/Projects/colin-inference-engine`. If
your clone is elsewhere (a plain `git clone` creates `./colinfer`), change `WorkingDirectory` and `ExecStart` first.
Then, from the root of the clone:

```bash
systemctl --user link "$PWD/deploy/colinfer.service"
systemctl --user enable --now colinfer
journalctl --user -u colinfer -f
```

# Use Qwen3.8-27B in Claude Code

This manual tells you how to run Qwen3.8-27B on a DGX Spark with colinfer. It also tells you how to connect the Claude
Code CLI to this model. Claude Code then sends its model requests to your machine, not to Anthropic.

We tested this procedure on 2026-10-08 with these versions:

- colinfer at the commit that added this version of the file
- Claude Code 2.1.290

> **NOTE:** Anthropic does not support Claude Code with models other than Claude. Qwen3.8-27B is a smaller model than
> Claude. Some functions of Claude Code do not work with it, and the quality of the work is lower.

## How the parts connect

```
Claude Code CLI  --Anthropic Messages API-->  colinfer server  -->  GB10 GPU
(claude-qwen)                                 127.0.0.1:8000
```

Claude Code sends its requests in the Anthropic Messages format. The colinfer server accepts this format at
`/v1/messages`, so you do not need a proxy.

You use two terminals:

| Terminal | Process | When you close it |
|---|---|---|
| 1 | The colinfer server | At the end of your work |
| 2 | Claude Code | When you want |

## 1. Requirements

- A DGX Spark with colinfer installed. First, do the "Quick start" steps in `README.md` (`uv sync` and the checkpoint
  download).
- Approximately 65 GB of free unified memory for the server.
- The Claude Code CLI. For the installation, refer to <https://code.claude.com/docs/en/setup>. To examine the
  installation, type `claude --version`.
- The directory `~/.local/bin` in your `PATH`.

## 2. Start the colinfer server (terminal 1)

> **CAUTION:** Stop all other GPU work before you start the server. If the processes use more memory than the machine
> has, the machine can power off. It does not always show an out-of-memory error first.

1. Go to the root directory of your colinfer clone:

   ```bash
   cd colinfer
   ```

2. Start the server:

   ```bash
   uv run python -m engine.server
   ```

3. Wait for the line that ends with `listening on http://127.0.0.1:8000/v1`. The startup takes 30 to 60 seconds. On
   the first start, the CUDA kernels also compile. This adds approximately 1 minute.

4. In a different terminal, make sure that the Anthropic API of the server answers:

   ```bash
   curl -s localhost:8000/v1/messages/count_tokens -H 'content-type: application/json' \
     -d '{"messages": [{"role": "user", "content": "hi"}]}'
   ```

   The result must be `{"input_tokens":13}`.

Keep terminal 1 open. The server shows one line for each request that it completes. The requests from Claude Code
show as `[msg ...]`, for example:

```
22:10:33 [msg 24] slot 0 prompt 15887 (cached 15818) -> 588 tok, stop; queue 0.00s ttft 0.13s decode 49.7 tok/s
```

> **NOTE:** To run the server as a systemd service, refer to the "systemd" section of `docs/server.md`.

## 3. Make the claude-qwen command (one time only)

The `claude-qwen` command starts Claude Code with the settings for the local model. The `claude` command does not
change. It continues to use Claude.

1. Make the script:

   ```bash
   cat > ~/.local/bin/claude-qwen <<'EOF'
   #!/usr/bin/env bash
   # Claude Code -> colinfer server (127.0.0.1:8000, Anthropic Messages API): Qwen3.8-27B
   export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
   export ANTHROPIC_AUTH_TOKEN="$(cat ~/.config/colinfer/api.key 2>/dev/null || echo colinfer)"
   export ANTHROPIC_MODEL=qwen3.8-27b
   export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-27b
   export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-27b
   export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-27b
   export CLAUDE_CODE_ATTRIBUTION_HEADER=0
   export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
   exec claude --permission-mode acceptEdits "$@"
   EOF
   ```

2. Make the script executable:

   ```bash
   chmod +x ~/.local/bin/claude-qwen
   ```

3. Make sure that your shell finds the script:

   ```bash
   command -v claude-qwen
   ```

   The result must be the path of the script. If the result is empty, add `~/.local/bin` to your `PATH`.

The script sets these values:

| Setting | Function |
|---|---|
| `ANTHROPIC_BASE_URL` | Claude Code sends all model requests to the colinfer server. |
| `ANTHROPIC_AUTH_TOKEN` | The key that Claude Code sends to the server. It is the content of `~/.config/colinfer/api.key` if this file exists (section 6), else `colinfer`. Without a key, the server ignores this value. You must set it: if you do not, Claude Code sends the token of your claude.ai login to the server. |
| `ANTHROPIC_MODEL` | The model name of the session. The server accepts all model names. Claude Code uses the name to select the context window and the functions of the model. |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Subagents and background tasks can use the model aliases `opus`, `sonnet` and `haiku`. These three lines give them the same model name, so Claude Code treats all requests the same. |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Removes a block at the start of the system prompt. This block is different in each conversation. Without it, a new session can use the prefix cache of the server. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | Stops telemetry, error reports and automatic updates. |
| `--permission-mode acceptEdits` | Claude Code edits files without a question. It asks before it runs most commands. Do not use auto mode (refer to section 4). |

## 4. Use Claude Code with the local model (terminal 2)

> **CAUTION:** Do not use auto mode with the local model. In auto mode, Claude Code sends each command to a safety
> check before the command runs. With this configuration, Qwen3.8-27B does this check, not Claude. In our test, one
> check took 33 seconds. The check was made for Claude models.

1. Make sure that the server runs (section 2).

2. Go to the directory of your project.

3. Start Claude Code:

   ```bash
   claude-qwen
   ```

4. Type your request, then push Enter.

5. Look at terminal 1. Make sure that a new `[msg ...]` line shows for each request.

For one request without the interactive screen, use the `-p` option:

```bash
claude-qwen -p "Add a function perimeter(w, h) to shapes.py."
```

At the start, Claude Code shows these messages. They are normal for a model that is not Claude:

- `"qwen3.8-27b" isn't described by this version's model catalog ...`: Claude Code does not know the model, so it
  assumes a context window of 200,000 tokens. A server slot holds 262,144 tokens, so you do not have to change this
  value.
- `claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth source is set ...`: the connectors of
  your claude.ai account are not available in this session.

To change the permission mode during a session, push Shift+Tab. Use `acceptEdits` or `default`.

## 5. Stop the system

1. In terminal 2, type `/exit`.
2. In terminal 1, push Ctrl+C. The server stops in approximately 3 seconds and releases its GPU memory.

## 6. Use an API key (optional)

By default, the server has no authentication. It listens only on `127.0.0.1`, but other users and programs on the same
machine can send requests to it. To prevent this, use an API key.

1. Make a random key. Only your user can read the key file:

   ```bash
   mkdir -p ~/.config/colinfer
   (umask 077 && openssl rand -hex 16 > ~/.config/colinfer/api.key)
   ```

2. Start the server with the key:

   ```bash
   COLINFER_API_KEY="$(cat ~/.config/colinfer/api.key)" uv run python -m engine.server
   ```

3. Start Claude Code with `claude-qwen`. The script reads the key from the same file.

With a key, the `/v1/` endpoints of the server reject requests without the key. `/health` and `/metrics` stay open.

## 7. Performance and limits

We measured these values on a DGX Spark with the default server settings and a small Python project:

| Item | Value |
|---|---|
| Prompt of the first request of a session (system prompt and tool definitions) | Approximately 15,600 tokens |
| Time to the first token, first request after the server starts | 5.0 s |
| Time to the first token, first request of a later session | 0.9 s (13,400 tokens from the prefix cache) |
| Time to the first token, later requests of a session | 0.1 s to 0.4 s |
| Output speed | 48 to 95 tokens per second |
| Task: add a function to a file, then run a command (`-p`, first session after the server starts) | 25 s |
| Task: find the functions of a file with the Explore subagent (`-p`) | 32 s |
| Task: continue the conversation (`-p --continue`), add a function, then run a command | 20 s |
| Prompt of approximately 130,000 tokens, with no prefix in the cache | 53 s to the first token |

Know these limits:

- **One request at a time.** Subagents and background tasks wait in a queue while another request runs, so parallel
  subagents run one after another. The three slots keep the last three conversations cached.
- **Context length.** Claude Code compacts the conversation before it reaches 200,000 tokens. If a request is longer
  than a server slot, the server sends a `prompt is too long` error. Claude Code shows `Prompt is too long`.
- **Long prompts without the cache.** The server keeps its prefix cache in memory only. After a server restart, the
  first request of a long conversation reads the full prompt again. During this time, the server sends a keep-alive
  event every 10 seconds, so Claude Code does not stop the request.
- **No web search.** Web search is a function of the Anthropic servers. With this configuration, the WebSearch tool
  gets an error.
- **Text only.** The engine does not read images or PDF documents. The model gets a short note in their place.
- **Reasoning.** Claude Code turns on the reasoning mode of the model for the main conversation. Claude Code shows the
  reasoning as thinking blocks.
- **Quality.** Qwen3.8-27B is smaller than Claude. Give it small and clear tasks. Examine each change before you keep
  it.

## 8. Troubleshooting

| Symptom | Possible cause | Remedy |
|---|---|---|
| After approximately 3 minutes, Claude Code shows `API Error: Connection refused ... (ECONNREFUSED)`. | The server does not run, or its startup is not complete. | Start the server. Wait for the `listening on` line (section 2). |
| Claude Code shows `Failed to authenticate. API Error: 401 invalid or missing API key`. | The server uses an API key, and Claude Code sent a different key. | Start the server and Claude Code with the same key file (section 6). |
| Each command waits 15 to 35 seconds before it runs. | Auto mode is on. | Push Shift+Tab until the mode is `acceptEdits` or `default` (section 4). |
| Claude Code shows `Prompt is too long`. | The conversation is longer than a server slot. | Type `/compact`, or type `/clear` to start a new conversation. |
| The WebSearch tool fails with `API Error: 400 this server does not run server tools (web_search_20250305)`. | Web search is not available (section 7). | Give the information to the model in your request. |
| The first answer after a server restart is slow. | The server reads the full prompt again. | No action is necessary. Refer to section 7. |
| The server stops with `engine failure`. | A CUDA error occurred. | Start the server again. If the server runs under systemd, the service restarts automatically. |

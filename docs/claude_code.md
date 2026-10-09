# Use Qwen3.8-27B in Claude Code

This manual tells you how to run Qwen3.8-27B on a DGX Spark with colinfer. It also tells you how to connect the Claude
Code CLI to this model. Claude Code then sends its model requests to your machine, not to Anthropic.

We tested this procedure on 2026-10-08 with these versions:

- colinfer at the commit that added this file
- Claude Code 2.1.290
- LiteLLM 1.104.2

> **NOTE:** Anthropic does not support Claude Code with models other than Claude. Qwen3.8-27B is a smaller model than
> Claude. Some functions of Claude Code do not work with it, and the quality of the work is lower.

## How the parts connect

```
Claude Code CLI  --Anthropic API-->  LiteLLM proxy   --OpenAI API-->  colinfer server  -->  GB10 GPU
(claude-qwen)                        127.0.0.1:4000                   127.0.0.1:8000
```

Claude Code sends requests in the Anthropic Messages format. The colinfer server accepts only the OpenAI format. The
LiteLLM proxy changes each request and each reply from one format to the other.

You use three terminals:

| Terminal | Process | When you close it |
|---|---|---|
| 1 | The colinfer server | At the end of your work |
| 2 | The LiteLLM proxy | At the end of your work |
| 3 | Claude Code | When you want |

## 1. Requirements

- A DGX Spark with colinfer installed. First, do the "Quick start" steps in `README.md` (`uv sync` and the checkpoint
  download).
- Approximately 65 GB of free unified memory. The server uses approximately 64 GB, and the proxy uses approximately
  0.4 GB.
- The Claude Code CLI. For the installation, refer to <https://code.claude.com/docs/en/setup>. To examine the
  installation, type `claude --version`.
- An internet connection when you start the proxy for the first time. At that time, `uvx` downloads LiteLLM.
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

4. In a different terminal, make sure that the server answers:

   ```bash
   curl -s localhost:8000/health
   ```

   The result must be `{"status":"ok"}`.

Keep terminal 1 open. The server shows one line for each request that it completes, for example:

```
21:40:39 [chat 7] slot 1 prompt 15792 (cached 0) -> 213 tok, stop; queue 0.00s ttft 5.14s decode 77.2 tok/s
```

> **NOTE:** To run the server as a systemd service, refer to the "systemd" section of `docs/server.md`.

## 3. Configure the proxy (one time only)

1. Make a directory for the configuration files:

   ```bash
   mkdir -p ~/.config/colinfer
   ```

2. Make a random key for the proxy. Only your user can read the key file:

   ```bash
   (umask 077 && echo "sk-$(openssl rand -hex 16)" > ~/.config/colinfer/litellm.key)
   ```

3. Make the configuration file of the proxy:

   ```bash
   cat > ~/.config/colinfer/litellm.yaml <<'EOF'
   model_list:
     - model_name: qwen3.8-27b
       litellm_params:
         model: hosted_vllm/nvidia/Qwen3.8-27B-NVFP4
         api_base: http://127.0.0.1:8000/v1
         api_key: none
   litellm_settings:
     drop_params: true
   EOF
   ```

The configuration file has these entries:

| Entry | Function |
|---|---|
| `model_name: qwen3.8-27b` | The model name that Claude Code sends. Section 5 uses the same name. |
| `model: hosted_vllm/...` | The prefix `hosted_vllm/` tells LiteLLM that the server has an OpenAI-compatible API. The server ignores the name after the prefix. |
| `api_base` | The address of the colinfer server. |
| `api_key: none` | The colinfer server has no authentication. LiteLLM sends this value, and the server ignores it. |
| `drop_params: true` | LiteLLM removes the request fields that an OpenAI-compatible server does not accept. |

## 4. Start the proxy (terminal 2)

> **CAUTION:** Always use `--host 127.0.0.1`. Without this option, LiteLLM listens on all network interfaces
> (`0.0.0.0`). Then other computers on your network can connect to the proxy.

1. Open a second terminal.

2. Start the proxy:

   ```bash
   LITELLM_MASTER_KEY="$(cat ~/.config/colinfer/litellm.key)" \
     uvx --from 'litellm[proxy]==1.104.2' litellm \
     --config ~/.config/colinfer/litellm.yaml --host 127.0.0.1 --port 4000
   ```

3. Wait for this line:

   ```
   INFO:     Uvicorn running on http://127.0.0.1:4000 (Press CTRL+C to quit)
   ```

   The first start downloads LiteLLM, so it takes more time. Later starts take a few seconds.

4. In a different terminal, make sure that the proxy answers:

   ```bash
   curl -s localhost:4000/health/liveliness
   ```

   The result must be `"I'm alive!"`.

Keep terminal 2 open.

> **NOTE:** The command uses a fixed LiteLLM version (`==1.104.2`). We tested this procedure with this version only. A
> different version can change how LiteLLM converts the requests.

> **NOTE:** LiteLLM does not start without `LITELLM_MASTER_KEY`. Claude Code sends the same key (section 5).

## 5. Make the claude-qwen command (one time only)

The `claude-qwen` command starts Claude Code with the settings for the local model. The `claude` command does not
change. It continues to use Claude.

1. Make the script:

   ```bash
   cat > ~/.local/bin/claude-qwen <<'EOF'
   #!/usr/bin/env bash
   # Claude Code -> LiteLLM proxy (127.0.0.1:4000) -> colinfer server (127.0.0.1:8000): Qwen3.8-27B
   export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
   export ANTHROPIC_AUTH_TOKEN="$(cat ~/.config/colinfer/litellm.key)"
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
| `ANTHROPIC_BASE_URL` | Claude Code sends all model requests to the proxy. |
| `ANTHROPIC_AUTH_TOKEN` | The key of the proxy. Claude Code sends it in the `Authorization` header, instead of your claude.ai login. |
| `ANTHROPIC_MODEL` | The model of the session. It must be equal to `model_name` in `litellm.yaml`. |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Subagents and background tasks can use the model aliases `opus`, `sonnet` and `haiku`. These three lines send the aliases to the local model. Without them, the proxy rejects these requests. |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Removes a block at the start of the system prompt. This block is different in each conversation. Without it, a new session can use the prefix cache of the server. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | Stops telemetry, error reports and automatic updates. |
| `--permission-mode acceptEdits` | Claude Code edits files without a question. It asks before it runs most commands. Do not use auto mode (refer to section 6). |

## 6. Use Claude Code with the local model (terminal 3)

> **CAUTION:** Do not use auto mode with the local model. In auto mode, Claude Code sends each command to a safety
> check before the command runs. With this configuration, Qwen3.8-27B does this check, not Claude. In our test, one
> check took 33 seconds. The check was made for Claude models.

1. Make sure that the server and the proxy run (sections 2 and 4).

2. Go to the directory of your project.

3. Start Claude Code:

   ```bash
   claude-qwen
   ```

4. Type your request, then push Enter.

5. Look at terminal 1. Make sure that a new `[chat ...]` line shows for each request.

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

## 7. Stop the system

1. In terminal 3, type `/exit`.
2. In terminal 2, push Ctrl+C. The proxy stops.
3. In terminal 1, push Ctrl+C. The server stops in approximately 3 seconds and releases its GPU memory.

## 8. Performance and limits

We measured these values on a DGX Spark with the default server settings and a small Python project:

| Item | Value |
|---|---|
| Prompt of the first request of a session (system prompt and tool definitions) | Approximately 16,000 tokens |
| Time to the first token, first request after the server starts | 5.1 s |
| Time to the first token, first request of a later session | Approximately 1 s, if the server keeps the prefix in its cache |
| Time to the first token, later requests of a session | 0.1 s to 1.1 s |
| Output speed | 36 to 77 tokens per second |
| Task: add a function to a file, then run a command (`-p`) | 14 s |
| Task: add a function, then list the files with the Explore subagent (`-p`) | 31 s |
| Prompt of 233,000 tokens, with no prefix in the cache | 94 s to the first token |

Know these limits:

- **Three requests at the same time.** The server has three slots. Subagents and background tasks also use slots.
  Other requests wait in a queue.
- **Context length.** Claude Code compacts the conversation before it reaches 200,000 tokens. If a request is longer
  than a server slot, the server sends an error.
- **Long prompts without the cache.** The server keeps its prefix cache in memory only. After a server restart, the
  first request of a long conversation reads the full prompt again.
- **Functions that use Anthropic servers.** Some functions of Claude Code use the Anthropic servers, for example web
  search. These functions can fail with this configuration.
- **Reasoning.** By default, the server turns on the reasoning mode of the model. Claude Code shows the reasoning as
  thinking blocks.
- **Quality.** Qwen3.8-27B is smaller than Claude. Give it small and clear tasks. Examine each change before you keep
  it.

## 9. Troubleshooting

| Symptom | Possible cause | Remedy |
|---|---|---|
| Claude Code shows a connection error. | The proxy does not run. | Start the proxy (section 4). |
| Claude Code shows an error with `Cannot connect to host 127.0.0.1:8000`. | The server does not run, or its startup is not complete. | Start the server. Wait for the `listening on` line (section 2). |
| Claude Code shows an error with `No connected db.` | Claude Code sent a key that is not the key of the proxy. | Start Claude Code with `claude-qwen`. Start the proxy with the command in section 4. Both use `~/.config/colinfer/litellm.key`. |
| LiteLLM stops at startup with `no master key is set`. | `LITELLM_MASTER_KEY` is not set. | Use the command in section 4, step 2. |
| Claude Code shows an error with `Invalid model name passed in model=...`. | Claude Code sent a model name that is not in `litellm.yaml`. | Start Claude Code with `claude-qwen`. Do not select a Claude model with `/model`. |
| Each command waits 15 to 35 seconds before it runs. | Auto mode is on. | Push Shift+Tab until the mode is `acceptEdits` or `default` (section 6). |
| Claude Code shows an error with `prompt has ... tokens`. | The conversation is longer than a server slot. | Type `/compact`, or type `/clear` to start a new conversation. |
| The first answer after a server restart is slow. | The server reads the full prompt again. | No action is necessary. Refer to section 8. |
| The server stops with `engine failure`. | A CUDA error occurred. | Start the server again. If the server runs under systemd, the service restarts automatically. |

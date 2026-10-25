# Use Qwen3.8-27B in Claude Code

This manual tells you how to run Qwen3.8-27B on a DGX Spark with colinfer, and how to use the model from Claude Code.
The main path is the Claude Code extension of VS Code. Section 9 describes the command line as an alternative.

We tested this procedure on 2026-10-15 with these versions:

- colinfer at the commit that added this version of the file
- Claude Code 2.1.296 (the VS Code extension and the CLI)

> **NOTE:** Anthropic does not support Claude Code with models other than Claude. Qwen3.8-27B is a smaller model than
> Claude. Some functions of Claude Code do not work with it, and the quality of the work is lower.

## How the parts connect

```
Claude Code (VS Code extension or CLI)  -->  gateway 127.0.0.1:8010  --Qwen rows-->    colinfer server 127.0.0.1:8000  -->  GB10 GPU
                                                                      --Claude rows-->  api.anthropic.com
```

Claude Code sends every model request to one address, `ANTHROPIC_BASE_URL`. A small gateway on this machine
(`engine/server/gateway.py`) listens at that address and routes each request by its model name. The Qwen rows of the
model picker go to the colinfer server. The Claude rows go to `api.anthropic.com` with your claude.ai login. You
switch between the two with the model button at the bottom of the prompt box, or with `/model`.

The gateway needs no GPU. When the server is down, the Claude rows keep working.

Three processes take part. The server and the gateway run as systemd user services, so you do not keep a terminal open
for them:

| Process | Where it runs | Function |
|---|---|---|
| colinfer server | `systemctl --user` service `colinfer`, port 8000 | Runs the model on the GPU. |
| gateway | `systemctl --user` service `colinfer-gateway`, port 8010 | Routes the requests of Claude Code. |
| Claude Code | The VS Code extension, or `claude` in a terminal | Your editor assistant. |

## 1. Requirements

- A DGX Spark with colinfer installed. First, do the "Quick start" steps in `README.md` (`uv sync` and the checkpoint
  download).
- Approximately 90 GB of free unified memory for the server (3 slots x 524,288 tokens).
- VS Code with the Claude Code extension, connected to the DGX Spark. A VS Code Server over SSH works. The extension
  brings its own `claude` binary.
- For the command line (section 9): the Claude Code CLI. For the installation, refer to
  <https://code.claude.com/docs/en/setup>.

## 2. Start the server and the gateway (one time only)

> **CAUTION:** Stop all other GPU work before you start the server. If the processes use more memory than the machine
> has, the machine can power off. It does not always show an out-of-memory error first.

1. Go to the root directory of your colinfer clone:

   ```bash
   cd colinfer
   ```

2. Install and start the two services:

   ```bash
   systemctl --user link "$PWD/deploy/colinfer.service" "$PWD/deploy/colinfer-gateway.service"
   systemctl --user enable --now colinfer colinfer-gateway
   ```

   The services start again at each login. To start them at boot, without a login, run `loginctl enable-linger $USER`.

3. Wait for the server. The startup takes 15 to 60 seconds. On the first start, the CUDA kernels also compile. This
   adds approximately 1 minute. This command shows the log of the server:

   ```bash
   journalctl --user -u colinfer -f
   ```

   Wait for the line that ends with `listening on http://127.0.0.1:8000/v1`.

4. Make sure that the gateway and the server answer:

   ```bash
   curl -s localhost:8010/health
   curl -s localhost:8010/v1/messages/count_tokens -H 'content-type: application/json' \
     -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}]}'
   ```

   The first result names the server and the model markers. The second result must be `{"input_tokens":13}`.

The server log shows one line for each request that it completes. The requests from Claude Code show as `[msg ...]`,
for example:

```
22:10:33 [msg 24] slot 0 prompt 15887 (cached 15818) -> 588 tok, stop; queue 0.00s ttft 0.13s | prefill 69 tok 530 tok/s | decode 587 tok 49.7 tok/s | accept 62% (612/980 drafts, 3.6 tok/cycle)
```

The line has one segment per phase. `prefill` counts the prompt tokens that no checkpoint served, with their speed.
`decode` counts the output tokens after the first, with their speed. `accept` is the draft acceptance rate: the draft
tokens that the speculative cycles accepted, over the draft tokens that they verified, and the output tokens per cycle.

The gateway log (`journalctl --user -u colinfer-gateway -f`) shows one line per request: method, path, model,
destination, status and time. It shows no header and no body.

> **NOTE:** To run the server in a terminal instead, use `uv run python -m engine.server`. The server flags are in
> `docs/server.md`. The gateway runs in a terminal with `uv run python -m engine.server.gateway`.

> **NOTE:** The server can also run Qwen3.8-Flash-Next (`--model nvidia/Qwen3.8-Flash-Next-NVFP4`, `docs/server.md`).
> The gateway sends each model name that contains `qwen` to the server, whatever model the server runs. Thus Claude
> Code uses Flash-Next the same way. That model reads text only: a screenshot becomes a short text note.

## 3. Connect Claude Code (one time only)

Claude Code reads its settings from `~/.claude/settings.json`. The same file applies to the VS Code extension and to
the CLI. Add these two keys to it, and keep the keys that are there already:

```json
{
  "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:8010" },
  "modelPicker": {
    "options": [
      { "model": "qwen3.8-27b", "label": "Qwen3.8-27B (local)",
        "description": "colinfer on this machine, through the gateway on 127.0.0.1:8010" },
      { "model": "qwen3.8-27b-committee", "label": "Qwen3.8-27B committee (local)",
        "description": "three members and a reconcile pass, about 4x slower" }
    ]
  }
}
```

| Key | Function |
|---|---|
| `env.ANTHROPIC_BASE_URL` | Claude Code sends all model requests to the gateway. |
| `modelPicker.options` | The gateway routes a model whose name contains `qwen` or `committee` to the server. These rows make the names selectable. Claude Code adds them after the built-in rows. |

> **CAUTION:** Do not set `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` in this setup. A token there replaces your
> claude.ai login, and the Claude rows stop with a 401.

Then reload VS Code: open the command palette (Ctrl+Shift+P) and run "Developer: Reload Window". The extension starts
Claude Code with the new settings.

## 4. Use the local model in VS Code

> **CAUTION:** Do not use auto mode with the local model. In auto mode, Claude Code sends each command to a safety
> check before the command runs. With this configuration, Qwen3.8-27B does this check, not Claude. In our test, one
> check took 33 seconds. The check was made for Claude models.

1. Make sure that the two services run (section 2). `systemctl --user is-active colinfer colinfer-gateway` must show
   `active` twice.

2. Open a new Claude Code chat in VS Code.

3. Click the model button at the bottom of the prompt box, or type `/model`. Pick "Qwen3.8-27B (local)". The Default
   row stays your Claude model.

4. Set the permission mode to `acceptEdits` or `default`. Claude Code then edits files without a question and asks
   before it runs most commands.

5. Type your request, then push Enter.

6. To check the routing, look at the server log (`journalctl --user -u colinfer -f`). A new `[msg ...]` line shows for
   each request.

To go back to Claude, pick a Claude row. The choice applies to the chat. A new chat starts on the Default row.

At the start, Claude Code shows these messages. They are normal for a model that is not Claude:

- `"qwen3.8-27b" isn't described by this version's model catalog ...`: Claude Code does not know the model, so it
  assumes a context window of 200,000 tokens and compacts the conversation before that. A server slot holds 524,288
  tokens, so you do not have to change this value.
- `claude.ai connectors are disabled ...` appears only in the CLI setup of section 9, where a token is set.

## 5. Stop the system

The services keep running until you stop them. The server holds approximately 88 GB of unified memory while it runs.
Stop it before you use the GPU for other work:

```bash
systemctl --user stop colinfer            # the server: frees the GPU memory in approximately 3 seconds
systemctl --user stop colinfer-gateway    # the gateway: the Claude rows then stop until you start it again
```

Without the gateway, every Claude Code request fails with `Connection refused`. To use Claude Code without the local
model at all, remove `env.ANTHROPIC_BASE_URL` from `~/.claude/settings.json` and reload the window.

## 6. Performance and limits

We measured these values on a DGX Spark with the default server settings and a small Python project:

| Item | Value |
|---|---|
| Prompt of the first request of a session (system prompt and tool definitions) | Approximately 15,600 tokens |
| Time to the first token, first request after the server starts | 5.0 s |
| Time to the first token, first request of a later session | 0.9 s (13,400 tokens from the prefix cache) |
| Time to the first token, later requests of a session | 0.1 s to 0.4 s |
| Output speed | 48 to 95 tokens per second |
| Task: add a function to a file, then run a command (first session after the server starts) | 25 s |
| Task: find the functions of a file with the Explore subagent | 32 s |
| Task: continue the conversation, add a function, then run a command | 20 s |
| Prompt of approximately 130,000 tokens, with no prefix in the cache | 53 s to the first token |

Know these limits:

- **One request at a time.** Subagents and background tasks wait in a queue while another request runs, so parallel
  subagents run one after another. The three slots keep the last three conversations cached; a side request with the
  same system prompt runs in another slot, so the main conversation keeps its cache.
- **Context length.** Claude Code compacts the conversation before it reaches 200,000 tokens. If a request is longer
  than a server slot, the server sends a `prompt is too long` error. Claude Code shows `Prompt is too long`.
- **Long prompts without the cache.** The server keeps the prefixes of past conversations in memory and on disk
  (`--session-cache-gb`), so a conversation continues quickly after a server restart, or after other conversations
  took its slot. A long prompt that is new reads in full. During this time, the server sends a keep-alive event every
  10 seconds, so Claude Code does not stop the request.
- **No web search.** Web search is a function of the Anthropic servers. With a Qwen row, the WebSearch tool gets an
  error. The Claude rows keep web search.
- **Images, not PDFs.** The model sees screenshots and image files (a pasted screenshot, or an image that the Read
  tool opens). Small text in an image can be misread. PDF documents become a short note.
- **Reasoning.** Claude Code turns on the reasoning mode of the model for the main conversation. Claude Code shows the
  reasoning as thinking blocks.
- **Quality.** Qwen3.8-27B is smaller than Claude. Give it small and clear tasks. Examine each change before you keep
  it.

## 7. Committee mode for one question (optional)

The server can answer one question from three members of the model and a reconcile pass (`docs/server.md`, "Committee
mode"). The reply takes about four times as long. Use it for one review or design question, not for a whole session:
each tool call of an agentic turn would take four times as long too.

1. Pick "Qwen3.8-27B committee (local)" in the model picker, or type `/model qwen3.8-27b-committee`.

2. Ask the question. The server log shows the three members, then the reply:

   ```
   22:14:03 [msg member 1/3 31] slot 1 prompt 15902 (cached 15887) -> 412 tok, stop; ...
   22:14:15 [msg member 2/3 32] ...
   22:14:27 [msg member 3/3 33] ...
   22:14:40 [msg 34] slot 1 prompt 17204 (cached 15887) -> 388 tok, stop; ...
   ```

3. Pick "Qwen3.8-27B (local)" again.

## 8. What the gateway does with a request

- To `api.anthropic.com`: the request as it came, with the login token and the `anthropic-beta` header. The Claude
  rows bill to your subscription as before.
- To the server: the request without the login token, so the token never reaches the local process, and without the
  attribution block that Claude Code puts first in the system prompt. Thus the prefix cache of the server works across
  sessions.
- It streams each response as the bytes arrive, so a stream and its keep-alive events reach Claude Code at once.
- When the server does not answer, a Qwen row gets a 503 with a clear message. The Claude rows keep working.

Run the server without an API key in this setup. The gateway removes the credentials of a local request, so a server
key would reject every request. The server listens only on `127.0.0.1`.

> **NOTE:** Anthropic does not support non-Claude models through a gateway. With only `ANTHROPIC_BASE_URL` set,
> Claude Code keeps the claude.ai login as its credential.

## 9. The command line (optional)

With the settings of section 3, the `claude` command uses the gateway too. Pick a Qwen row with `/model`, or start a
session on it:

```bash
claude --model qwen3.8-27b
claude --model qwen3.8-27b -p "Add a function perimeter(w, h) to shapes.py."
```

A session that runs only on the local model can also go to the server directly, without the gateway. The `claude-qwen`
command below does this. It also sets an authentication token, so that Claude Code does not send the token of your
claude.ai login to the server. Use it when the gateway is not running, or with a server that has an API key.

1. Make the script:

   ```bash
   cat > ~/.local/bin/claude-qwen <<'EOS'
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
   EOS
   chmod +x ~/.local/bin/claude-qwen
   ```

2. Make sure that your shell finds the script: `command -v claude-qwen` must print its path. If the result is empty,
   add `~/.local/bin` to your `PATH`.

The script sets these values:

| Setting | Function |
|---|---|
| `ANTHROPIC_BASE_URL` | Claude Code sends all model requests to the colinfer server. The variable of the script replaces the one of the settings file for this session. |
| `ANTHROPIC_AUTH_TOKEN` | The key that Claude Code sends to the server. It is the content of `~/.config/colinfer/api.key` if this file exists, else `colinfer`. Without a key, the server ignores this value. You must set it: if you do not, Claude Code sends the token of your claude.ai login to the server. |
| `ANTHROPIC_MODEL` | The model name of the session. The server accepts all model names. |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Subagents and background tasks can use the model aliases `opus`, `sonnet` and `haiku`. These three lines give them the same model name. |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Removes a block at the start of the system prompt. This block is different in each conversation. Without it, a new session can use the prefix cache of the server. The gateway removes the block itself. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | Stops telemetry, error reports and automatic updates. |
| `--permission-mode acceptEdits` | Claude Code edits files without a question. It asks before it runs most commands. Do not use auto mode (section 4). |

To give the server an API key, make a random key, start the server with it, and start Claude Code with `claude-qwen`:

```bash
mkdir -p ~/.config/colinfer
(umask 077 && openssl rand -hex 16 > ~/.config/colinfer/api.key)
COLINFER_API_KEY="$(cat ~/.config/colinfer/api.key)" uv run python -m engine.server
```

With a key, the `/v1/` endpoints of the server reject requests without the key. `/health` and `/metrics` stay open.
The gateway of section 2 cannot reach a server with a key.

## 10. Troubleshooting

| Symptom | Possible cause | Remedy |
|---|---|---|
| Every row answers `API Error: Connection refused ... (ECONNREFUSED)`. | The gateway does not run. | `systemctl --user start colinfer-gateway`. |
| A Qwen row answers `the colinfer server ... does not answer`. | The server does not run, or its startup is not complete. | `systemctl --user start colinfer`, then wait for the `listening on` line in its log. The Claude rows work meanwhile. |
| Every row answers `401`. | `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` is set, so Claude Code sends that token instead of your login. | Remove the variable from `~/.claude/settings.json` and your shell, then reload the window. |
| The Qwen rows are not in the model picker. | The settings file is not valid JSON, or the window was not reloaded. | Run `jq . ~/.claude/settings.json`. Fix the error. Reload the window. |
| Claude Code shows `Failed to authenticate. API Error: 401 invalid or missing API key`. | The server uses an API key, and Claude Code sent a different key (section 9). | Start the server and Claude Code with the same key file, or run the server without a key. |
| Each command waits 15 to 35 seconds before it runs. | Auto mode is on. | Set the permission mode to `acceptEdits` or `default` (section 4). |
| Claude Code shows `Prompt is too long`. | The conversation is longer than a server slot. | Type `/compact`, or start a new chat. |
| The WebSearch tool fails with `API Error: 400 this server does not run server tools (web_search_20250305)`. | Web search is not available on a Qwen row (section 6). | Give the information to the model in your request, or pick a Claude row. |
| The first answer of a long new conversation is slow. | The server reads the full prompt. | No action is necessary. Refer to section 6. |
| The server stops with `engine failure`. | A CUDA error occurred. | The service restarts automatically. Ask again after a minute. |

## 11. Run tasks while you sleep (optional)

The night shift runs a list of Claude Code tasks against the server while the machine idles. Each task works in its
own git worktree, on a branch `night/<date>/<name>`, and a report lists the branches to review in the morning.
`docs/night_shift.md` is the manual. The short version:

1. Copy `deploy/night_shift.example.toml` to `~/.config/colinfer/night_shift.toml`. Write one `[[task]]` per job: the
   repository, the prompt, and the commands the task may run, for example `allow = ["Bash(uv run pytest*)"]`.

2. Check the file, then run one task now:

   ```bash
   uv run python tools/night_shift.py --dry-run
   uv run python tools/night_shift.py --only <task name>
   ```

3. Schedule the nights (01:00). From the root of the clone:

   ```bash
   systemctl --user link "$PWD/deploy/colinfer-night-shift.service" "$PWD/deploy/colinfer-night-shift.timer"
   systemctl --user enable --now colinfer-night-shift.timer
   ```

4. In the morning, open `~/.cache/colinfer/night_shift/<date>/report.md`. It lists each task, its status and its
   branch, with the commands to review the branch and to resume the task's session.

A task edits files in its worktree and runs the commands you allowed. Anything else that would ask a question is
denied, so nothing waits at night. Do not allow a task to run the checks that load the model next to the server
(`docs/night_shift.md`, section 6).

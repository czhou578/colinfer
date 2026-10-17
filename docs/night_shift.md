# Run Claude Code tasks at night

The night shift runs a list of Claude Code tasks against the colinfer server while the machine idles. Each task works
in its own git worktree, on its own branch. In the morning, a report lists the branches to review.

The server runs one request at a time, so the tasks run one after another. A machine that serves one person is idle
most of the day, and a task of 30 to 60 model turns takes 5 to 10 minutes, so one night holds dozens of tasks.

> **NOTE:** Qwen3.8-27B is a smaller model than Claude. Give it small and clear tasks: add docstrings, run a test
> suite and fix what is simple, apply a lint, write a small function with its tests. Examine each branch before you
> merge it. A task that needs judgment over many steps, such as a flaky test with an unclear cause, often ends in a
> note rather than a fix.

## The short version

1. Copy `deploy/night_shift.example.toml` to `~/.config/colinfer/night_shift.toml` and write one `[[task]]` per job
   (section 2).
2. `uv run python tools/night_shift.py --dry-run`, then `--only <name>` for one task now (section 3).
3. Link and enable `deploy/colinfer-night-shift.timer` for 01:00 every night (section 4).
4. In the morning, read `~/.cache/colinfer/night_shift/<date>/report.md` and review the branches it lists (section 5).

## How the parts connect

```
systemd timer (01:00)  -->  tools/night_shift.py  --claude -p-->  Claude Code  --Anthropic Messages API-->  colinfer server
                            one task at a time                   in a worktree                           127.0.0.1:8000
                            report.md                            of the task's repository
```

## 1. Requirements

- The colinfer server, as a systemd service (`docs/server.md`, section "systemd"). The night shift starts it when it is
  not running, and waits for it.
- The Claude Code CLI (`claude` on your `PATH`). The night shift sets the server address and the model itself, so
  the gateway and the settings of `docs/claude_code.md` are not needed for it. Make sure that one request works once
  before you plan a night: `claude --model qwen3.8-27b -p "say ready"` with the gateway, or `claude-qwen -p "say ready"`
  (`docs/claude_code.md`, section 9).
- A task file, `~/.config/colinfer/night_shift.toml` (section 2).

## 2. Make the task file

Copy `deploy/night_shift.example.toml` to `~/.config/colinfer/night_shift.toml` and edit it. The file has a `[defaults]`
table and one `[[task]]` table per task. A task takes any key of the defaults, and `allow` and `env` add to the
defaults instead of replacing them.

| Key | Default | |
|---|---|---|
| `name` | | The branch and directory name of the task: `night/<date>/<name>`. |
| `prompt`, `prompt_file` | | What the task does. One of the two. A path is relative to the task file. |
| `repo` | | The git repository the task works in. |
| `base` | `HEAD` | The commit the branch starts from: `HEAD`, `main`, `origin/main`, a hash. |
| `timeout` | `30m` | The wall-clock limit of the task. What the task changed before the limit stays on its branch. |
| `allow` | `[]` | The commands the task may run, as Claude Code permission rules: `Bash(uv run pytest*)`. Read-only git (`git status`, `diff`, `log`, `show`) is always allowed. Edits inside the worktree are always allowed. |
| `env` | `{}` | Environment variables of the claude process and of the commands it runs. |
| `effort` | | The reasoning effort Claude Code asks for: `low`, `medium`, `high`, `xhigh`, `max`. Without it, the server's `--reasoning-effort` applies. |
| `model` | `qwen3.8-27b` | The model name Claude Code sends. The server accepts any name. |
| `bare` | `false` | `claude --bare`: no hooks, plugins, auto-memory or CLAUDE.md. |
| `until` | | Local time, `HH:MM`. No task starts after it. |
| `min_free_gb` | `16` | A task starts only with this much memory available. |
| `kill_below_gb` | `8` | A running task is killed when the available memory falls below this. |
| `wait_server` | `10m` | How long a task waits for the server to answer `/health`. |
| `server` | `http://127.0.0.1:8000` | |
| `state_dir` | `~/.cache/colinfer/night_shift` | Where each run keeps its worktrees, logs and report. |
| `preamble` | (see `tools/night_shift.py`) | Text put before every prompt: the task runs unattended, assume and say so, keep the change small, end with a summary. `""` for none. |

Every prompt gets the API key of the server from `~/.config/colinfer/api.key` when that file exists
(`docs/claude_code.md`, section 6).

> **NOTE:** For a project that uses `uv`, set `UV_PROJECT_ENVIRONMENT` in `[defaults.env]` to the project's existing
> `.venv`. Otherwise, the first `uv run` in a worktree makes a new environment there.

## 3. Try a task in the daytime

1. Check the file, the repositories and the server:

   ```bash
   uv run python tools/night_shift.py --dry-run
   ```

   The output shows each task with its repository, base commit, timeout and allowed commands, then the state of the
   server and the memory.

2. Run one task now:

   ```bash
   uv run python tools/night_shift.py --only docstrings-utils
   ```

   The runner prints one line when the task starts and one when it ends, then the path of the report.

Each run makes a directory `~/.cache/colinfer/night_shift/<date>/`, with `report.md` and one directory per task:
`tree/` (the worktree), `prompt.md`, `claude.jsonl` (every event of the Claude Code session), `stderr.txt` and
`result.json`.

## 4. Schedule the nights

The units in `deploy/` run the task file at 01:00. Their paths assume that the clone is
`~/Projects/colin-inference-engine`, like `deploy/colinfer.service`. From the root of the clone:

```bash
systemctl --user link "$PWD/deploy/colinfer-night-shift.service" "$PWD/deploy/colinfer-night-shift.timer"
systemctl --user enable --now colinfer-night-shift.timer
systemctl --user list-timers colinfer-night-shift    # the next run
journalctl --user -u colinfer-night-shift -f         # the log of a run
```

`systemctl --user start colinfer-night-shift` runs the file now. `systemctl --user stop colinfer-night-shift` kills the
running task, commits what it changed, writes the report and starts no other task.

Set `until` so that the machine is free when you start work. A task that runs at that time still gets its timeout.

## 5. The morning

Open `~/.cache/colinfer/night_shift/<date>/report.md`. It has a table of the tasks, then one section per task with the
last message of the model and the commands to review its branch:

```
| # | Task | Status | Time | Turns | Branch | Changes |
|---|---|---|---|---|---|---|
| 1 | docstrings-utils | done | 6 min | 23 | `night/2026-10-14/docstrings-utils` | 1 commit, 3 files changed, 120 insertions(+), 4 deletions(-) |
| 2 | flaky-test-eviction | timeout (after 45m) | 45 min | 61 | `night/2026-10-14/flaky-test-eviction` | 2 commits, ... |
| 3 | issue-42 | no changes | 2 min | 7 | | |
```

| Status | Meaning |
|---|---|
| `done` | Claude Code finished, and the branch holds the changes. |
| `no changes` | It finished without a change. The branch and the worktree were removed. |
| `timeout` | The task hit its limit. What it had changed is committed on its branch. |
| `error` | Claude Code exited with an error. The detail holds the exit code and the end of its stderr. |
| `killed` | The available memory fell below `kill_below_gb`. |
| `skipped` | The task did not start: past `until`, the server did not answer, or not enough memory. |

To review a branch:

```bash
git -C ~/Projects/my-project log --oneline main..night/2026-10-14/docstrings-utils
git -C ~/Projects/my-project diff main night/2026-10-14/docstrings-utils
git -C ~/Projects/my-project merge night/2026-10-14/docstrings-utils      # or cherry-pick, or rebase
```

To continue a task's conversation, go to its worktree and resume the session that the report names:

```bash
cd ~/.cache/colinfer/night_shift/2026-10-14/docstrings-utils/tree && claude-qwen --resume <session id>
```

The worktrees stay until you remove them. The branches are in the repository, so the worktrees are only needed to
resume a session or to run the code in place:

```bash
uv run python tools/night_shift.py --prune ~/.cache/colinfer/night_shift/2026-10-14
```

## 6. What a task can do, and what protects the machine

**Permissions.** Claude Code runs in the `acceptEdits` mode with `--permission-prompts none`: it edits files in its
worktree without a question, it runs the commands of the task's `allow` list, and every other action that would ask a
question is denied at once. The model sees the denial and goes on. WebSearch and WebFetch are off, since the server
has no web search. The night shift never uses `--dangerously-skip-permissions`. If you want a task to run any command,
run the night shift in a container.

**The commands you allow.** A rule such as `Bash(uv run pytest*)` lets the model run any command that starts with
`uv run pytest`. Allow the test runner, the linter and the build. Do not allow `rm`, `git push`, package installation,
or anything that reaches outside the worktree.

> **CAUTION:** Do not allow a task to run anything that loads the model next to the server. `tests/check.sh` without
> `--quick`, `tests/golden.py`, `tests/perplexity.py` and an HF reference run each take 40 GB or more. With the server's
> 61 GB, this is within 20 GB of the memory of the machine, and an over-commit powers the machine off. The
> `kill_below_gb` guard kills the task when the available memory falls under its limit, but a single large allocation
> can get past it. For this repository, allow `tests/check.sh --quick` and `uv run pytest tests/` only.

**Memory.** A task starts only with `min_free_gb` available (`MemAvailable`; the GPU allocations of the server count,
since the memory is unified). A running task is killed below `kill_below_gb`.

**Time.** Each task has a `timeout`, and no task starts after `until`.

**The server.** Run the server from a committed checkout, as the systemd service does, not from a tree with changes
you are still working on. The night shift does not restart the server. If the server fails, systemd restarts it, and
the next task waits up to `wait_server` for it.

**Branches.** A task never touches the branch you work on. It starts from the `base` commit on a new branch in a
separate worktree. A task that changes nothing leaves no branch.

## 7. Troubleshooting

| Symptom | Possible cause | Remedy |
|---|---|---|
| Every task is `skipped: the server ... does not answer`. | The server is not running, or took longer than `wait_server` to start. | `systemctl --user status colinfer`. Start it, or raise `wait_server`. |
| A task is `error: claude: not found`. | `claude` is not on the PATH of the service. | The unit sets `PATH=%h/.local/bin:...`. If Claude Code is elsewhere, set `claude = "/path/to/claude"` in `[defaults]`. |
| A task ends in seconds with `no changes`, and the log shows many denied tools. | The task needs a command that `allow` does not name. | Look at `claude.jsonl` for the denied commands. Add a rule for them. |
| A task is `error` and its stderr shows `401`. | The server runs with an API key, and the task file does not point at the key file. | Set `key_file` in `[defaults]`, or keep the key in `~/.config/colinfer/api.key`. |
| The report shows `timeout` for most tasks. | The tasks are too large for the time, or the model loops. | Split the tasks. Look at the last turns in `claude.jsonl`. A tighter prompt with a named file and a named check helps. |
| The branch holds a `.venv` directory or build output. | The repository does not ignore them. | Add them to `.gitignore`, or set `UV_PROJECT_ENVIRONMENT` in `[defaults.env]`. |

#!/usr/bin/env python3
"""Night shift: a queue of Claude Code tasks that run headless against the colinfer server while the machine idles
(docs/night_shift.md).

A task is a prompt for one git repository. For each task, in order, the runner:
  1. makes a git worktree of the repository on a new branch night/<date>/<name>, from the task's base commit;
  2. runs `claude -p` in it against the local model (the settings of docs/claude_code.md), with a wall-clock timeout;
  3. commits what the task left uncommitted, or removes the branch when nothing changed;
  4. adds the task to the report of the run: status, branch, files changed, the model's last message.
Tasks run one after another, as the server runs one request at a time. The morning starts with the report.

What a task may do: edit files in its worktree (the acceptEdits mode), and run the commands that its `allow` list
names, such as `Bash(uv run pytest*)`, plus the read-only git commands of ALLOW. Anything else that would ask a question
is denied (--permission-prompts none), so nothing waits for an answer at night. WebSearch and WebFetch are off: the
local server has no web search, and a denied tool costs no turn.

Guards. A task starts only when the server answers /health and at least `min_free_gb` of memory is available. It is
killed when the available memory falls below `kill_below_gb`: an over-commit of the unified memory powers the machine
off instead of raising an error. No task starts after `until`.

   uv run python tools/night_shift.py --dry-run             # read the task file; check the repositories and the server
   uv run python tools/night_shift.py --only docstrings     # one task, now
   uv run python tools/night_shift.py                       # every task (deploy/colinfer-night-shift.timer runs this at 01:00)
   uv run python tools/night_shift.py --prune ~/.cache/colinfer/night_shift/2026-10-14   # remove the worktrees of a run
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

TASKS_FILE = os.path.expanduser("~/.config/colinfer/night_shift.toml")
KEY_FILE = os.path.expanduser("~/.config/colinfer/api.key")  # the server's API key, as docs/claude_code.md section 6 makes it
STATE_DIR = os.path.expanduser("~/.cache/colinfer/night_shift")
ALLOW = ("Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git show*)")  # read-only git: every task may run these
DENY = ("WebSearch", "WebFetch")
KILL_GRACE_S = 10.0    # SIGTERM to a task's process group, then SIGKILL this much later
MESSAGE_CHARS = 1500   # of the model's last message kept in the report
PREAMBLE = ("You run unattended at night. Nobody answers questions, so make reasonable assumptions and say which ones you made. "
            "Keep the change small and complete, run the checks you are allowed to run, and do not touch files outside this "
            "repository. End with a short summary of what you changed and what you could not do.")


# ------------------------------------------------------------------------------------------------------- configuration
@dataclasses.dataclass
class Settings:
    """The run: the [defaults] keys of the task file that are not task keys."""
    server: str = "http://127.0.0.1:8000"
    model: str = "qwen3.8-27b"              # the model name Claude Code sends; the server accepts any name
    state_dir: str = STATE_DIR              # <state_dir>/<date>/<task>/{tree, claude.jsonl, stderr.txt, result.json}
    until: str | None = None                # HH:MM, local time: no task starts after it
    min_free_gb: float = 16.0               # memory available (MemAvailable) for a task to start
    kill_below_gb: float = 8.0              # a running task is killed below this
    wait_server: str = "10m"                # how long a task waits for /health (the service may be restarting)
    preamble: str = PREAMBLE                # put before every prompt ("" for none)
    claude: str = "claude"                  # the Claude Code executable
    key_file: str = KEY_FILE
    poll_s: float = 1.0                     # the watch interval of a running task

    @property
    def wait_server_s(self) -> float:
        return parse_duration(self.wait_server)


@dataclasses.dataclass
class Task:
    name: str
    prompt: str
    repo: str
    base: str = "HEAD"                       # the commit the branch starts from
    timeout: str = "30m"
    allow: list[str] = dataclasses.field(default_factory=list)   # permission rules, as in Claude Code's settings
    deny: list[str] = dataclasses.field(default_factory=lambda: list(DENY))
    env: dict[str, str] = dataclasses.field(default_factory=dict)  # extra environment of the claude process (and its commands)
    effort: str | None = None                # the reasoning effort Claude Code asks for (low, medium, high, xhigh, max)
    bare: bool = False                       # claude --bare: no hooks, plugins, auto-memory or CLAUDE.md discovery
    model: str | None = None                 # else Settings.model

    @property
    def timeout_s(self) -> float:
        return parse_duration(self.timeout)


TASK_KEYS = {f.name for f in dataclasses.fields(Task)} | {"prompt_file"}
SETTING_KEYS = {f.name for f in dataclasses.fields(Settings)} - {"poll_s"}


def parse_duration(s) -> float:
    """'90', '90s', '20m', '1.5h' -> seconds."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(s))
    if not m:
        raise ValueError(f"not a duration: {s!r} (use 90s, 20m or 1.5h)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def safe_name(name: str) -> str:
    """A task name as a branch and directory component."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.")
    if not s:
        raise ValueError(f"task name {name!r} has no usable characters")
    return s


def load(path: str) -> tuple[Settings, list[Task]]:
    """The task file: [defaults] (settings and task defaults) and [[task]] tables. A task's `allow` adds to the
    defaults' and its `env` overrides the defaults' key by key; the other keys replace the default."""
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    defaults = doc.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("[defaults] must be a table")
    unknown = set(defaults) - SETTING_KEYS - TASK_KEYS
    if unknown:
        raise ValueError(f"[defaults]: unknown keys {sorted(unknown)}")
    settings = Settings(**{k: v for k, v in defaults.items() if k in SETTING_KEYS})
    base_dir = os.path.dirname(os.path.abspath(path))
    tasks, names = [], set()
    for i, t in enumerate(doc.get("task", [])):
        if not isinstance(t, dict):
            raise ValueError(f"task {i + 1} must be a table")
        unknown = set(t) - TASK_KEYS
        if unknown:
            raise ValueError(f"task {i + 1}: unknown keys {sorted(unknown)}")
        kw = {k: v for k, v in defaults.items() if k in TASK_KEYS and k not in ("prompt", "prompt_file", "name")}
        kw.update({k: v for k, v in t.items() if k not in ("allow", "env", "prompt_file")})
        kw["allow"] = list(defaults.get("allow", [])) + list(t.get("allow", []))
        kw["env"] = dict(defaults.get("env", {})) | dict(t.get("env", {}))
        if "prompt_file" in t:
            if "prompt" in t:
                raise ValueError(f"task {i + 1}: prompt and prompt_file are exclusive")
            with open(os.path.join(base_dir, os.path.expanduser(t["prompt_file"]))) as f:
                kw["prompt"] = f.read()
        for k in ("name", "prompt", "repo"):
            if not kw.get(k):
                raise ValueError(f"task {i + 1}: {k} is required")
        kw["name"] = safe_name(kw["name"])
        kw["repo"] = os.path.expanduser(kw["repo"])
        if kw["name"] in names:
            raise ValueError(f"task name {kw['name']!r} is used twice")
        names.add(kw["name"])
        parse_duration(kw.get("timeout", Task.timeout))  # a bad duration fails now, not at night
        tasks.append(Task(**kw))
    parse_duration(settings.wait_server)
    if settings.until is not None and not re.fullmatch(r"\d{1,2}:\d{2}", str(settings.until)):
        raise ValueError(f"until must be HH:MM, not {settings.until!r}")
    settings.state_dir = os.path.expanduser(settings.state_dir)
    settings.key_file = os.path.expanduser(settings.key_file)
    return settings, tasks


# -------------------------------------------------------------------------------------------------------------- guards
def mem_available_gb() -> float:
    """MemAvailable of /proc/meminfo. GPU allocations come from the same unified memory, so this falls as the server or
    another CUDA process grows."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return float("inf")


def server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def until_time(start: dt.datetime, hhmm: str | None) -> dt.datetime | None:
    """The first HH:MM after the start: tonight's 07:00 for a run that starts at 01:00, tomorrow's for one at 23:30."""
    if not hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    t = start.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > start else t + dt.timedelta(days=1)


# ----------------------------------------------------------------------------------------------------------------- git
def git(repo: str, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def is_repo(path: str) -> bool:
    return os.path.isdir(path) and subprocess.run(["git", "-C", path, "rev-parse", "--git-dir"], capture_output=True).returncode == 0


def unique_branch(repo: str, name: str) -> str:
    """name, or name-2, name-3, ... when the branch exists (the same task on the same date)."""
    cand, i = name, 1
    while subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "--quiet", "refs/heads/" + cand], capture_output=True).returncode == 0:
        i += 1
        cand = f"{name}-{i}"
    return cand


def unique_dir(path: str) -> str:
    cand, i = path, 1
    while os.path.exists(cand):
        i += 1
        cand = f"{path}-{i}"
    return cand


def commit_all(tree: str, message: str) -> bool:
    """Commits everything the task left uncommitted (gitignored files stay out). False when there was nothing."""
    if not git(tree, "status", "--porcelain", "--untracked-files=all"):
        return False
    git(tree, "add", "-A")
    r = subprocess.run(["git", "-C", tree, "commit", "-q", "-m", message], capture_output=True, text=True)
    if r.returncode != 0:  # a repository without a configured identity
        git(tree, "-c", "user.name=night shift", "-c", "user.email=night-shift@localhost", "commit", "-q", "-m", message)
    return True


# -------------------------------------------------------------------------------------------------------- Claude Code
def claude_env(settings: Settings, task: Task) -> dict[str, str]:
    """The environment of docs/claude_code.md section 3: every model request goes to the server, with its key (or a
    placeholder, which the server ignores without --api-key; without it, Claude Code would send the claude.ai login)."""
    try:
        with open(settings.key_file) as f:
            key = f.read().strip() or "colinfer"
    except OSError:
        key = "colinfer"
    model = task.model or settings.model
    env = dict(os.environ)
    env.update(ANTHROPIC_BASE_URL=settings.server, ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_API_KEY=key, ANTHROPIC_MODEL=model,
               ANTHROPIC_DEFAULT_OPUS_MODEL=model, ANTHROPIC_DEFAULT_SONNET_MODEL=model, ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
               CLAUDE_CODE_ATTRIBUTION_HEADER="0", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1")
    env.update({k: os.path.expanduser(str(v)) for k, v in task.env.items()})
    return env


def claude_argv(settings: Settings, task: Task, prompt: str) -> list[str]:
    """Print mode with the events of the run as JSON lines. The permission rules go in one --settings argument, so no
    option takes a list that could swallow the prompt. --permission-prompts none: whatever would ask is denied."""
    perms = {"permissions": {"allow": list(ALLOW) + list(task.allow), "deny": list(task.deny)}}
    argv = [settings.claude, "-p", prompt, "--output-format", "stream-json", "--verbose", "--permission-mode", "acceptEdits",
            "--permission-prompts", "none", "--settings", json.dumps(perms), "--model", task.model or settings.model]
    if task.effort:
        argv += ["--effort", task.effort]
    if task.bare:
        argv.append("--bare")
    return argv


def parse_stream(path: str) -> dict:
    """What the report needs from the stream-json events: the session id, turns, the tools used, the final message."""
    info = {"session": None, "turns": None, "ms": None, "message": "", "is_error": None, "subtype": None, "tools": collections.Counter()}
    try:
        f = open(path, errors="replace")
    except OSError:
        return info
    with f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            t = ev.get("type")
            if t == "system" and ev.get("session_id"):
                info["session"] = ev["session_id"]
            elif t == "assistant":
                msg = ev.get("message")
                for b in (msg.get("content") if isinstance(msg, dict) else None) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        info["tools"][str(b.get("name") or "?")] += 1
            elif t == "result":
                info.update(session=ev.get("session_id") or info["session"], turns=ev.get("num_turns"), ms=ev.get("duration_ms"),
                            message=str(ev.get("result") or ""), is_error=bool(ev.get("is_error")), subtype=ev.get("subtype"))
    return info


def kill_group(p: subprocess.Popen):
    """SIGTERM to the task's process group (claude and the commands it runs), SIGKILL after KILL_GRACE_S."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.monotonic()
    while p.poll() is None and time.monotonic() - t0 < KILL_GRACE_S:
        time.sleep(0.1)
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


# ----------------------------------------------------------------------------------------------------------------- run
@dataclasses.dataclass
class Result:
    task: Task
    status: str                 # done | no changes | timeout | error | killed | stopped | skipped
    detail: str = ""
    branch: str | None = None   # kept only when the branch holds commits
    tree: str | None = None
    base_commit: str | None = None
    commits: int = 0
    diffstat: str = ""
    seconds: float = 0.0
    turns: int | None = None
    tools: dict = dataclasses.field(default_factory=dict)
    session: str | None = None
    message: str = ""           # the model's last message
    dir: str = ""               # the task's directory of the run (prompt, events, stderr, result.json)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["task"] = {"name": self.task.name, "repo": self.task.repo, "base": self.task.base}
        return d


class Shift:
    """One run of the queue: a directory under state_dir named by the date, a subdirectory per task, a report."""

    def __init__(self, settings: Settings, tasks: list[Task], log=None):
        self.settings, self.tasks = settings, tasks
        self.log = log or (lambda msg: print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True))
        self.start = dt.datetime.now()
        self.date = self.start.strftime("%Y-%m-%d")
        self.run_dir = unique_dir(os.path.join(settings.state_dir, self.date))
        self.until = until_time(self.start, settings.until)
        self.results: list[Result] = []
        self.stop = False  # set by SIGTERM / SIGINT: the running task is killed and no other starts

    def request_stop(self, *_):
        """SIGTERM / SIGINT: the watch loop kills the running task within poll_s and records it as stopped."""
        self.stop = True

    # ................................................................................................... the queue
    def run(self) -> list[Result]:
        os.makedirs(self.run_dir)
        self.log(f"night shift: {len(self.tasks)} tasks, report in {self.run_dir}/report.md"
                 + (f", no task starts after {self.until:%H:%M}" if self.until else ""))
        for task in self.tasks:
            if self.stop:
                r = Result(task, "skipped", "the run was stopped")
            elif self.until is not None and dt.datetime.now() >= self.until:
                r = Result(task, "skipped", f"past {self.settings.until}")
            else:
                r = self.run_task(task)
            self.results.append(r)
            self.log(f"[{task.name}] {r.status}" + (f": {r.detail}" if r.detail else "") + (f" -> {r.branch}" if r.branch else ""))
            self.write_report()
        self.log(f"night shift: done, {self.summary()}")
        return self.results

    def wait_server(self) -> bool:
        s = self.settings
        t0 = time.monotonic()
        while True:
            if server_up(s.server):
                return True
            if self.stop or time.monotonic() - t0 >= s.wait_server_s:
                return False
            time.sleep(min(15.0, max(s.poll_s, 0.1)))

    def run_task(self, task: Task) -> Result:
        s = self.settings
        d = os.path.join(self.run_dir, task.name)
        os.makedirs(d, exist_ok=True)
        r = Result(task, "skipped", dir=d)
        if not self.wait_server():
            r.detail = f"the server at {s.server} does not answer"
            return self._finish(r)
        free = mem_available_gb()
        if free < s.min_free_gb:
            r.detail = f"{free:.0f} GB of memory available, {s.min_free_gb:.0f} needed"
            return self._finish(r)
        if not is_repo(task.repo):
            return self._finish(dataclasses.replace(r, status="error", detail=f"{task.repo} is not a git repository"))
        try:
            r.base_commit = git(task.repo, "rev-parse", "--verify", "--quiet", task.base + "^{commit}")
        except RuntimeError:
            return self._finish(dataclasses.replace(r, status="error", detail=f"unknown base {task.base!r} in {task.repo}"))
        r.branch = unique_branch(task.repo, f"night/{self.date}/{task.name}")
        r.tree = os.path.join(d, "tree")
        git(task.repo, "worktree", "add", "-q", "-b", r.branch, r.tree, r.base_commit)
        prompt = (s.preamble + "\n\n" + task.prompt.strip()) if s.preamble else task.prompt.strip()
        with open(os.path.join(d, "prompt.md"), "w") as f:
            f.write(prompt + "\n")
        self.log(f"[{task.name}] start: {task.repo} at {r.base_commit[:9]}, timeout {task.timeout}, branch {r.branch}")
        events, errs = os.path.join(d, "claude.jsonl"), os.path.join(d, "stderr.txt")
        t0 = time.monotonic()
        rc, reason = self._run_claude(claude_argv(s, task, prompt), claude_env(s, task), r.tree, events, errs, task.timeout_s)
        r.seconds = time.monotonic() - t0
        info = parse_stream(events)
        r.turns, r.session, r.tools, r.message = info["turns"], info["session"], dict(info["tools"]), info["message"][:MESSAGE_CHARS]
        if reason == "memory":
            r.status, r.detail = "killed", f"memory available fell below {s.kill_below_gb:.0f} GB"
        elif reason == "timeout":
            r.status, r.detail = "timeout", f"after {task.timeout}"
        elif reason == "stopped":
            r.status, r.detail = "stopped", "the run was stopped"
        elif rc is None:
            r.status, r.detail = "error", f"{s.claude}: not found"
        elif rc != 0 or info["is_error"]:
            tail = ""
            try:
                with open(errs, errors="replace") as f:
                    tail = f.read()[-400:].strip()
            except OSError:
                pass
            r.status, r.detail = "error", (f"exit {rc}" if rc else str(info["subtype"] or "is_error")) + (f": {tail}" if tail else "")
        else:
            r.status = "done"
        # whatever the task left uncommitted goes on the branch too (a timeout can still leave reviewable work)
        commit_all(r.tree, f"night shift: {task.name}\n\n{task.prompt.strip()}\n\nStatus: {r.status}" + (f" ({r.detail})" if r.detail else ""))
        r.commits = int(git(task.repo, "rev-list", "--count", f"{r.base_commit}..{r.branch}"))
        if r.commits == 0:  # nothing to review: no worktree and no branch to clean up in the morning
            git(task.repo, "worktree", "remove", "--force", r.tree)
            git(task.repo, "branch", "-D", r.branch)
            r.branch = r.tree = None
            if r.status == "done":
                r.status = "no changes"
        else:
            r.diffstat = git(task.repo, "diff", "--shortstat", r.base_commit, r.branch)
        return self._finish(r)

    def _run_claude(self, argv, env, cwd, out_path, err_path, timeout_s) -> tuple[int | None, str | None]:
        """(exit code, or None when claude is missing; why it was killed: timeout | memory | stopped | None)."""
        s = self.settings
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            try:
                p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            except FileNotFoundError:
                return None, None
            t0, reason = time.monotonic(), None
            while p.poll() is None:
                if self.stop:
                    reason = "stopped"
                elif time.monotonic() - t0 > timeout_s:
                    reason = "timeout"
                elif mem_available_gb() < s.kill_below_gb:
                    reason = "memory"
                if reason:
                    kill_group(p)
                    break
                time.sleep(s.poll_s)
            return p.wait(), reason

    def _finish(self, r: Result) -> Result:
        if r.dir:
            with open(os.path.join(r.dir, "result.json"), "w") as f:
                json.dump(r.to_json(), f, indent=1)
        return r

    # .................................................................................................. the report
    def summary(self) -> str:
        c = collections.Counter(r.status for r in self.results)
        return ", ".join(f"{n} {k}" for k, n in c.items()) or "no tasks"

    @staticmethod
    def fmt_seconds(s: float) -> str:
        return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"

    def write_report(self):
        with open(os.path.join(self.run_dir, "report.md"), "w") as f:
            f.write(self.report())

    def report(self) -> str:
        home = os.path.expanduser("~")
        tilde = lambda p: ("~" + p[len(home):]) if p.startswith(home + "/") else p  # noqa: E731
        L = [f"# Night shift {self.date}", "",
             f"Started {self.start:%H:%M:%S}, last update {dt.datetime.now():%H:%M:%S}. "
             f"{len(self.results)} of {len(self.tasks)} tasks: {self.summary()}.",
             "", "| # | Task | Status | Time | Turns | Branch | Changes |", "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(self.results, 1):
            changes = f"{r.commits} commit{'s' if r.commits != 1 else ''}, {r.diffstat.strip()}" if r.commits else ""
            secs = self.fmt_seconds(r.seconds) if r.seconds else ""
            turns = r.turns if r.turns is not None else ""
            L.append(f"| {i} | {r.task.name} | {r.status} | {secs} | {turns} | {'`' + r.branch + '`' if r.branch else ''} | {changes} |")
        for i, r in enumerate(self.results, 1):
            L += ["", f"## {i}. {r.task.name}: {r.status}" + (f" ({r.detail})" if r.detail else ""), ""]
            line = f"Repository `{tilde(r.task.repo)}`, from `{r.task.base}`" + (f" ({r.base_commit[:9]})" if r.base_commit else "") + "."
            if r.tools:
                line += " Tools: " + ", ".join(f"{k} {v}" for k, v in sorted(r.tools.items(), key=lambda kv: -kv[1])) + "."
            L.append(line)
            if r.message:
                L += [""] + ["> " + ln for ln in r.message.strip().splitlines()]
            if r.branch:
                L += ["", "Review:", "```",
                      f"git -C {tilde(r.task.repo)} log --oneline {r.base_commit[:9]}..{r.branch}",
                      f"git -C {tilde(r.task.repo)} diff {r.base_commit[:9]} {r.branch}"]
                if r.session:
                    L.append(f"cd {tilde(r.tree)} && claude-qwen --resume {r.session}   # continue the task's session (docs/claude_code.md)")
                L.append("```")
            if r.dir:
                L += ["", f"Log: `{tilde(r.dir)}/claude.jsonl`"]
        return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------ commands
def check(settings: Settings, tasks: list[Task], log=print) -> int:
    """--dry-run: what a run would do, and what is wrong with it. Returns the number of problems."""
    problems = 0
    for t in tasks:
        base = None
        if not is_repo(t.repo):
            log(f"  {t.name}: {t.repo} is not a git repository")
            problems += 1
        else:
            try:
                base = git(t.repo, "rev-parse", "--verify", "--quiet", t.base + "^{commit}")
            except RuntimeError:
                log(f"  {t.name}: unknown base {t.base!r} in {t.repo}")
                problems += 1
        log(f"  {t.name}: {t.repo} at {t.base}" + (f" ({base[:9]})" if base else "") + f", timeout {t.timeout}, "
            f"branch night/<date>/{t.name}\n    allow: {', '.join(list(ALLOW) + t.allow)}")
    up = server_up(settings.server)
    log(f"server {settings.server}: {'up' if up else 'DOWN (a run waits up to ' + settings.wait_server + ' for it)'}")
    log(f"memory available: {mem_available_gb():.0f} GB (a task needs {settings.min_free_gb:.0f}, is killed below {settings.kill_below_gb:.0f})")
    found = shutil.which(settings.claude)
    log(f"claude: {found or settings.claude + ' (NOT FOUND on PATH)'}")
    return problems + (0 if found else 1)


def prune(run_dir: str, log=print) -> int:
    """Removes the worktrees of a run (the branches stay). Returns how many."""
    n = 0
    for name in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, name, "result.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            r = json.load(f)
        tree, repo = r.get("tree"), r["task"]["repo"]
        if tree and os.path.isdir(tree):
            git(repo, "worktree", "remove", "--force", tree)
            log(f"removed {tree}")
            n += 1
        if is_repo(repo):
            git(repo, "worktree", "prune")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Claude Code tasks against the colinfer server, one after another (docs/night_shift.md)")
    ap.add_argument("tasks", nargs="?", default=TASKS_FILE, help=f"the task file (default {TASKS_FILE})")
    ap.add_argument("--dry-run", action="store_true", help="check the task file, the repositories and the server; run nothing")
    ap.add_argument("--only", action="append", metavar="NAME", help="run only this task (repeatable)")
    ap.add_argument("--prune", metavar="RUN_DIR", help="remove the worktrees of a finished run; the branches stay")
    ap.add_argument("--claude", help="the Claude Code executable (default: the task file's, else claude on PATH)")
    ap.add_argument("--state-dir", help=f"where runs keep their worktrees, logs and report (default {STATE_DIR})")
    a = ap.parse_args(argv)
    if a.prune:
        n = prune(a.prune)
        print(f"{n} worktrees removed")
        return 0
    try:
        settings, tasks = load(a.tasks)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
        print(f"{a.tasks}: {e}", file=sys.stderr)
        return 2
    if a.claude:
        settings.claude = a.claude
    if a.state_dir:
        settings.state_dir = os.path.expanduser(a.state_dir)
    if a.only:
        missing = set(a.only) - {t.name for t in tasks}
        if missing:
            print(f"no task named {sorted(missing)}", file=sys.stderr)
            return 2
        tasks = [t for t in tasks if t.name in a.only]
    if not tasks:
        print(f"{a.tasks}: no tasks", file=sys.stderr)
        return 2
    if a.dry_run:
        print(f"{len(tasks)} tasks in {a.tasks}:")
        return 1 if check(settings, tasks) else 0
    shift = Shift(settings, tasks)
    signal.signal(signal.SIGTERM, shift.request_stop)
    signal.signal(signal.SIGINT, shift.request_stop)
    results = shift.run()
    print(f"report: {shift.run_dir}/report.md")
    return 1 if all(r.status == "skipped" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

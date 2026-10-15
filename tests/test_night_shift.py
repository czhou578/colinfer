"""The night-shift runner (tools/night_shift.py) on a stand-in for Claude Code: a script that edits, commits, sleeps or
fails as its prompt says and prints the stream-json events of a run. A small HTTP server stands in for the colinfer
server's /health. Each test gets its own git repository."""
import datetime as dt
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import types

import pytest

from tools import night_shift as ns

FAKE_CLAUDE = f'''#!{sys.executable}
"""Claude Code in print mode, as far as the runner can tell: argv and the environment go to $FAKE_CLAUDE_LOG first (so a
killed run leaves them too), then the prompt says what happens: EDIT writes a file, COMMIT commits it, SLEEP writes a
file and sleeps, FAIL exits 2. Events come out as JSON lines."""
import json, os, subprocess, sys, time
argv = sys.argv[1:]
prompt = argv[argv.index("-p") + 1]
with open(os.environ["FAKE_CLAUDE_LOG"], "w") as f:
    json.dump({{"argv": argv, "cwd": os.getcwd(),
               "env": {{k: v for k, v in os.environ.items() if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_")) or k == "EXTRA"}}}}, f)
emit = lambda **o: print(json.dumps(o), flush=True)
emit(type="system", subtype="init", session_id="sess-1")
if "EDIT" in prompt:
    with open("hello.py", "a") as f:
        f.write("def hi():\\n    return 1\\n")
    emit(type="assistant", message={{"content": [{{"type": "tool_use", "name": "Edit", "input": {{}}}}, {{"type": "text", "text": "ok"}}]}})
    emit(type="assistant", message={{"content": [{{"type": "tool_use", "name": "Bash", "input": {{}}}}]}})
if "COMMIT" in prompt:
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-qm", "by the model"], check=True)
if "SLEEP" in prompt:
    with open("partial.txt", "w") as f:
        f.write("half done\\n")
    time.sleep(60)
if "FAIL" in prompt:
    sys.exit(2)
emit(type="result", subtype="success", is_error=False, duration_ms=1234, num_turns=3, result="I added hi().", session_id="sess-1")
'''


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def run(*args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kw).stdout.strip()


def make_repo(path):
    os.makedirs(path)
    run("git", "init", "-q", "-b", "main", path)
    run("git", "-C", path, "config", "user.name", "Test")
    run("git", "-C", path, "config", "user.email", "test@example.com")
    with open(os.path.join(path, "README.md"), "w") as f:
        f.write("hello\n")
    run("git", "-C", path, "add", "-A")
    run("git", "-C", path, "commit", "-qm", "init")
    return path


@pytest.fixture
def env(tmp_path, server):
    """A repository, the fake claude, and settings that point the runner at both (no memory guard, no server wait)."""
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    settings = ns.Settings(server=server, state_dir=str(tmp_path / "state"), claude=str(fake), min_free_gb=0, kill_below_gb=0,
                           wait_server="0s", poll_s=0.1, key_file=str(tmp_path / "no.key"))
    e = types.SimpleNamespace(repo=make_repo(str(tmp_path / "repo")), settings=settings, tmp=tmp_path, fake=str(fake))

    def task(name, prompt, **kw):
        kw.setdefault("env", {})["FAKE_CLAUDE_LOG"] = str(tmp_path / f"{name}.log.json")
        return ns.Task(name=name, prompt=prompt, repo=e.repo, timeout=kw.pop("timeout", "30s"), **kw)

    def claude_log(name):
        with open(tmp_path / f"{name}.log.json") as f:
            return json.load(f)
    e.task, e.claude_log = task, claude_log
    return e


def shift(e, tasks):
    s = ns.Shift(e.settings, tasks, log=lambda m: None)
    s.run()
    return s


def git(repo, *args):
    return run("git", "-C", repo, *args)


# ----------------------------------------------------------------------------------------------------------- the run
def test_done_and_no_changes(env):
    t1 = env.task("edit", "EDIT please", allow=["Bash(uv run pytest*)"], env={"EXTRA": "1"})
    t2 = env.task("noop", "nothing to do")
    s = shift(env, [t1, t2])
    r1, r2 = s.results
    assert r1.status == "done" and r1.branch == f"night/{s.date}/edit" and r1.commits == 1
    assert "1 file changed" in r1.diffstat and os.path.isfile(os.path.join(r1.tree, "hello.py"))
    msg = git(env.repo, "log", "-1", "--format=%B", r1.branch)
    assert msg.startswith("night shift: edit\n\nEDIT please\n\nStatus: done")
    assert r1.turns == 3 and r1.session == "sess-1" and r1.tools == {"Edit": 1, "Bash": 1} and r1.message == "I added hi()."
    assert r2.status == "no changes" and r2.branch is None and r2.tree is None
    assert not os.path.exists(os.path.join(s.run_dir, "noop", "tree"))
    assert git(env.repo, "branch", "--list", "night/*/noop") == ""
    # what claude was given: the prompt right after -p (no list option before it), the rules in one --settings argument
    log = env.claude_log("edit")
    argv = log["argv"]
    assert argv[0] == "-p" and argv[1].startswith(ns.PREAMBLE) and argv[1].endswith("EDIT please")
    assert argv[argv.index("--permission-prompts") + 1] == "none" and argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    perms = json.loads(argv[argv.index("--settings") + 1])["permissions"]
    assert perms["allow"] == list(ns.ALLOW) + ["Bash(uv run pytest*)"] and perms["deny"] == ["WebSearch", "WebFetch"]
    assert argv[argv.index("--model") + 1] == "qwen3.8-27b" and "--effort" not in argv and "--bare" not in argv
    assert os.path.realpath(log["cwd"]) == os.path.realpath(r1.tree)
    e = log["env"]
    assert e["ANTHROPIC_BASE_URL"] == env.settings.server and e["ANTHROPIC_MODEL"] == "qwen3.8-27b" and e["ANTHROPIC_AUTH_TOKEN"] == "colinfer"
    assert e["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0" and e["EXTRA"] == "1"
    # the report and the per-task record
    with open(os.path.join(s.run_dir, "report.md")) as f:
        rep = f.read()
    assert "| 1 | edit | done |" in rep and "| 2 | noop | no changes |" in rep
    assert f"git -C {env.repo} diff" in rep.replace(os.path.expanduser("~"), "~") or "git -C ~" in rep
    assert "claude-qwen --resume sess-1" in rep and "> I added hi()." in rep
    with open(os.path.join(s.run_dir, "edit", "result.json")) as f:
        assert json.load(f)["branch"] == r1.branch
    assert os.path.isfile(os.path.join(s.run_dir, "edit", "prompt.md")) and os.path.isfile(os.path.join(s.run_dir, "edit", "claude.jsonl"))


def test_options_reach_claude(env):
    env.settings.preamble = ""
    t = env.task("opts", "EDIT", effort="medium", bare=True, model="other")
    shift(env, [t])
    argv = env.claude_log("opts")["argv"]
    assert argv[1] == "EDIT" and argv[argv.index("--effort") + 1] == "medium" and "--bare" in argv and argv[argv.index("--model") + 1] == "other"
    assert env.claude_log("opts")["env"]["ANTHROPIC_MODEL"] == "other"


def test_key_file(env):
    with open(env.settings.key_file, "w") as f:
        f.write("s3cret\n")
    shift(env, [env.task("key", "EDIT")])
    e = env.claude_log("key")["env"]
    assert e["ANTHROPIC_AUTH_TOKEN"] == "s3cret" and e["ANTHROPIC_API_KEY"] == "s3cret"


def test_model_commits_itself(env):
    s = shift(env, [env.task("own", "EDIT then COMMIT")])
    r = s.results[0]
    assert r.status == "done" and r.commits == 1
    assert git(env.repo, "log", "-1", "--format=%s", r.branch) == "by the model"  # nothing was left to commit


def test_timeout_keeps_partial_work(env):
    s = shift(env, [env.task("slow", "SLEEP", timeout="1s")])
    r = s.results[0]
    assert r.status == "timeout" and r.detail == "after 1s" and r.seconds < 15
    assert r.commits == 1 and "partial.txt" in git(env.repo, "show", "--stat", "--format=", r.branch)
    assert "Status: timeout (after 1s)" in git(env.repo, "log", "-1", "--format=%B", r.branch)


def test_error_without_changes(env):
    s = shift(env, [env.task("bad", "FAIL")])
    r = s.results[0]
    assert r.status == "error" and r.detail.startswith("exit 2") and r.branch is None and r.commits == 0
    assert git(env.repo, "branch", "--list", "night/*") == ""


def test_memory_kill(env, monkeypatch):
    env.settings.kill_below_gb = 8
    first = []

    def fake_mem():
        first.append(time.monotonic())
        return 100.0 if time.monotonic() - first[0] < 0.8 else 1.0  # the start check passes; the watch loop then sees the memory gone
    monkeypatch.setattr(ns, "mem_available_gb", fake_mem)
    s = shift(env, [env.task("mem", "SLEEP")])
    r = s.results[0]
    assert r.status == "killed" and "8 GB" in r.detail and r.seconds < 15 and r.commits == 1


def test_low_memory_skips(env, monkeypatch):
    env.settings.min_free_gb = 16
    monkeypatch.setattr(ns, "mem_available_gb", lambda: 3.0)
    s = shift(env, [env.task("skip", "EDIT")])
    assert s.results[0].status == "skipped" and "3 GB" in s.results[0].detail
    assert git(env.repo, "worktree", "list").count("\n") == 0  # no worktree was made


def test_server_down_skips(env):
    env.settings.server = "http://127.0.0.1:9"
    s = shift(env, [env.task("a", "EDIT"), env.task("b", "EDIT")])
    assert [r.status for r in s.results] == ["skipped", "skipped"] and "does not answer" in s.results[0].detail


def test_until_and_stop(env):
    s = ns.Shift(env.settings, [env.task("late", "EDIT")], log=lambda m: None)
    s.until = dt.datetime.now() - dt.timedelta(seconds=1)
    s.run()
    assert s.results[0].status == "skipped" and s.results[0].detail.startswith("past")
    s = ns.Shift(env.settings, [env.task("stopped", "EDIT")], log=lambda m: None)
    s.request_stop()
    s.run()
    assert s.results[0].status == "skipped" and "stopped" in s.results[0].detail


def test_until_time():
    f = ns.until_time
    assert f(dt.datetime(2026, 10, 14, 1, 0), "07:00") == dt.datetime(2026, 10, 14, 7, 0)
    assert f(dt.datetime(2026, 10, 14, 23, 30), "07:00") == dt.datetime(2026, 10, 15, 7, 0)
    assert f(dt.datetime(2026, 10, 14, 7, 0), "07:00") == dt.datetime(2026, 10, 15, 7, 0)
    assert f(dt.datetime(2026, 10, 14, 1, 0), None) is None


def test_bad_base_and_repo(env):
    s = shift(env, [env.task("nobase", "EDIT", base="nope"), ns.Task("norepo", "EDIT", repo=str(env.tmp / "missing"))])
    assert s.results[0].status == "error" and "unknown base" in s.results[0].detail
    assert s.results[1].status == "error" and "not a git repository" in s.results[1].detail


def test_same_task_twice(env):
    s1 = shift(env, [env.task("again", "EDIT")])
    s2 = shift(env, [env.task("again", "EDIT")])
    assert s1.results[0].branch == f"night/{s1.date}/again" and s2.results[0].branch == f"night/{s1.date}/again-2"
    assert s2.run_dir == s1.run_dir + "-2"


def test_prune(env):
    s = shift(env, [env.task("edit", "EDIT"), env.task("noop", "nothing")])
    tree, branch = s.results[0].tree, s.results[0].branch
    assert os.path.isdir(tree)
    assert ns.prune(s.run_dir, log=lambda m: None) == 1
    assert not os.path.exists(tree) and git(env.repo, "rev-parse", "--verify", branch)  # the branch stays


# ---------------------------------------------------------------------------------------------------- the task file
def test_parse_duration():
    assert ns.parse_duration("90") == 90 and ns.parse_duration("90s") == 90 and ns.parse_duration("20m") == 1200 and ns.parse_duration("1.5h") == 5400
    with pytest.raises(ValueError):
        ns.parse_duration("soon")


def test_load(tmp_path):
    (tmp_path / "p.md").write_text("from a file\n")
    cfg = tmp_path / "tasks.toml"
    cfg.write_text(f'''
[defaults]
repo = "{tmp_path}"
timeout = "20m"
allow = ["Bash(git diff*)"]
until = "07:00"
min_free_gb = 1
model = "m"
[defaults.env]
A = "1"

[[task]]
name = "one two"
prompt = "x"
allow = ["Bash(uv run pytest*)"]
env = {{ B = "2" }}

[[task]]
name = "two"
prompt_file = "p.md"
timeout = "90s"
effort = "medium"
repo = "~/elsewhere"
''')
    settings, tasks = ns.load(str(cfg))
    assert settings.until == "07:00" and settings.min_free_gb == 1 and settings.model == "m" and settings.wait_server_s == 600
    a, b = tasks
    assert a.name == "one-two" and a.allow == ["Bash(git diff*)", "Bash(uv run pytest*)"] and a.env == {"A": "1", "B": "2"}
    assert a.timeout_s == 1200 and a.repo == str(tmp_path) and a.model == "m" and a.deny == ["WebSearch", "WebFetch"]
    assert b.prompt == "from a file\n" and b.timeout_s == 90 and b.effort == "medium" and b.repo == os.path.expanduser("~/elsewhere")
    assert b.allow == ["Bash(git diff*)"] and b.env == {"A": "1"}


@pytest.mark.parametrize("body, msg", [
    ('[[task]]\nname = "a"\nprompt = "x"\nrepo = "."\nbogus = 1\n', "unknown keys"),
    ('[defaults]\nbogus = 1\n', "unknown keys"),
    ('[[task]]\nname = "a"\nprompt = "x"\nrepo = "."\n[[task]]\nname = "a"\nprompt = "y"\nrepo = "."\n', "used twice"),
    ('[[task]]\nname = "a"\nprompt = "x"\nprompt_file = "p"\nrepo = "."\n', "exclusive"),
    ('[[task]]\nname = "a"\nprompt = "x"\nrepo = "."\ntimeout = "soon"\n', "duration"),
    ('[defaults]\nuntil = "7am"\n[[task]]\nname = "a"\nprompt = "x"\nrepo = "."\n', "HH:MM"),
    ('[[task]]\nname = "a"\nrepo = "."\n', "prompt is required"),
    ('[[task]]\nname = "a"\nprompt = "x"\n', "repo is required"),
    ('[[task]]\nname = "///"\nprompt = "x"\nrepo = "."\n', "no usable characters"),
])
def test_load_errors(tmp_path, body, msg):
    cfg = tmp_path / "t.toml"
    cfg.write_text(body)
    with pytest.raises(ValueError, match=msg):
        ns.load(str(cfg))


# ------------------------------------------------------------------------------------------------- the command line
def test_main_dry_run_and_only(env, capsys):
    cfg = env.tmp / "tasks.toml"
    cfg.write_text(f'[[task]]\nname = "a"\nprompt = "EDIT"\nrepo = "{env.repo}"\n[[task]]\nname = "b"\nprompt = "x"\nrepo = "{env.tmp}/missing"\n')
    assert ns.main(["--dry-run", str(cfg), "--claude", env.fake, "--only", "a", "--state-dir", str(env.tmp / "state")]) == 0
    out = capsys.readouterr().out
    assert "a: " in out and "b: " not in out and "allow:" in out and "claude: " in out
    assert not os.path.exists(env.tmp / "state")  # a dry run makes nothing
    assert ns.main(["--dry-run", str(cfg), "--claude", env.fake]) == 1  # task b's repository is missing
    assert ns.main([str(cfg), "--only", "zzz"]) == 2
    assert ns.main([str(env.tmp / "absent.toml")]) == 2
    (env.tmp / "empty.toml").write_text("")
    assert ns.main([str(env.tmp / "empty.toml")]) == 2


def test_main_runs(env, capsys):
    cfg = env.tmp / "tasks.toml"
    cfg.write_text(f'''
[defaults]
server = "{env.settings.server}"
state_dir = "{env.tmp}/state"
min_free_gb = 0
wait_server = "0s"
[defaults.env]
FAKE_CLAUDE_LOG = "{env.tmp}/main.log.json"
[[task]]
name = "a"
prompt = "EDIT"
repo = "{env.repo}"
timeout = "30s"
''')
    assert ns.main([str(cfg), "--claude", env.fake]) == 0
    out = capsys.readouterr().out
    assert "[a] done" in out and "report:" in out
    assert git(env.repo, "branch", "--list", "night/*/a").strip().endswith("/a")

"""The bookkeeping of the review harness (tools/review.py) on a small git repository per test: the scope of a run, the
checks and the Stop hook, the export, the snapshots, the ledger and the eval score."""
import json
import os

import pytest

from tests.test_night_shift import make_repo, run
from tools import review as rv

REVIEW_MD = "# Review\n\n## Skip\n\n- `vendor`\n- `*.lock`\n\n## Severity\n\n- `not-a-skip`\n"


@pytest.fixture
def repo(tmp_path):
    r = make_repo(str(tmp_path / "repo"))
    write(r, "REVIEW.md", REVIEW_MD)
    write(r, "a.py", "x = 1\n")
    run("git", "-C", r, "add", "-A")
    run("git", "-C", r, "commit", "-qm", "base")
    return r


def write(repo, name, text):
    path = os.path.join(repo, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def read(repo, name):
    with open(os.path.join(repo, name)) as f:
        return f.read()


def with_checks(repo, **kw):
    """REVIEW.md with a Checks section: one bullet per key (an underscore stands for a dash), the values in backticks."""
    bullets = [f"- `{k.replace('_', '-')}`: " + ", ".join(f"`{x}`" for x in (v if isinstance(v, (list, tuple)) else [v]))
               for k, v in kw.items()]
    write(repo, "REVIEW.md", REVIEW_MD + "\n## Checks\n\nThe hook reads this section.\n\n" + "\n".join(bullets) + "\n")


def finding(**kw):
    return {"lens": "security", "severity": "high", "file": "a.py", "title": "t", "bug_class": "c"} | kw


def test_skip_paths_come_from_the_skip_section_only(repo):
    assert rv.skip_paths(repo) == [".review", "vendor", "*.lock"]
    with_checks(repo, code=["a.py"], lint="true")
    assert rv.skip_paths(repo) == [".review", "vendor", "*.lock"]  # the bullets of the Checks section are not paths


def test_checks_come_from_the_checks_section(repo):
    assert rv.checks(repo) == {}
    with_checks(repo, code=["a.py", "src"], lint="ruff check .", test="pytest -q -x", full="make check",
                full_for=["src/core", "*.cu"], min_gb="16")
    assert rv.checks(repo) == {"code": ["a.py", "src"], "lint": "ruff check .", "test": "pytest -q -x", "full": "make check",
                               "full-for": ["src/core", "*.cu"], "min-gb": 16.0}
    with_checks(repo, lint="true", min_gb="sixteen")
    with pytest.raises(rv.ReviewError, match="min-gb"):
        rv.checks(repo)


def test_init_writes_the_contract_and_the_git_lines(tmp_path):
    r = make_repo(str(tmp_path / "repo"))
    write(r, ".gitignore", "*.pyc")  # no final newline
    done = rv.init(r)
    assert done[0].startswith("REVIEW.md: written") and rv.checks(r) == {}  # the template's example is not a live Checks section
    assert read(r, ".gitignore") == "*.pyc\n.review/runs/\n.review/cache/\n"
    assert read(r, ".gitattributes") == ".review/ledger.jsonl merge=union\n"
    assert rv.init(r) == ["REVIEW.md exists: left as it is"]  # a second init changes nothing
    assert read(r, ".gitignore").count(".review/runs/") == 1


def test_export_copies_the_harness_and_adds_the_stop_hook(tmp_path):
    src = os.path.dirname(os.path.dirname(os.path.abspath(rv.__file__)))  # this repository
    target = make_repo(str(tmp_path / "other"))
    write(target, ".claude/settings.json", json.dumps({"permissions": {"allow": ["Bash(ls *)"]}}))
    done = rv.export(src, target)
    for rel in (".claude/agents/refuter.md", ".claude/skills/reviewing-features/SKILL.md", ".claude/review/slop-detector.md",
                ".claude/review/REVIEW.template.md", ".claude/hooks/quick-check.sh", "tools/review.py", "REVIEW.md"):
        assert os.path.isfile(os.path.join(target, rel)), rel
    assert os.access(os.path.join(target, ".claude/hooks/quick-check.sh"), os.X_OK)
    settings = json.loads(read(target, ".claude/settings.json"))
    assert settings["permissions"] == {"allow": ["Bash(ls *)"]} and settings["hooks"]["Stop"] == [{"hooks": [rv.STOP_HOOK]}]
    assert any(x.startswith("REVIEW.md: written") for x in done) and ".review/runs/" in read(target, ".gitignore")
    write(target, "REVIEW.md", "# mine\n")
    rv.export(src, target)  # a second export keeps REVIEW.md and adds no second hook
    assert read(target, "REVIEW.md") == "# mine\n" and len(json.loads(read(target, ".claude/settings.json"))["hooks"]["Stop"]) == 1
    with pytest.raises(rv.ReviewError, match="this repository"):
        rv.export(src, src)
    with pytest.raises(rv.ReviewError, match="not a git repository"):
        rv.export(src, str(tmp_path / "nowhere"))


def test_snapshot_sees_untracked_files_and_leaves_the_index(repo):
    s0 = rv.snapshot(repo)
    write(repo, "new.py", "y = 2\n")
    s1 = rv.snapshot(repo)
    assert s1 != s0 and "new.py" in run("git", "-C", repo, "ls-tree", "--name-only", s1)
    assert run("git", "-C", repo, "diff", "--cached", "--name-only") == ""  # nothing staged
    assert run("git", "-C", repo, "status", "--porcelain") == "?? new.py"
    code = rv.snapshot(repo, ("a.py",))
    write(repo, "new.py", "y = 3\n")
    assert rv.snapshot(repo, ("a.py",)) == code  # a change outside the paths: the same fingerprint
    write(repo, "a.py", "x = 2\n")
    assert rv.snapshot(repo, ("a.py",)) != code


def test_snapshot_sees_a_change_of_the_same_size_in_the_second_of_the_index(repo):
    # The times are pinned, as when a.py, its commit and its edit fall in one second. git then hashes a.py again
    # (racy git), but only while the copy of the index keeps the mtime of the index.
    run("git", "-C", repo, "config", "core.trustCtime", "false")  # the kernel sets the ctime, the test cannot pin it
    t = 1_700_000_000 * 10**9
    os.utime(os.path.join(repo, "a.py"), ns=(t, t))
    run("git", "-C", repo, "commit", "-qam", "pinned", "--allow-empty")
    os.utime(os.path.join(repo, ".git", "index"), ns=(t, t))
    write(repo, "a.py", "x = 5\n")
    os.utime(os.path.join(repo, "a.py"), ns=(t, t))  # the same size and mtime as the stat data of the index
    assert run("git", "-C", repo, "show", rv.snapshot(repo) + ":a.py") == "x = 5"


def test_scope_leaves_out_the_skipped_paths(repo):
    run("git", "-C", repo, "checkout", "-qb", "feature")
    write(repo, "a.py", "x = 2\n")
    write(repo, "b.py", "z = 3\n")
    run("git", "-C", repo, "add", "-A")
    run("git", "-C", repo, "commit", "-qm", "feature")
    write(repo, "c.py", "w = 4\n")  # untracked: part of the feature
    for name in ("vendor/v.py", "uv.lock", ".review/ledger.jsonl"):
        write(repo, name, "skip me\n")
    s = rv.scope(repo)
    assert [f["path"] for f in s["files"]] == ["a.py", "b.py", "c.py"]
    assert s["base"] == run("git", "-C", repo, "rev-parse", "main") and (s["added"], s["removed"]) == (3, 1)
    with open(os.path.join(repo, s["dir"], "diff.patch")) as f:
        patch = f.read()
    assert "c.py" in patch and "vendor" not in patch and "skip me" not in patch


def test_nothing_to_review_names_a_base(repo):
    with pytest.raises(rv.ReviewError, match="HEAD~3"):
        rv.scope(repo)
    write(repo, "a.py", "x = 5\n")  # on main, uncommitted work is the scope
    assert [f["path"] for f in rv.scope(repo)["files"]] == ["a.py"]


def test_gate_selects_the_full_check_for_its_paths(repo):
    with_checks(repo, lint="true", test="true", full="make check", full_for=["src/core", "*.cu"])
    write(repo, "a.py", "x = 2\n")
    assert rv.gate(repo, rv.scope(repo)["run"]) == "quick"
    write(repo, "k.cu", "__global__\n")
    assert rv.gate(repo, rv.scope(repo)["run"]) == "full: make check"  # the glob
    os.remove(os.path.join(repo, "k.cu"))
    write(repo, "src/core/k.py", "y = 1\n")
    assert rv.gate(repo, rv.scope(repo)["run"]) == "full: make check"  # a file under the directory
    write(repo, "REVIEW.md", REVIEW_MD)
    assert rv.gate(repo, rv.scope(repo)["run"]).startswith("none")


def test_check_runs_lint_then_tests_and_records_a_pass(repo):
    with_checks(repo, code=["a.py", "missing"], lint="echo linted", test="echo tested")
    rc, out = rv.run_check(repo)
    assert rc == 0 and "linted" in out and "tested" in out
    assert read(repo, ".review/cache/passed") == rv.snapshot(repo, ("a.py",))  # a code path that matches nothing is left out
    with_checks(repo, lint="echo linted", test="echo broken; exit 1")
    rc, out = rv.run_check(repo)
    assert rc == 1 and "broken" in out and read(repo, ".review/cache/last.log") == out
    with_checks(repo, lint="exit 1", test="echo tested")
    rc, out = rv.run_check(repo)
    assert rc == 1 and "tested" not in out  # the tests do not run after a lint failure
    with_checks(repo, lint="echo linted", test="echo tested", min_gb="1000000")
    rc, out = rv.run_check(repo)
    assert rc == 3 and "did not run" in out and "tested" not in out
    write(repo, "REVIEW.md", REVIEW_MD)
    with pytest.raises(rv.ReviewError, match="Checks"):
        rv.run_check(repo)


def test_the_stop_hook_blocks_a_failing_check_at_most_three_times(repo):
    def hook(active=False, **env):
        return rv.stop_hook(repo, {"stop_hook_active": active}, env)
    with_checks(repo, code=["a.py"], lint="true", test="exit 1")
    assert hook() == (0, "", "")  # the code equals HEAD
    write(repo, "a.py", "x = 2\n")
    assert hook(REVIEW_STOP_CHECK="0") == (0, "", "")  # the kill switch
    rc, out, err = hook()
    assert rc == 2 and out == "" and "Fix the cause" in err and "$ exit 1" in err
    assert read(repo, ".review/cache/blocks") == "1"
    assert hook(active=True)[0] == 2 and hook(active=True)[0] == 2 and read(repo, ".review/cache/blocks") == "3"
    rc, out, err = hook(active=True)
    assert rc == 0 and err == "" and "still failing after 3 attempts" in json.loads(out)["systemMessage"]
    assert hook()[0] == 2 and read(repo, ".review/cache/blocks") == "1"  # a new turn: the count starts again
    with_checks(repo, code=["a.py"], lint="true", test="exit 1", min_gb="1000000")
    rc, out, err = hook()
    assert rc == 0 and "did not run" in json.loads(out)["systemMessage"]  # too little memory: a message, no block
    with_checks(repo, code=["a.py"], lint="true", test="true")
    assert hook() == (0, "", "") and read(repo, ".review/cache/passed") == rv.snapshot(repo, ("a.py",))
    with_checks(repo, code=["a.py"], lint="true", test="exit 1")
    assert hook() == (0, "", "")  # the same code as the last pass: the check does not run again
    write(repo, "a.py", "x = 3\n")
    write(repo, "REVIEW.md", REVIEW_MD)
    assert hook() == (0, "", "")  # no Checks section: nothing to do


def test_the_last_decision_wins(repo):
    ids = rv.add(repo, "r1", [finding(title="first"), finding(lens="slop", severity="low", title="second")])
    assert ids == ["r1.1", "r1.2"]
    rv.decide(repo, "r1.1", "fixed")
    rv.decide(repo, "r1.2", "wrong", "the caller checks it")
    rv.decide(repo, "r1.2", "intentional")
    recs = rv.load(repo)
    assert recs["r1.1"]["decision"] == "fixed" and recs["r1.2"]["decision"] == "intentional"
    assert rv.add(repo, "r1", [finding()], round_=2) == ["r1.3"]  # numbering goes on within the run
    with pytest.raises(rv.ReviewError):
        rv.decide(repo, "r9.1", "fixed")
    with pytest.raises(rv.ReviewError, match="lens"):
        rv.add(repo, "r1", [finding(lens="style")])
    sec = next(x for x in rv.stats(repo) if x.startswith("security"))
    assert sec.split()[1:3] == ["2", "1"] and sec.endswith("100%")  # two findings, one fixed, the other undecided


def test_known_lists_the_run_and_the_unfixed_earlier_findings_on_its_files(repo):
    rv.add(repo, "old", [finding(file="a.py", title="on a"), finding(file="z.py", title="on z"),
                         finding(file="a.py", title="fixed on a")])
    rv.decide(repo, "old.3", "fixed")
    write(repo, "a.py", "x = 6\n")
    s = rv.scope(repo)
    rv.add(repo, s["run"], [finding(file="a.py", title="this run")])
    rv.decide(repo, f"{s['run']}.1", "fixed")  # round 2 reviews the fix diff: the run's own fixes stay known
    titles = [x.split(" | ")[1] for x in rv.known(repo, s["run"])]
    assert titles == ["on a", "this run"]  # a fixed finding of an earlier run that comes back is a regression


def test_diff_refuses_an_option_in_place_of_a_snapshot(repo, monkeypatch):
    monkeypatch.chdir(repo)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    out = os.path.join(repo, "written")
    with pytest.raises(rv.ReviewError):
        rv.main(["diff", "--", f"--output={out}", "HEAD"])
    assert not os.path.exists(out)  # git diff --output writes the diff to any file


def test_recurring_needs_fixes_in_two_runs(repo):
    rv.add(repo, "r1", [finding(bug_class="pinned-buffer-reuse"), finding(bug_class="catch-all-except")])
    rv.add(repo, "r2", [finding(bug_class="pinned-buffer-reuse")])
    for fid in ("r1.1", "r1.2", "r2.1"):
        rv.decide(repo, fid, "fixed")
    out = rv.recurring(repo)
    assert len(out) == 1 and out[0].startswith("pinned-buffer-reuse: fixed in 2 runs")


def test_eval_score(repo):
    old = run("git", "-C", repo, "rev-parse", "HEAD")
    write(repo, ".review/evals/cases.jsonl", "\n".join(json.dumps(c) for c in [
        {"name": "bug", "tree": old, "paths": ["a.py"], "expect": [{"files": ["a.py"], "any": ["overflow"], "what": "x overflows"}]},
        {"name": "clean", "tree": old, "paths": ["a.py"], "expect": []}]) + "\n")
    ev = rv.eval_setup(repo)
    assert all(os.path.isfile(os.path.join(repo, c["dir"], "a.py")) for c in ev["cases"])
    results = {"bug": [finding(title="int Overflow in x", verdict="confirmed")],
               "clean": [finding(title="a false alarm", verdict="refuted"), finding(severity="low", title="a nit")]}
    write(repo, os.path.join(ev["dir"], "results.json"), json.dumps(results))
    out = rv.eval_score(repo, ev["run"])
    assert out[0] == "bug: x overflows: caught by security: int Overflow in x (confirmed)"
    assert out[-1].startswith("score: caught 1/1 (1 confirmed), false alarms 1 (0 confirmed), 3 findings")
    assert not any(os.path.exists(os.path.join(repo, c["dir"])) for c in ev["cases"])  # the worktrees are gone
    ev2 = rv.eval_setup(repo)
    write(repo, os.path.join(ev2["dir"], "results.json"), "{}")
    assert rv.eval_score(repo, ev2["run"])[-1].startswith("previous: caught 1/1")


def test_a_baseline_score_compares_with_the_harness(repo):
    old = run("git", "-C", repo, "rev-parse", "HEAD")
    write(repo, ".review/evals/cases.jsonl",
          json.dumps({"name": "bug", "tree": old, "paths": ["a.py"], "expect": [{"files": ["a.py"], "any": ["overflow"]}]}) + "\n")

    def score(baseline, results):
        ev = rv.eval_setup(repo, baseline=baseline)
        write(repo, os.path.join(ev["dir"], "results.json"), json.dumps(results))
        return ev, rv.eval_score(repo, ev["run"])
    ev, out = score(True, {"bug": [finding(title="int overflow in x")]})
    assert ev["prompt_hash"] == rv.BASELINE and out[-1].startswith("score: caught 1/1")  # nothing to compare with yet
    ev, out = score(False, {})
    assert ev["prompt_hash"] != rv.BASELINE and out[-2].startswith("score: caught 0/1")
    assert out[-1].startswith("baseline: caught 1/1 (0 confirmed)")  # no harness run before it: no "previous" line
    _, out = score(True, {})
    assert out[-2].startswith("previous: caught 1/1") and "prompts baseline" in out[-2]  # the baseline before it
    assert out[-1].startswith("harness: caught 0/1")

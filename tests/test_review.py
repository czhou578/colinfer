"""The bookkeeping of the review harness (tools/review.py) on a small git repository per test: the scope of a run, the
snapshots, the ledger and the eval score."""
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


def finding(**kw):
    return {"lens": "security", "severity": "high", "file": "a.py", "title": "t", "bug_class": "c"} | kw


def test_skip_paths_come_from_the_skip_section_only(repo):
    assert rv.skip_paths(repo) == [".review", "vendor", "*.lock"]


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

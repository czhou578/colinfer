"""The bookkeeping of the review harness (.claude/skills/reviewing-features/SKILL.md, docs/review.md). The model does the
judgment: the finders, the refuters and the fixes. This script does the parts that must give the same answer each time:
the scope of a review, the snapshots of the working tree, the ledger of findings and the score of the eval set.

  python3 tools/review.py scope [BASE]                 # a new run: .review/runs/<run>/scope.json and diff.patch
  python3 tools/review.py snapshot [--paths P ...]     # a hash of the working tree, untracked files included
  python3 tools/review.py diff A B                     # the diff between two snapshots, without the skipped paths
  python3 tools/review.py add RUN FINDINGS.json [--round N]   # append findings to the ledger; prints their ids
  python3 tools/review.py decide ID DECISION [NOTE]    # what happened to a finding: one of DECISIONS
  python3 tools/review.py known [RUN]                  # the findings that the finders must not report again
  python3 tools/review.py stats                        # the decisions per lens: how noisy each lens is
  python3 tools/review.py recurring                    # bug classes fixed in two or more runs: candidates for a rule
  python3 tools/review.py eval-setup [--baseline]      # a run with one worktree per case of .review/evals/cases.jsonl
  python3 tools/review.py eval-score RUN               # the score of that run's results.json; removes its worktrees

The ledger (.review/ledger.jsonl) only grows: a decision is a new line for the same id, and the last line wins. Thus two
branches that both append to it merge without a conflict (.gitattributes: merge=union).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

LEDGER = ".review/ledger.jsonl"
RUNS = ".review/runs"
CASES = ".review/evals/cases.jsonl"
HISTORY = ".review/evals/history.jsonl"
LENSES = ("slop", "security", "memrace")
SEVERITIES = ("critical", "high", "medium", "low")
DECISIONS = ("open", "fixed", "wrong", "intentional", "not-worth-it", "backlog")
DISMISSED = ("wrong", "intentional", "not-worth-it")
# The text of the reviewers. An eval score compares with an earlier one only when this text is the same.
PROMPT_FILES = ("REVIEW.md", ".claude/review", ".claude/agents", ".claude/skills/reviewing-features")
BASELINE = "baseline"  # the prompt_hash of a run of Claude without the harness (eval-mode.md)


class ReviewError(Exception):
    """A request that the script cannot do: the message says what to pass instead."""


def git(repo: str, *args: str, env: dict | None = None, strip: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, env=env)
    if r.returncode:
        raise ReviewError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip() if strip else r.stdout


def rev(repo: str, name: str, kind: str = "commit") -> str:
    """The hash of a commit or a tree. A name that starts with "-" is an error, because git reads it as an option
    (git diff --output=FILE writes a file)."""
    if name.startswith("-"):
        raise ReviewError(f"{name!r} is not a {kind}: pass a hash or a branch name")
    return git(repo, "rev-parse", "--verify", f"{name}^{{{kind}}}")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------------ scope and diffs
def skip_paths(repo: str) -> list[str]:
    """The paths in the "## Skip" section of REVIEW.md, one `path` per bullet, and .review itself. No diff of the
    harness shows them, so no finder reads them."""
    try:
        with open(os.path.join(repo, "REVIEW.md")) as f:
            text = f.read()
    except FileNotFoundError:
        text = ""
    m = re.search(r"^## Skip\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return [".review", *(re.findall(r"^- `([^`]+)`", m.group(1), re.M) if m else [])]


def pathspec(repo: str) -> list[str]:
    return ["--", ".", *(f":(exclude){p}" for p in skip_paths(repo))]


def snapshot(repo: str, paths: tuple[str, ...] = ()) -> str:
    """A tree hash of the working tree: the tracked files as they are on disk, and the untracked files that .gitignore
    does not exclude. It goes through a copy of the index, so what you staged does not change. With paths, a hash of
    only those paths (the Stop hook's fingerprint of the code)."""
    real = git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index")
    with tempfile.TemporaryDirectory() as d:
        idx = os.path.join(d, "index")
        if os.path.exists(real):
            shutil.copyfile(real, idx)  # its stat data saves the hashing of the unchanged files
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        git(repo, "add", "-A", "--", *(paths or ["."]), env=env)
        tree = git(repo, "write-tree", env=env)
    if not paths:
        return tree
    return hashlib.sha256(git(repo, "ls-tree", tree, "--", *paths).encode()).hexdigest()[:16]


def default_base(repo: str) -> str:
    for ref in ("main", "origin/main"):
        try:
            git(repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}")
        except ReviewError:
            continue
        return git(repo, "merge-base", "HEAD", ref)
    raise ReviewError("no main branch: pass the base commit, e.g. HEAD~3")


def new_run(repo: str) -> tuple[str, str]:
    run = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d, n = os.path.join(repo, RUNS, run), 1
    while os.path.exists(d):
        n += 1
        d = os.path.join(repo, RUNS, f"{run}-{n}")
    os.makedirs(d)
    return os.path.basename(d), d


def scope(repo: str, base: str | None = None) -> dict:
    """A new run: the files and the diff between the base commit (default: the merge base with main) and the working
    tree, without the skipped paths. Writes scope.json and diff.patch into the run's directory."""
    base_ref = base or "merge-base with main"
    base = rev(repo, base or default_base(repo))
    snap, spec = snapshot(repo), pathspec(repo)
    files = []
    for line in git(repo, "diff", "--name-status", base, snap, *spec).splitlines():
        status, *paths = line.split("\t")
        files.append({"status": status[0], "path": paths[-1]})
    if not files:
        raise ReviewError(f"nothing to review: the working tree equals {base[:9]} ({base_ref}) outside the skipped paths; "
                          "pass a base commit, e.g. HEAD~3")
    added = removed = 0
    for line in git(repo, "diff", "--numstat", base, snap, *spec).splitlines():
        a, r, _ = line.split("\t", 2)
        added += int(a) if a.isdigit() else 0
        removed += int(r) if r.isdigit() else 0
    run, d = new_run(repo)
    with open(os.path.join(d, "diff.patch"), "w") as f:
        f.write(git(repo, "diff", base, snap, *spec, strip=False))
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    s = {"run": run, "dir": os.path.relpath(d, repo), "base": base, "base_ref": base_ref, "head": head, "branch": branch,
         "snapshot": snap, "files": files, "added": added, "removed": removed, "skip": skip_paths(repo)}
    with open(os.path.join(d, "scope.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s


def load_scope(repo: str, run: str) -> dict:
    try:
        with open(os.path.join(repo, RUNS, run, "scope.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ReviewError(f"no run {run} in {RUNS}") from None


# ---------------------------------------------------------------------------------------------------------------- ledger
def load(repo: str) -> dict[str, dict]:
    """The ledger folded by id: each finding with its last decision."""
    recs: dict[str, dict] = {}
    try:
        with open(os.path.join(repo, LEDGER)) as f:
            lines = [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        return recs
    for r in lines:
        if r.get("event") == "decide":
            if r["id"] in recs:
                recs[r["id"]].update(decision=r["decision"], note=r.get("note", ""), decided=r["ts"])
        else:
            recs[r["id"]] = r
    return recs


def append(repo: str, records: list[dict]):
    path = os.path.join(repo, LEDGER)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def add(repo: str, run: str, findings: list[dict], round_: int = 1) -> list[str]:
    """Appends the findings of one round; every finding must name its lens, severity, file and title."""
    for i, x in enumerate(findings):
        missing = [k for k in ("lens", "severity", "file", "title") if not x.get(k)]
        if missing:
            raise ReviewError(f"finding {i} has no {', '.join(missing)}")
        if x["lens"] not in LENSES or x["severity"] not in SEVERITIES or x.get("decision", "open") not in DECISIONS:
            raise ReviewError(f"finding {i}: lens must be one of {LENSES}, severity one of {SEVERITIES}, decision one of {DECISIONS}")
    n = sum(r["run"] == run for r in load(repo).values())
    recs = [{"decision": "open", **x, "id": f"{run}.{n + i + 1}", "run": run, "round": round_, "ts": now()}
            for i, x in enumerate(findings)]
    append(repo, recs)
    return [r["id"] for r in recs]


def decide(repo: str, fid: str, decision: str, note: str = ""):
    if decision not in DECISIONS:
        raise ReviewError(f"decision must be one of {DECISIONS}")
    if fid not in load(repo):
        raise ReviewError(f"no finding {fid} in the ledger")
    append(repo, [{"event": "decide", "id": fid, "decision": decision, "note": note, "ts": now()}])


def known(repo: str, run: str | None = None) -> list[str]:
    """One line per finding that a finder must not report again: every finding of the run, and the findings of earlier
    runs on the files of its scope that did not get a fix. A fixed finding that comes back is a regression, so the
    finders must report it. Without a run, every finding."""
    files = {f["path"] for f in load_scope(repo, run)["files"]} if run else None
    return [f"{r['id']} {r['lens']} {r['decision']} {r.get('verdict', '-')} {r['file']}:{r.get('symbol') or r.get('line') or '-'} "
            f"{r.get('bug_class') or '-'} | {r['title']}"
            for r in load(repo).values()
            if files is None or r["run"] == run or (r["file"] in files and r["decision"] != "fixed")]


def stats(repo: str) -> list[str]:
    """Per lens: the decisions, the refuted findings, and the share of the decided findings that led to a fix."""
    cols = ("fixed", *DISMISSED, "backlog", "open")
    out = [f"{'lens':9}{'all':>5}" + "".join(f"{c:>14}" for c in cols) + f"{'refuted':>9}{'fixed share':>13}"]
    for lens in LENSES:
        rs = [r for r in load(repo).values() if r["lens"] == lens]
        n = {c: sum(r["decision"] == c for r in rs) for c in cols}
        decided = n["fixed"] + sum(n[c] for c in DISMISSED)
        share = f"{n['fixed'] / decided:.0%}" if decided else "-"
        out.append(f"{lens:9}{len(rs):>5}" + "".join(f"{n[c]:>14}" for c in cols)
                   + f"{sum(r.get('verdict') == 'refuted' for r in rs):>9}{share:>13}")
    return out


def recurring(repo: str, min_runs: int = 2) -> list[str]:
    """The bug classes with a fixed finding in min_runs or more runs: the builder session should not make them."""
    by: dict[str, list[dict]] = {}
    for r in load(repo).values():
        if r["decision"] == "fixed" and r.get("bug_class"):
            by.setdefault(r["bug_class"], []).append(r)
    out = []
    for cls, rs in sorted(by.items()):
        runs = sorted({r["run"] for r in rs})
        if len(runs) >= min_runs:
            out.append(f"{cls}: fixed in {len(runs)} runs ({', '.join(runs)}); e.g. {rs[-1]['title']} ({rs[-1]['file']})")
    return out


# ------------------------------------------------------------------------------------------------------------------ eval
def prompt_hash(repo: str) -> str:
    h = hashlib.sha256()
    for p in PROMPT_FILES:
        full = os.path.join(repo, p)
        names = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(full) for f in fs) if os.path.isdir(full) else [full]
        for name in names:
            if os.path.exists(name):
                h.update(os.path.relpath(name, repo).encode())
                with open(name, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()[:12]


def eval_setup(repo: str, baseline: bool = False) -> dict:
    """A run with a detached worktree of each case's commit. The finders read the code of a case in its worktree. A
    baseline run (Claude without the harness) records BASELINE in place of the hash of the prompt files."""
    try:
        with open(os.path.join(repo, CASES)) as f:
            cases = [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        raise ReviewError(f"no eval set: write the cases into {CASES} (docs/review.md)") from None
    run, d = new_run(repo)
    for c in cases:
        c["dir"] = os.path.join(os.path.relpath(d, repo), "trees", c["name"])
        git(repo, "worktree", "add", "--detach", "-q", os.path.join(repo, c["dir"]), rev(repo, c["tree"]))
    ev = {"run": run, "dir": os.path.relpath(d, repo), "prompt_hash": BASELINE if baseline else prompt_hash(repo), "cases": cases}
    with open(os.path.join(d, "eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    return ev


def _text(f: dict) -> str:
    return " ".join(str(f.get(k, "")) for k in ("title", "failure_scenario", "evidence", "symbol", "bug_class")).lower()


def eval_score(repo: str, run: str) -> list[str]:
    """Scores the run's results.json ({case name: [findings with their verdicts]}) against the expected findings of
    each case: an expected finding is caught when a finding names one of its files and one of its keywords. A finding
    of severity critical or high on a clean case is a false alarm. Appends the score to the history and removes the
    run's worktrees. Prints the score before it of the same kind, and the latest score of the other kind (the harness
    against the baseline)."""
    d = os.path.join(repo, RUNS, run)
    with open(os.path.join(d, "eval.json")) as f:
        ev = json.load(f)
    with open(os.path.join(d, "results.json")) as f:
        results = json.load(f)
    score = {"expected": 0, "caught": 0, "caught_confirmed": 0, "false_alarms": 0, "false_alarms_confirmed": 0, "findings": 0}
    out = []
    for c in ev["cases"]:
        fs = results.get(c["name"], [])
        score["findings"] += len(fs)
        for e in c.get("expect", []):
            hits = [f for f in fs if f.get("file") in e["files"] and any(k.lower() in _text(f) for k in e["any"])]
            conf = [f for f in hits if f.get("verdict") == "confirmed"]
            score["expected"] += 1
            score["caught"] += bool(hits)
            score["caught_confirmed"] += bool(conf)
            got = f"caught by {(conf or hits)[0]['lens']}: {(conf or hits)[0]['title']}" if hits else "MISSED"
            out.append(f"{c['name']}: {e.get('what', e['files'][0])}: {got}" + (" (confirmed)" if conf else ""))
        if not c.get("expect"):
            alarms = [f for f in fs if f.get("severity") in ("critical", "high")]
            score["false_alarms"] += len(alarms)
            score["false_alarms_confirmed"] += sum(f.get("verdict") == "confirmed" for f in alarms)
            out.append(f"{c['name']} (clean): {len(alarms)} critical/high findings" + "".join(f"\n  - {f['title']}" for f in alarms))
    try:
        with open(os.path.join(repo, HISTORY)) as f:
            lines = [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        lines = []
    is_base = ev["prompt_hash"] == BASELINE
    prev = next((x for x in reversed(lines) if (x["prompt_hash"] == BASELINE) == is_base), None)
    other = next((x for x in reversed(lines) if (x["prompt_hash"] == BASELINE) != is_base), None)
    line = {"ts": now(), "run": run, "prompt_hash": ev["prompt_hash"], **score}
    os.makedirs(os.path.dirname(os.path.join(repo, HISTORY)), exist_ok=True)
    with open(os.path.join(repo, HISTORY), "a") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    fmt = lambda s: (f"caught {s['caught']}/{s['expected']} ({s['caught_confirmed']} confirmed), "  # noqa: E731
                     f"false alarms {s['false_alarms']} ({s['false_alarms_confirmed']} confirmed), {s['findings']} findings")
    out.append(f"score: {fmt(score)}; prompts {ev['prompt_hash']}")
    if prev:
        out.append(f"previous: {fmt(prev)}; prompts {prev['prompt_hash']} ({prev['run']})")
    if other:
        out.append(f"{'harness' if is_base else 'baseline'}: {fmt(other)}; prompts {other['prompt_hash']} ({other['run']})")
    for c in ev["cases"]:
        if os.path.exists(os.path.join(repo, c["dir"])):
            git(repo, "worktree", "remove", "--force", os.path.join(repo, c["dir"]))
    return out


# ------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="the bookkeeping of the review harness (docs/review.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scope", help="a new run: the files and the diff against BASE (default: the merge base with main)")
    p.add_argument("base", nargs="?")
    p = sub.add_parser("snapshot", help="a hash of the working tree, untracked files included")
    p.add_argument("--paths", nargs="+", default=[])
    p = sub.add_parser("diff", help="the diff between two snapshots, without the skipped paths")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("add", help="append the findings of a JSON file (a list) to the ledger")
    p.add_argument("run")
    p.add_argument("file")
    p.add_argument("--round", type=int, default=1)
    p = sub.add_parser("decide", help=f"record what happened to a finding: {', '.join(DECISIONS)}")
    p.add_argument("id")
    p.add_argument("decision", choices=DECISIONS)
    p.add_argument("note", nargs="?", default="")
    p = sub.add_parser("known", help="the findings that the finders of a run must not report again")
    p.add_argument("run", nargs="?")
    sub.add_parser("stats", help="the decisions per lens")
    sub.add_parser("recurring", help="bug classes fixed in two or more runs")
    p = sub.add_parser("eval-setup", help="a run with one worktree per eval case")
    p.add_argument("--baseline", action="store_true", help="a run of Claude without the harness")
    p = sub.add_parser("eval-score", help="score an eval run's results.json; removes its worktrees")
    p.add_argument("run")
    a = ap.parse_args(argv)
    repo = git(os.getcwd(), "rev-parse", "--show-toplevel")
    if a.cmd == "scope":
        s = scope(repo, a.base)
        print(f"run {s['run']}: {len(s['files'])} files, +{s['added']} -{s['removed']} lines against {s['base'][:9]} ({s['base_ref']})")
        print(f"  {s['dir']}/scope.json, {s['dir']}/diff.patch; snapshot {s['snapshot']}")
    elif a.cmd == "snapshot":
        print(snapshot(repo, tuple(a.paths)))
    elif a.cmd == "diff":
        sys.stdout.write(git(repo, "diff", rev(repo, a.a, "tree"), rev(repo, a.b, "tree"), *pathspec(repo), strip=False))
    elif a.cmd == "add":
        with open(a.file) as f:
            findings = json.load(f)
        for fid, x in zip(add(repo, a.run, findings, a.round), findings):
            print(f"{fid} {x['lens']} {x['severity']} {x.get('verdict', '-')} {x['file']}: {x['title']}")
    elif a.cmd == "decide":
        decide(repo, a.id, a.decision, a.note)
    elif a.cmd == "eval-setup":
        ev = eval_setup(repo, a.baseline)
        print(f"run {ev['run']}: {len(ev['cases'])} cases; prompts {ev['prompt_hash']}")
        for c in ev["cases"]:
            print(f"  {c['name']}: {c['dir']} ({c['tree']}), review {', '.join(c['paths'])}")
    else:
        lines = {"known": lambda: known(repo, a.run), "stats": lambda: stats(repo), "recurring": lambda: recurring(repo),
                 "eval-score": lambda: eval_score(repo, a.run)}[a.cmd]()
        print("\n".join(lines) if lines else "(none)")


if __name__ == "__main__":
    try:
        main()
    except ReviewError as e:
        print(f"review.py: {e}", file=sys.stderr)
        sys.exit(1)

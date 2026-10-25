# Review a feature with the review harness

The review harness reviews a finished feature in one command, `/reviewing-features`, in Claude Code. Three finders look
for low-quality engineering, security defects, and leaks and races. Other agents try to disprove each serious finding,
and a test run must confirm each leak or race. The harness then fixes the confirmed findings that have an easy fix,
and it reviews its own fixes, for at most three rounds. It replaces the manual loop of slop detector, security audit
and memory review.

The harness is generic. `REVIEW.md` at the root of the repository tells it what colinfer is, what failure looks like
here, the conventions, and which checks to run. The skill, the agents and the script hold nothing that is true of
colinfer only. Section 8 copies the harness into another repository with one command.

The harness follows the method that the Claude Code team describes in "Solving code review at Anthropic":

- Parallel finders in a fresh context, not the session that wrote the code.
- Votes that remove false findings before anyone reads them.
- Re-review of the fix diff only, for at most three rounds, without a repeated finding.
- A fix of the class of a defect, not of one line.
- A ledger of what happened to each finding, and an eval set of known bugs.

## The parts

| Part | Function |
|---|---|
| `/reviewing-features` (`.claude/skills/reviewing-features/SKILL.md`) | Runs the review: scope, gate, find, verify, triage, fix, review the fixes, summary. |
| `.claude/skills/reviewing-features/eval-mode.md` | The steps of an eval run. Claude reads it only for `/reviewing-features eval`. |
| `.claude/agents/slop-finder.md` | Applies the slop detector (`.claude/review/slop-detector.md`) to the diff, against the "Conventions" of REVIEW.md. |
| `.claude/agents/security-finder.md` | Audits the diff against the trust boundaries and the "Security checklist" of REVIEW.md. |
| `.claude/agents/memrace-finder.md` | Looks for leaks, races, and CUDA memory and synchronization errors, from the "Concurrency map" of REVIEW.md. |
| `.claude/agents/refuter.md` | Tries to disprove one finding. Two or three of them vote. |
| `.claude/agents/memrace-verifier.md` | Writes a repro test for one leak or race, and runs it under a sanitizer or a stress loop, within the limits of the "Repro" section of REVIEW.md. |
| `.claude/review/finding-format.md` | The JSON fields of a finding, and the names of the bug classes. |
| `.claude/review/REVIEW.template.md` | The REVIEW.md that `tools/review.py init` writes into a new repository. |
| `REVIEW.md` | The contract of this repository: what failure looks like here, the conventions, the security checklist, the repro limits, severity, feasibility, triage, evidence, the paths to skip, and the check commands. |
| `tools/review.py` | The bookkeeping: the scope, the checks, the snapshots of the working tree, the ledger, the eval score, the export. Also the Stop hook. |
| `.claude/hooks/quick-check.sh` | The quick check (the `lint` and `test` commands of REVIEW.md): the Stop hook, and the gate of a review. It runs `tools/review.py`. |
| `.review/ledger.jsonl` | Each finding and its decision. In this repository, git ignores all of `.review/`: the ledger stays on this machine. |
| `.review/evals/cases.jsonl` | The eval set: known bugs at old commits. git ignores it too. |
| `.claude/rules/sanitizers.md` | Tells the session that builds a feature which sanitizer to run after a change of a kernel, a stream or a thread. Colinfer only; not part of the export. |

## How the parts connect

```
/reviewing-features  -->  tools/review.py scope          the files and the diff against main, in .review/runs/<run>/
                 -->  tools/review.py gate           quick-check.sh, or tests/check.sh; stop if the checks fail
                 -->  slop / security / memrace finders, in parallel           JSON findings
                 -->  2-3 refuters per serious finding (votes), or one memrace-verifier (a run)
                 -->  tools/review.py add            the ledger
                 -->  fix by class, check again, review the fix diff (rounds 2 and 3)
                 -->  summary                        .review/runs/<run>/summary.md
```

## 1. Before you start

- Use Claude Code with a Claude model. The agents and the skill set `model: opus`. Through the gateway of
  `docs/claude_code.md`, these requests go to api.anthropic.com, also when your session uses Qwen. The local server runs
  one request at a time, so it cannot run the parallel agents.
- A review starts many Opus agents: three finders per round, two or three refuters per serious finding, and one
  verifier per serious leak or race. Run it once per finished feature, not after each change.
- Claude Code loads the Stop hook, the skill and the agents when a session starts. Open a new session after you
  install or change them.

## 2. Run a review

1. Finish the feature on its branch. Commit it, or leave the changes uncommitted: the scope includes both.
2. In the prompt box of the VS Code extension, type `/reviewing-features`.

| Command | Scope |
|---|---|
| `/reviewing-features` | The diff from the merge base with `main` to the working tree. On `main`, the uncommitted changes. |
| `/reviewing-features HEAD~3` | The diff from that commit to the working tree. |
| `/reviewing-features report` | The same review, without fixes. |
| `/reviewing-features eval` | The eval set instead of a review (section 6). |
| `/reviewing-features eval baseline` | The eval set, reviewed by Claude without the harness (section 6). |

When a feature touches the engine (the `full-for` paths of the "Checks" section of REVIEW.md: `engine/model`,
`engine/spec`, `engine/runtime`, `engine/weights`, `csrc`), the gate runs the full `tests/check.sh`. That check needs
about 105 GB of free memory, so stop the server first. Any other change gets the quick check: ruff, then pytest.

The review ends with a summary. Its first line gives the counts, or says "Nothing to fix." The summary lists the
findings that it fixed, the findings that need your decision, and the findings that it logged. The harness does not
commit. Read the fixes and their tests, then commit.

## 3. Record your decisions

The ledger keeps each finding with its decision: `open`, `fixed`, `wrong`, `intentional`, `not-worth-it` or `backlog`.
The harness records `fixed`, `backlog` and the refuted findings (`wrong`). Record your own decisions for the rest:

```bash
python3 tools/review.py decide 20261016-221500.4 intentional "the slots are a cache; one request at a time"
python3 tools/review.py stats       # per lens: fixed, dismissed, refuted, and the share of decided findings that got a fix
python3 tools/review.py recurring   # bug classes fixed in two or more runs
```

A later run does not report a finding again when the ledger has it. The exception is a fixed finding: when it comes
back, it is a regression, and the finders report it. A low fixed share for a lens means that the lens is noisy: tune
its agent (section 7). A class that `recurring` lists belongs in a rule for the session that builds
features. The skill proposes the rule at the end of a run.

## 4. The Stop hook

When a turn of Claude Code ends with changed code, the Stop hook (`.claude/settings.json`, `quick-check.sh --stop-hook`)
runs the `lint` command, then the `test` command of the "Checks" section of REVIEW.md: ruff, then `pytest tests/ -x`. A
failure sends Claude back to fix it. The hook does nothing in these cases:

- The code paths (`code` in the "Checks" section: `engine`, `tests`, `tools`, `bench`, `csrc`, `pyproject.toml`) equal
  `HEAD`.
- The code equals the code of the last passed check (`.review/cache/passed`).
- `REVIEW_STOP_CHECK=0` is set. The night shift sets it (`docs/night_shift.md`, section 6).

With less than 16 GB of memory available (`min-gb`), the hook runs ruff only, and shows a message. After three failed
checks in a row, it lets the turn end and shows a message. The output of the last check is in
`.review/cache/last.log`.

## 5. Sanitizers while you build

`.claude/rules/sanitizers.md` loads when Claude works on a kernel, a side stream, a pinned buffer or a thread. It
tells Claude to run the matching sanitizer before it ends the turn: compute-sanitizer memcheck, racecheck or synccheck
for a kernel, `TORCH_CUDA_SANITIZER=1` for streams, repeated runs for threads.

> **NOTE:** The caching allocator of torch hides out-of-bounds accesses in kernels. Set
> `PYTORCH_NO_CUDA_MEMORY_CACHING=1` for memcheck and initcheck. Without it, a write 400 KB past the end of a tensor
> went undetected on this machine.

## 6. The eval set

The eval set measures the harness itself. Each line of `.review/evals/cases.jsonl` is one case: a commit with a known
bug, the files to review, and the findings that a good review gives.

```json
{"name": "client-input-raises-on-the-engine-thread", "tree": "<commit with the bug>",
 "paths": ["engine/server/api.py", "engine/server/chat.py"],
 "expect": [{"what": "a boolean property schema raises in parse_tool_call",
             "files": ["engine/server/chat.py"], "any": ["boolean", "attributeerror", "parse_tool_call"]}]}
```

A finding catches an expected finding when it names one of the `files` and contains one of the `any` keywords. The
match is a heuristic. The score lists the finding that matched, so check each match yourself. A case with an empty
`expect` is a clean case: each critical or high finding on it is a false alarm.

The set starts with three cases from the history of this repository, at the parent commits of 1e86a8a, 3c7b717 and
425e003. Add the bugs that you find from now on. Also add clean cases: commits that you know have no defect in the
listed files.

Measure a baseline first. `/reviewing-features eval baseline` gives each case to one plain agent with the instructions
of the manual loop: the slop detector, then a security audit, then a review for leaks and races. It has no refuters
and no rules of REVIEW.md. The harness must catch more than this baseline, without more false alarms. Run the
baseline again when the model changes.

Run `/reviewing-features eval` after each change to REVIEW.md, to an agent, to the skill or to the slop detector.
`.review/evals/history.jsonl` keeps each score with a hash of the prompt files, or `baseline`. The score of a run
shows the score before it and the latest score of the other kind. Keep a change only when it catches more without
more false alarms.

## 7. Tune the harness

| To change | Edit |
|---|---|
| What counts as a defect here, the conventions, the security checklist, the repro limits, the severity, the triage, the paths to skip, the check commands | `REVIEW.md` |
| The smells of the slop lens | `.claude/review/slop-detector.md` |
| What one lens looks for, in every repository | its file in `.claude/agents/` |
| The steps of a review | `.claude/skills/reviewing-features/SKILL.md` |

Put what is true of colinfer only into `REVIEW.md`. Put what is true of every repository into the skill, the agents or
the script, and export again (section 8).

Change the text of the harness from the eval set, not from an impression. First add a case that shows the gap: a
bug that the harness missed, or a false alarm on a clean case. Run the eval and keep the score. Then write the
least text that closes the gap, and run the eval again.

The skill follows the rules for skills: a gerund name, a description that says what it does and when to use it, a
SKILL.md under 500 lines with a table of contents, and reference files that it names directly. `eval-mode.md` loads
only for an eval run. Keep these properties when you edit the skill.

`.claude/review/slop-detector.md` is a copy of `docs/slop-detector.md` of the vitesse-platform repository, at commit
8918d780. When you change one, copy it to the other.

Feasibility, in REVIEW.md, means how cheap and safe the fix is. It is not the probability of the failure: the severity
includes that.

## 8. Use the harness in another repository

One command copies the harness into another git repository:

```bash
python3 tools/review.py export ../vitesse-platform
```

It copies `.claude/agents/`, `.claude/skills/reviewing-features/`, `.claude/review/`, `.claude/hooks/quick-check.sh`
and `tools/review.py` into the target. It adds the Stop hook to the target's `.claude/settings.json`, and keeps the
other settings. Then it runs `init` there: a `REVIEW.md` from `.claude/review/REVIEW.template.md` when the target has
none, `.review/runs/` and `.review/cache/` in `.gitignore`, and `.review/ledger.jsonl merge=union` in `.gitattributes`.
It never overwrites the target's `REVIEW.md` or its ledger, so run it again after each change to the harness here.

Then, in the target:

1. Fill in each section of `REVIEW.md`. The template says what each section is for. The "Checks" section gives the
   lint and test commands; without it, the Stop hook does nothing and a review runs no check.
2. Commit `REVIEW.md`, `.claude/`, `tools/review.py`, `.gitignore` and `.gitattributes`.
3. Open a new Claude Code session there, and run `/reviewing-features`.

The script needs `python3` and `git`, and nothing else. In a repository without `REVIEW.md`, `/reviewing-features`
runs `tools/review.py init` and stops, so you can also start from there.

## 9. Later: pull requests on GitHub

When GitHub Actions works for this repository, `REVIEW.md` and the agents can also review pull requests:

- Code Review for Claude Code (Team and Enterprise plans) reads `REVIEW.md` from the base branch of a pull request. It
  posts its findings as inline comments.
- The Claude Code GitHub Action runs Claude Code on your own runner, with the agents of `.claude/agents/`.
- A GitHub runner has no GB10 GPU. The gate, the GPU tests and the memrace verifier stay with the local run.

A review never blocks a merge itself. To block a merge while a finding is open, turn on "Require conversation
resolution before merging" in the branch protection of `main`. Then each finding needs a fix, or a reply that closes
its thread with a reason.

---
name: reviewing-features
description: Reviews a finished feature of the repository for AI slop, security defects, and memory leaks or race conditions. Three finders read the diff in parallel. Refuters and sanitizer repro runs verify each serious finding. The skill fixes the confirmed findings that have an easy fix, and reviews its own fixes for at most three rounds. Use it when a feature is done and the user asks for a code review of the branch or the diff. Also use it for a slop check, a security audit, a leak or race review, or the eval set of the reviewers.
argument-hint: "[base-commit] [report] | eval [baseline]"
disable-model-invocation: true
model: opus
allowed-tools: Bash(python3 tools/review.py *) Bash(.claude/hooks/quick-check.sh) Bash(git blame *) Bash(git status *)
---

# Reviewing a finished feature

You run the review of one feature. You start the agents, collect their answers and apply the rules of the review.

The arguments are: `$ARGUMENTS`

- No argument: the base is the merge base with `main`. On `main` itself, the scope is the uncommitted work.
- A commit or a branch: the base.
- `report`: stop after step 6, without fixes.
- `eval` or `eval baseline`: do not do the steps below. Read [eval-mode.md](${CLAUDE_SKILL_DIR}/eval-mode.md) and
  follow it.

## Contents

1. [Scope](#1-scope): freeze the diff into a run directory.
2. [Gate](#2-gate): run the checks, and stop if they fail.
3. [Find](#3-find): three finders in parallel.
4. [Merge](#4-merge): one list, without duplicates.
5. [Verify](#5-verify): refuter votes, and repro runs for leaks and races.
6. [Record and triage](#6-record-and-triage): the ledger, and what to fix.
7. [Fix](#7-fix): fix by class, with a test.
8. [Review the fixes](#8-review-the-fixes): rounds 2 and 3, on the fix diff only.
9. [Summary](#9-summary): the counts first.
10. [Close the loop](#10-close-the-loop): rules for the session that builds the next feature.

## Files

This list names each file that the review uses. No file in it sends you to a further file.

| File | When |
|---|---|
| `tools/review.py` | Run it with `python3`, from the root of the repository. Do not read it: this file gives its commands and outputs. |
| [REVIEW.md](${CLAUDE_PROJECT_DIR}/REVIEW.md) | The contract of the repository. Read it before step 7: what a fix must not change, and the severity and feasibility levels. |
| [finding-format.md](${CLAUDE_PROJECT_DIR}/.claude/review/finding-format.md) | Read it before step 4: the fields of a finding. |
| [eval-mode.md](${CLAUDE_SKILL_DIR}/eval-mode.md) | Read it only for the argument `eval`. |
| `.claude/agents/<name>.md` | Start each agent by its name with the Agent tool. Do not read these files: each agent reads its own. |

An agent does not see this conversation. Give it everything that it needs in its prompt. Start agents in the
foreground (`run_in_background: false`). Several Agent calls in one message run in parallel and return together.

## 1. Scope

If the root of the repository has no `REVIEW.md`, run `python3 tools/review.py init`. It writes REVIEW.md from the
template (`.claude/review/REVIEW.template.md`), and the `.review` lines of `.gitignore` and `.gitattributes`. Tell the
user to fill in the sections of REVIEW.md, and stop.

Run `python3 tools/review.py scope [BASE]`. It prints the run id `RUN` and its directory `D` (`.review/runs/RUN`). Read
`D/scope.json`: it holds the base commit `B`, the snapshot of the working tree and the files.

If the diff has thousands of changed lines, tell the user, and ask whether to review it in one run or in parts.

## 2. Gate

Run `python3 tools/review.py gate RUN`. It reads the "Checks" section of REVIEW.md and prints one line:

- `quick`: run `.claude/hooks/quick-check.sh`. It runs the lint command, then the test command, and records a pass for the
  Stop hook. Give the Bash call `timeout: 300000`, because the default limit is 2 minutes. Exit status 3 means that
  the tests did not run for lack of memory. Record it in the summary.
- `full: <command>`: a file of the scope needs the full check. Run the command in the foreground with
  `timeout: 600000`, the maximum of the Bash tool. Do not start it in the background: the turn then ends, and the Stop
  hook runs the quick check next to it. If the "Checks" section says that the full check needs resources that are not
  free (memory, a stopped server), ask the user to select one: free them and run the full check, or run the quick
  check and record in the summary which checks did not run.
- `none`: REVIEW.md has no "Checks" section. Record in the summary that no check ran.

If the check fails, stop and report the failure. Do not review code that fails its checks.

## 3. Find

Run `python3 tools/review.py known RUN`. Its output is `KNOWN`.

Start the three finders in one message: `slop-finder`, `security-finder` and
`memrace-finder`. Give each one this prompt:

> Review run RUN, round 1, base B. The scope is D/scope.json and the patch is D/diff.patch. Read the code in the
> working tree of this repository. Do not report these known findings again:
> KNOWN

Save the JSON array of each answer as `D/r1-<lens>.json`. If an answer is not valid JSON, ask that agent once more
(SendMessage). If the second answer is not valid either, go on without that lens and record it in the summary.

## 4. Merge

Put the findings of the three lenses in one list. Two findings on the same lines with the same root cause are one
finding: keep the one with the higher severity, and add the place of the other one to its `other_instances`. Remove a
finding that `KNOWN` already has.

## 5. Verify

Verify a finding when its severity is critical, or when its severity is high and the diff introduced it. Give each
other finding `"verdict": "unverified"`.

**Slop and security findings.** Start two `refuter` agents for each finding, all in one message, at
most ten at a time. Give each one this prompt:

> Run RUN, base B, scope D/scope.json. Try to disprove this finding: <the finding as JSON>

Count the votes:

- Two `stands`: `"verdict": "confirmed"`. Two `refuted`: `"verdict": "refuted"`.
- Any other pair: start a third refuter. Two of three `stands` is confirmed, two of three `refuted` is refuted, and
  anything else is unverified.
- Record `"votes"` (for example `"2/2 stands"`) and, for a refuted finding, the refuters' reason and evidence in
  `"refutation"`.
- When two refuters say `pre-existing`, set `introduced` to `pre-existing`. When two say that the severity is lower,
  lower it by one level.

**Memrace findings.**

> **CAUTION:** A verifier runs tests under a sanitizer or in a stress loop, and the "Repro" section of REVIEW.md names
> the limits of the machine. Start one `memrace-verifier` at a time. Do not start two at once, and do
> not start one while the full check runs.

1. Run `python3 tools/review.py snapshot`. Its output is `PRE`.
2. For the finding with the number n in the list, start `memrace-verifier` with this prompt:

   > Run RUN, base B. Verify this finding with a run. Write your files only in D/repro/n/.
   > <the finding as JSON>

3. Apply the verdict:
   - `confirmed`: `"verdict": "confirmed"`. Keep `repro`, `command` and `output` in the finding.
   - `not-reproduced`: `"verdict": "unverified"`, the severity one level lower, and the verifier's notes in `"repro_notes"`.
   - `needs-model`: verify the finding with two refuters, as for a security finding.
4. After the last verifier, run `python3 tools/review.py snapshot` again. If the output differs from `PRE`, a verifier changed
   a file of the repository. Show `python3 tools/review.py diff PRE <the new snapshot>` to the user, and ask what to do before
   you go on.

## 6. Record and triage

Write the list, with the verdicts, to `D/r1.json`. Run `python3 tools/review.py add RUN D/r1.json --round 1`. It prints the id
of each finding.

Sort the findings. This is the rule of the "Triage" section of REVIEW.md:

- **Fix:** confirmed, critical or high, feasibility high, introduced by the diff.
- **Your decision:** confirmed, critical or high, feasibility medium or low. Also a confirmed critical finding that the
  diff did not introduce. Leave these open.
- **Backlog:** the other findings that the diff did not introduce. Run `python3 tools/review.py decide ID backlog`.
- **Refuted:** run `python3 tools/review.py decide ID wrong "refuted <votes>: <reason>"`.
- **Logged:** the medium, low and unverified findings. Leave these open.

With the argument `report`, go to step 9.

## 7. Fix

Run `python3 tools/review.py snapshot`. Its output is `S_BEFORE`.

For each finding to fix:

1. Fix the cited line and each place in `other_instances`. Then search the files of the scope for more places of the
   same `bug_class`, and fix them too.
2. Make the smallest fix, in the style of the code around it. Do not refactor around it.
3. Add a test that fails on the old code, as the repository's other fix commits do. A confirmed memrace finding already
   has one: move its repro from `D/repro/n/` into the test file of the module, with a name that says the behavior, as
   the existing tests do.

Then run the check of step 2 again. If it fails, fix the cause. After three failed attempts, stop and report.

Run `python3 tools/review.py decide ID fixed "<what changed; the test>"` for each fixed finding. Run `python3 tools/review.py snapshot`.
Its output is `S_AFTER`.

## 8. Review the fixes

Rounds 2 and 3 review the fix diff only. For round N:

1. Run `python3 tools/review.py diff S_BEFORE S_AFTER > D/rN.patch`, and `python3 tools/review.py known RUN`.
2. Start the three finders in one message, with this prompt:

   > Review run RUN, round N, base B. Review only the fix diff D/rN.patch: the changes that fixed the findings of the
   > round before. The scope of the feature is D/scope.json. Do not report these known findings again:
   > <the output of known>

3. Do steps 4 to 7 for the new findings, with `D/rN-<lens>.json`, `D/rN.json` and `--round N`.

Stop when a round has no finding to fix. After round 3, do not fix: list the new findings to fix under "Rethink" in
the summary. Tell the user that each round of fixes makes new defects, so the approach needs a second look.

## 9. Summary

Start with one line of counts, for example: "Review RUN: 9 findings. 3 fixed, 1 needs your decision, 2 in the backlog,
2 refuted, 1 logged." When there is nothing to fix and nothing to decide, start with "Nothing to fix."

Then give these sections, each one only when it has items:

- **Fixed:** the id, the title, the `file:line`, and the test that shows the fix.
- **Needs your decision:** the id, the severity, the feasibility, and why the fix is not easy.
- **Rethink:** the findings after round 3.
- **Backlog:** the findings that the diff did not introduce.
- **Logged:** the medium, low and unverified findings, one line each.
- **Refuted:** one line each.
- **Checks:** the check that ran, and what did not run.

Write the summary to `D/summary.md` too. Tell the user how to record a decision:
`python3 tools/review.py decide ID wrong|intentional|not-worth-it|backlog|fixed "note"`. Do not commit: the user does.

## 10. Close the loop

Run `python3 tools/review.py recurring`. It lists the bug classes that got a fix in two or more runs. For each class that
`.claude/rules/review-findings.md` of the repository does not cover yet, propose one rule for the session that builds
the next feature. Example: "Check each value of a request at the HTTP layer, with a 400, before it reaches the worker."
Ask the user before you write the file. Give a rule `paths:` frontmatter when its class belongs to some files only.

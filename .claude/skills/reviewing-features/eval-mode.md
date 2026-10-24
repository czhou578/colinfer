# Eval mode

The eval set holds known bugs at old commits (`.review/evals/cases.jsonl`). Two runs use it:

- `eval`: the harness. The finders and the refuters review each case.
- `eval baseline`: Claude without the harness. One plain agent reviews each case with the instructions of the manual
  loop that the harness replaced: the slop detector, then a security audit, then a review for leaks and races.

Compare the two scores. The harness must catch more than the baseline, without more false alarms. A part of the
harness that does not change the score is not worth its text: tell the user which cases each part made a difference
on.

Do not write to the ledger in eval mode.

## The harness: `eval`

1. Run `python3 tools/review.py eval-setup`. It prints the run `RUN` and its directory `D`, and for each case its name,
   its worktree `T`, its commit `C` and the paths to review.
2. Start the three finders (`slop-finder`, `security-finder`, `memrace-finder`) for each case, at most nine agents in
   one message. Give each one this prompt:

   > Review eval case NAME. The code is in the directory T, a checkout of commit C. Review the files PATHS at that
   > commit as if each whole file were the feature diff. Read the code only in T, not in the working tree. Read
   > REVIEW.md and .claude/review/ in the working tree. There are no known findings.

3. Start two `refuter` agents for each finding of each lens, also for memrace: eval mode makes no repro runs. Tell
   each refuter that the code is in `T`, and that there is no base commit, so it skips the check of `introduced`. Count
   the votes as in step 5 of the review.
4. Write `D/results.json`: `{"<case name>": [the findings, with their verdicts]}`.
5. Run `python3 tools/review.py eval-score RUN`. It prints the score, the score before it, and the latest baseline
   score, and removes the worktrees.

## The baseline: `eval baseline`

1. Run `python3 tools/review.py eval-setup --baseline`. It prints the same lines as for the harness. The history
   records the run as `baseline`, not with the hash of the prompt files.
2. Start one `general-purpose` agent for each case, at most nine in one message. Give it this prompt, and nothing else
   from the harness:

   > The code is in the directory T, a checkout of commit C. Review the files PATHS at that commit. Read the code only
   > in T. First apply the slop detector in .claude/review/slop-detector.md of the working tree. Then do a security
   > audit. Then look for memory leaks and race conditions.
   > Your final message is one fenced json block: an array with one object per finding, with the keys lens (slop,
   > security or memrace), title, file (the path as in PATHS), line, severity (critical, high, medium or low) and
   > failure_scenario.

3. Do not start refuters: the manual loop had none. Give each finding `"verdict": "unverified"`.
4. Write `D/results.json`, as for the harness.
5. Run `python3 tools/review.py eval-score RUN`. It prints the baseline score, the baseline before it, and the latest
   score of the harness.

## The report

Give the two scores, one line each. Then, for each expected finding, say which run caught it: both, the harness only,
the baseline only, or neither. A finding that only the baseline caught shows a gap in the harness. A case that both
miss needs a better prompt, or a better case.

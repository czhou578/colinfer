#!/usr/bin/env bash
# The quick regression check of the review harness (docs/review.md): the lint command, then the test command of the
# "Checks" section of REVIEW.md. tools/review.py does the work. This file is the entry point that .claude/settings.json
# names, so the settings of a project do not change when the script does.
#   .claude/hooks/quick-check.sh               # run the check now (the gate of /reviewing-features); records the pass
#   .claude/hooks/quick-check.sh --stop-hook   # as the Stop hook: the hook's JSON on stdin, exit 2 blocks the turn's end
# The Stop hook does nothing without a "Checks" section, when the code paths equal HEAD, when they equal the code of the
# last pass (.review/cache/passed), or with REVIEW_STOP_CHECK=0 (the night shift sets it). It blocks at most three times
# in a row; then the turn ends with a message. The output of the last check is in .review/cache/last.log.
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
if [ "${1:-}" = "--stop-hook" ]; then
    exec python3 tools/review.py stop-hook
fi
exec python3 tools/review.py check

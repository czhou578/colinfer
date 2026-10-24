#!/usr/bin/env bash
# The quick regression check of the review harness (docs/review.md): ruff, then pytest, as tests/check.sh --quick does.
# It records the code that it passed on, so the Stop hook does not run it again on the same code.
#   .claude/hooks/quick-check.sh               # run the check now (the gate of /reviewing-features)
#   .claude/hooks/quick-check.sh --stop-hook   # as the Stop hook: the hook's JSON on stdin, exit 2 blocks the turn's end
# The Stop hook does nothing when the code paths equal HEAD, when they equal the code of the last pass, or with
# COLINFER_STOP_CHECK=0 (the night shift sets it). It blocks at most MAX_BLOCKS times in a row; then the turn ends.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
CODE=(engine tests tools bench csrc pyproject.toml)
STATE=.cache/quick-check
MIN_GB=16       # the pytest GPU tests next to the server: the night shift's min_free_gb
MAX_BLOCKS=3
mkdir -p "$STATE"

check() {
    uv run ruff check engine tests tools bench || return 1
    local avail_gb
    avail_gb=$(awk '/MemAvailable/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
    if [ "$avail_gb" -lt "$MIN_GB" ]; then
        echo "quick-check: pytest skipped, ${avail_gb} GB of memory available (the GPU tests need ${MIN_GB} GB)"
        return 3
    fi
    uv run pytest tests/ -q -x
}

if [ "${1:-}" != "--stop-hook" ]; then
    fp=$(python3 tools/review.py snapshot --paths "${CODE[@]}") || exit 1
    check 2>&1 | tee "$STATE/last.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && echo "$fp" > "$STATE/passed"
    exit "$rc"
fi

[ "${COLINFER_STOP_CHECK:-1}" = 0 ] && exit 0
active=$(jq -r '.stop_hook_active // false' 2>/dev/null || echo false)
[ -z "$(git status --porcelain -- "${CODE[@]}")" ] && exit 0
fp=$(python3 tools/review.py snapshot --paths "${CODE[@]}") || exit 0
[ "$fp" = "$(cat "$STATE/passed" 2>/dev/null)" ] && exit 0
[ "$active" = true ] || echo 0 > "$STATE/blocks"   # a new turn: the count starts again
blocks=$(cat "$STATE/blocks" 2>/dev/null || echo 0)
if [ "$blocks" -ge "$MAX_BLOCKS" ]; then
    echo "{\"systemMessage\": \"quick-check: still failing after $MAX_BLOCKS attempts; the turn ends. Output: $STATE/last.log\"}"
    exit 0
fi
check > "$STATE/last.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    echo "$fp" > "$STATE/passed"
    exit 0
fi
if [ "$rc" -eq 3 ]; then
    echo "{\"systemMessage\": \"$(tail -n 1 "$STATE/last.log")\"}"
    exit 0
fi
echo $((blocks + 1)) > "$STATE/blocks"
{
    echo "quick-check failed (ruff, then pytest -x). Fix the cause before you end the turn. The last lines of the output:"
    tail -n 60 "$STATE/last.log"
} >&2
exit 2

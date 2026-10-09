#!/usr/bin/env bash
# The regression checks of the engine, in order. The first failure stops the script with a non-zero status.
#   1. ruff: pyflakes, import order, line length
#   2. pytest: the kernels against their references, bit-identity properties, the scheduler and the HTTP layer on stubs (~1 min)
#   3. golden.py: 21 recorded requests, bit for bit, with speculation and without (~2 x 2 min)
#   4. scheduler_check.py: the scheduler against uncached single requests on the real model (~4 min)
# Steps 3 and 4 need ~40 GB of free GPU memory: stop the server and any other GPU job first. An over-commit of the
# unified memory can power off the machine, so the script refuses to run them with less than 50 GB available.
#   tests/check.sh            # everything (~8 min)
#   tests/check.sh --quick    # steps 1 and 2 only
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check engine tests tools bench
uv run pytest tests/ -q
if [ "${1:-}" = "--quick" ]; then
    exit 0
fi
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt $((50 * 1024 * 1024)) ]; then
    echo "check.sh: $((avail_kb / 1024 / 1024)) GB of memory available; the model checks need ~40 GB free (stop the server?)" >&2
    exit 1
fi
uv run python tests/golden.py check
uv run python tests/golden.py check --plain
uv run python tests/scheduler_check.py
echo "ALL CHECKS PASSED"

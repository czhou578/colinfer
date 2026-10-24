#!/usr/bin/env bash
# The regression checks of the engine, in order. The first failure stops the script with a non-zero status.
#   1. ruff: pyflakes, import order, line length
#   2. pytest: the kernels against their references, bit-identity properties, the scheduler and the HTTP layer on stubs (~1 min)
#   3. golden.py: 21 recorded requests, bit for bit, with speculation and without (~2 x 2 min)
#   4. scheduler_check.py: the scheduler against uncached single requests on the server's engine, YaRN included (~3 min)
# Step 3 needs ~40 GB of memory. Step 4 needs ~90 GB: it builds the server's 3 slots of 524288 tokens (measured
# 2026-10-23: 89 GB at the peak). Stop the server and any other GPU job first. An over-commit of the unified memory can
# power off the machine, so the script refuses to run them with less than 105 GB available (the need, plus 16 GB).
#   tests/check.sh            # everything (~7 min)
#   tests/check.sh --quick    # steps 1 and 2 only
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check engine tests tools bench
uv run pytest tests/ -q
if [ "${1:-}" = "--quick" ]; then
    exit 0
fi
avail_b=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 ))
if [ "$avail_b" -lt $((105 * 1000000000)) ]; then
    echo "check.sh: $((avail_b / 1000000000)) GB of memory available; the model checks need 105 GB (89 GB measured, plus 16 GB in reserve; GB = 10^9 bytes; stop the server?)" >&2
    exit 1
fi
uv run python tests/golden.py check
uv run python tests/golden.py check --plain
uv run python tests/scheduler_check.py
echo "ALL CHECKS PASSED"

"""The PLE row reader (csrc/ple_io.cpp, ops().ple_gather) over a synthetic file: plain pread and O_DIRECT reads give the
file's bytes, for rows inside a page, rows across a page boundary and the last row of the file. Run: uv run pytest
tests/test_ple_io.py -q"""
import os

import numpy as np
import pytest
import torch

ROW = 160


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    path = tmp_path_factory.mktemp("ple") / "rows.bin"
    data = np.random.default_rng(0).integers(0, 256, size=4096 * 64 + 160 * 3, dtype=np.uint8)  # not a whole number of pages
    data.tofile(path)
    return str(path), data


def _offsets(size):
    rng = np.random.default_rng(1)
    rows = list(rng.integers(0, size // ROW, size=500) * ROW)
    rows += [4096 - 80, 2 * 4096 - 1, 0, size - ROW]  # across a page boundary, at the start, the last row
    rows += list(rng.integers(0, size - ROW, size=100))  # any byte offset
    return torch.tensor(rows, dtype=torch.int64)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("threads", [1, 32])
def test_rows_equal_file_bytes(table, direct, threads):
    from engine.kernels import ops
    path, data = table
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:  # a file system without O_DIRECT (tmpfs)
        pytest.skip(f"O_DIRECT: {e}")
    try:
        off = _offsets(len(data))
        out = torch.empty(len(off), ROW, dtype=torch.uint8)
        ops().ple_gather(fd, off, out, direct, threads)
        want = np.stack([data[o:o + ROW] for o in off.tolist()])
        assert np.array_equal(out.numpy(), want)
        empty = torch.empty(0, ROW, dtype=torch.uint8)
        ops().ple_gather(fd, torch.empty(0, dtype=torch.int64), empty, direct, threads)
    finally:
        os.close(fd)


def test_read_past_the_end_fails(table):
    from engine.kernels import ops
    path, data = table
    fd = os.open(path, os.O_RDONLY)
    try:
        out = torch.empty(2, ROW, dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="read failed"):
            ops().ple_gather(fd, torch.tensor([0, len(data) - 10]), out)
    finally:
        os.close(fd)


def test_many_small_calls(table):
    """Many tiny jobs with changing thread counts: a worker that finishes early must not take another worker's wakeup
    (a lost wakeup hangs the call)."""
    from engine.kernels import ops
    path, data = table
    fd = os.open(path, os.O_RDONLY)
    try:
        off = torch.tensor([0, 160, 4000, 8000], dtype=torch.int64)
        out = torch.empty(4, ROW, dtype=torch.uint8)
        want = np.stack([data[o:o + ROW] for o in off.tolist()])
        for i in range(3000):
            ops().ple_gather(fd, off, out, False, 1 + (i % 4) * 11)
        assert np.array_equal(out.numpy(), want)
    finally:
        os.close(fd)

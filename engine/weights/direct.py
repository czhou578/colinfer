"""Direct reads of safetensors files into device tensors.

safetensors' own path maps a file and copies each tensor out of the mapping: page faults and copies from pageable
memory, ~3-4 GB/s here, and the file's pages stay in the page cache afterwards (on this machine's unified memory, that
cache competes with everything else until the kernel reclaims it). This reader opens the file with O_DIRECT, reads it
in large aligned chunks into pinned staging buffers on a few reader threads (an NVMe needs several reads in flight),
and copies each tensor's bytes into its preallocated device tensor asynchronously: the reads and the copies overlap,
and the page cache keeps nothing.
Where O_DIRECT is refused (tmpfs), it reads normally and drops the file's pages afterwards.
"""
from __future__ import annotations

import bisect
import errno
import json
import os
import queue
import struct
import threading

import torch

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E5M2": torch.float8_e5m2, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
          "BOOL": torch.bool}
ALIGN = 4096        # direct I/O: file offsets, lengths and buffer addresses
CHUNK = 64 << 20    # bytes per staging buffer
BUFFERS = 6         # staging buffers in flight
THREADS = 3         # reader threads: reads in flight on the NVMe
GAP = 8 << 20       # unwanted bytes between wanted tensors that are read through rather than skipped
_bufs: list[torch.Tensor] = []  # the staging buffers, kept between the files of one load (pinning 64 MB takes ~20 ms)


def _buffers() -> list[torch.Tensor]:
    global _bufs
    if len(_bufs) != BUFFERS or _bufs[0].numel() != CHUNK:
        _bufs = [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(BUFFERS)]
    return _bufs


def release_buffers():
    """Frees the staging buffers (BUFFERS x CHUNK of pinned memory): call when the loading is done."""
    global _bufs
    _bufs = []


def header(path: str) -> tuple[dict, int]:
    """({name: {"dtype", "shape", "data_offsets"}}, the file offset of the data section)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def tensor_names(path: str) -> list[str]:
    return list(header(path)[0])


def _open(path: str) -> tuple[int, bool]:
    """(fd, direct): opened with O_DIRECT where the filesystem allows it."""
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT), True
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        return os.open(path, os.O_RDONLY), False


def read_tensors(path: str, names=None, device="cuda", into: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """The tensors of a safetensors file (all, or `names`) on `device`, read directly (see the module docstring).

    into: {name: destination} for some of the names read. The bytes of such a tensor go straight into its destination
    (a preallocated contiguous tensor of the same dtype and element count, on any device, e.g. a slot of a stacked
    tensor), and the result leaves the name out. Destinations of names not read are ignored."""
    h, base = header(path)
    want = [(v["data_offsets"][0], v["data_offsets"][1], k) for k, v in h.items() if names is None or k in names]
    if names is not None and len(want) != len(set(names)):
        raise KeyError(f"{path}: missing {sorted(set(names) - set(k for _, _, k in want))[:5]}")
    want.sort()
    into = into or {}
    out, spans = {}, []  # spans: [file offset, length, name, destination's flat uint8 view] of each non-empty tensor, in file order
    for b, e, k in want:
        v = h[k]
        dt = DTYPES[v["dtype"]]
        n = 1
        for d in v["shape"]:
            n *= d
        if n * torch.empty(0, dtype=dt).element_size() != e - b:
            raise ValueError(f"{path}: {k}: {v['shape']} {v['dtype']} does not match its {e - b} bytes")
        dst = into.get(k)
        if dst is not None:
            if dst.dtype != dt or dst.numel() != n or not dst.is_contiguous():
                raise ValueError(f"{path}: {k}: the destination {dst.dtype} {tuple(dst.shape)} (contiguous {dst.is_contiguous()}) "
                                 f"does not take {v['dtype']} {v['shape']}")
            if e > b:
                spans.append([base + b, e - b, k, dst.view(-1).view(torch.uint8)])
        elif e > b:
            spans.append([base + b, e - b, k, None])  # allocated when its first chunk arrives: 10 GB of cudaMalloc overlaps the reads
        else:
            out[k] = torch.empty(v["shape"], dtype=dt, device=device)
    if not spans:
        return out

    def dest(j):
        off, m, k, dst = spans[j]
        if dst is None:
            v = h[k]
            out[k] = torch.empty(v["shape"], dtype=DTYPES[v["dtype"]], device=device)
            dst = spans[j][3] = out[k].view(-1).view(torch.uint8)  # view(-1) first: scalars have no dim to retype
        return off, m, dst
    # chunks: aligned, through gaps of up to GAP bytes, in reading order
    runs: list[list[int]] = []  # [first byte, last byte) of the file to read
    for off, n, _, _ in spans:
        if runs and off - runs[-1][1] <= GAP:
            runs[-1][1] = max(runs[-1][1], off + n)
        else:
            runs.append([off, off + n])
    chunks = []
    for lo, hi in runs:
        c0 = lo // ALIGN * ALIGN
        while c0 < hi:
            chunks.append((c0, min(CHUNK, -(-(hi - c0) // ALIGN) * ALIGN)))
            c0 += CHUNK
    size = os.path.getsize(path)
    fd, direct = _open(path)
    bufs = _buffers()
    if direct and any(b.data_ptr() % ALIGN for b in bufs):
        os.close(fd)
        fd, direct = os.open(path, os.O_RDONLY), False
    free: queue.Queue = queue.Queue()
    filled: queue.Queue = queue.Queue()
    todo: queue.Queue = queue.Queue()
    stop = threading.Event()
    for i in range(BUFFERS):
        free.put((i, None))
    for c in chunks:
        todo.put(c)
    for _ in range(THREADS):
        todo.put(None)

    def reader():
        try:
            while True:
                c = todo.get()
                if c is None or stop.is_set():
                    return
                c0, n = c
                i, ev = free.get()
                if stop.is_set():
                    return
                if ev is not None:
                    ev.synchronize()  # the copies out of this buffer are done
                n = min(n, size - c0) if not direct else n  # a direct read past the end returns what there is
                mv, got = memoryview(bufs[i].numpy())[:n], 0
                while got < min(n, size - c0):
                    r = os.preadv(fd, [mv[got:]], c0 + got)
                    if r <= 0:
                        raise OSError(f"{path}: short read at {c0 + got}")
                    got += r
                filled.put((i, c0, got))
        except BaseException as e:  # noqa: BLE001  reported by the consumer
            filled.put(e)

    ths = [threading.Thread(target=reader, name=f"direct-reader-{t}", daemon=True) for t in range(THREADS)]
    for th in ths:
        th.start()
    ends = [off + n for off, n, _, _ in spans]  # chunks arrive in any order: the spans a chunk touches, by bisection
    try:
        for _ in range(len(chunks)):
            item = filled.get()
            if isinstance(item, BaseException):
                raise item
            i, c0, n = item
            buf, c1 = bufs[i], c0 + n
            j = bisect.bisect_right(ends, c0)
            while j < len(spans) and spans[j][0] < c1:
                off, m, dst = dest(j)
                lo, hi = max(off, c0), min(off + m, c1)
                dst[lo - off:hi - off].copy_(buf[lo - c0:hi - c0], non_blocking=True)
                j += 1
            ev = torch.cuda.Event()
            ev.record()
            free.put((i, ev))
        torch.cuda.current_stream().synchronize()
    finally:
        stop.set()
        for _ in ths:  # unblock readers if the copies failed
            free.put((0, None))
            todo.put(None)
        for th in ths:
            th.join()
        if not direct:
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        os.close(fd)
    return out

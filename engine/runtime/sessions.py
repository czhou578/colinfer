"""Conversation cache on disk: the KV rows and GDN snapshots of past conversations, on the NVMe. A conversation that
left the slots (other conversations took them, or the server restarted) then restores in about a second instead of a
full prefill. A 128k-token conversation is ~4.4 GB: ~0.4 s at the ~12 GB/s of direct reads on this machine, against
~66 s of prefill.

What the store keeps (one directory per engine configuration, see `fingerprint`):

* KV blocks. Block j holds the rows [jB, (j+1)B) of the attention layers of the target (K and V, fp8 as cached) and
  the rows of the drafter shifted by one, [jB - 1, (j+1)B - 1): the drafter's row i reads token i + 1. Thus every byte
  of the block depends only on the tokens [0, (j+1)B). The hash chain of those tokens (and the cache salt) names the
  block, so conversations that share a prefix (a system prompt) share its blocks.
* Checkpoints. The GDN state (conv windows and recurrent states) and the drafter's input hidden state at a prefix of n
  tokens, plus the KV rows of its partial block [n - n % B, n). The hash of the tokens [0, n) names the checkpoint.

A restore of n tokens reads the n // B blocks and the checkpoint. These are exactly the bytes that the slot held, so a
reply after a restore from disk is the reply after a restore from memory, bit for bit.

The scheduler (engine/runtime/scheduler.py) saves the checkpoints that later requests restore: the end of each reply and
the message boundaries of each prompt, for prefixes of at least `min_tokens`. It saves them when a request ends, through
a writer thread. The writer reads the KV rows straight from the slot, so the scheduler calls wait() before it rewrites
rows of a slot that a pending save still has to read. A restore runs on the engine thread at admission, when the disk
holds a longer prefix of the prompt than the memory checkpoints. A failed read (a file evicted meanwhile, a disk error)
falls back to prefill.

The files are written with direct I/O, which leaves the page cache alone (the GPU shares this memory), to a temporary
name that is renamed when complete. The total size stays under a budget: the least recently used files go first,
blocks and checkpoints alike. A checkpoint whose blocks are gone no longer matches, and goes in its turn.
"""
from __future__ import annotations

import collections
import dataclasses
import errno
import hashlib
import json
import os
import queue
import threading
import time

import numpy as np
import torch

ROOT = os.path.expanduser("~/.cache/colinfer/sessions")
BLOCK = 1024        # tokens per KV block: 34 MB with the drafter's rows
MIN_TOKENS = 2048   # shorter prefixes are not worth a checkpoint of ~154 MB: their prefill takes ~0.6 s
MIN_GAIN = 256      # a disk checkpoint must beat the best memory checkpoint by this many tokens (any prefill costs a weight pass)
MAX_PENDING = 4     # saves that wait for the writer before save() blocks: each holds a copy of the GDN state on the GPU
ALIGN = 4096        # direct I/O: buffer addresses, file offsets and lengths
HEADER = 4096       # a JSON header at the start of each file, padded with spaces
VERSION = 2         # of the file layout and the token hashing


def _open(path: str, flags: int) -> int:
    """os.open with O_DIRECT where the filesystem supports it (not tmpfs)."""
    try:
        return os.open(path, flags | os.O_DIRECT, 0o644)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        return os.open(path, flags, 0o644)


def _h(*parts: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p)
    return h.digest()


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """A contiguous tensor as a flat uint8 view."""
    return t.reshape(-1).view(torch.uint8)


@dataclasses.dataclass
class Hit:
    """A checkpoint of n tokens whose blocks are all on disk."""
    n: int
    path: str
    reply: bool        # taken at the end of a reply (the scheduler's Checkpoint.reply)
    blocks: list       # the paths of blocks 0 .. n // B - 1


@dataclasses.dataclass
class _Job:
    slot: int
    n: int
    key: bytes
    reply: bool
    chain: list
    gdn: torch.Tensor | None   # the GDN state and the drafter's hidden as one flat uint8 copy (device)
    ready: torch.cuda.Event


class SessionStore:
    """The disk cache of one engine. state: the scheduler's FastState (all slots); mst: the drafter's MtpState or None;
    fingerprint: what the cached bytes depend on (engine/runtime/build.py: the checkpoint, the decode weights, the
    drafter); budget: bytes on disk, over all configurations under root."""

    def __init__(self, state, mst, fingerprint: dict, budget: int, root: str = ROOT, block: int = BLOCK,
                 min_tokens: int = MIN_TOKENS, log=print):
        self.state, self.mst, self.budget, self.block, self.min_tokens, self.log = state, mst, budget, block, min_tokens, log
        self.min_gain = MIN_GAIN
        self.layers = sorted(state.k)
        self.gdn_layers = sorted(state.rec)
        shapes = {"kv": [list(state.k[i].shape[1:]) for i in self.layers], "kv_dtype": str(state.k[self.layers[0]].dtype),
                  "conv": [list(state.conv[i].shape[1:]) for i in self.gdn_layers], "rec": [list(state.rec[i].shape[1:]) for i in self.gdn_layers],
                  "drafter_kv": list(mst.k[0].shape[1:]) if mst is not None else None}
        fp = dict(fingerprint, version=VERSION, block=block, shapes=shapes)
        self.fp = hashlib.blake2b(json.dumps(fp, sort_keys=True).encode(), digest_size=8).hexdigest()
        self.root, self.dir = root, os.path.join(root, self.fp)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "fingerprint.json"), "w") as f:
            json.dump(fp, f, indent=1, sort_keys=True)
        row = sum(2 * _nbytes(state.k[i][0, :, 0]) for i in self.layers) + (2 * _nbytes(mst.k[0][0, :, 0]) if mst is not None else 0)
        self.gdn_bytes = sum(_nbytes(state.conv[i][0]) + _nbytes(state.rec[i][0]) for i in self.gdn_layers)
        self.h_dim = state.cfg.hidden_size if mst is not None else 0
        largest = HEADER + max(block * row, self.gdn_bytes + 2 * self.h_dim + (block - 1) * row)
        largest = -(-largest // ALIGN) * ALIGN
        self.wbuf = torch.empty(largest, dtype=torch.uint8, pin_memory=True)  # the writer's
        self.rbuf = [torch.empty(largest, dtype=torch.uint8, pin_memory=True) for _ in range(2)]  # restores alternate between them
        self.wstream = torch.cuda.Stream(state.pos_t.device)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.files: collections.OrderedDict[str, int] = collections.OrderedDict()  # path -> bytes, least recently used first
        self.meta: dict[str, tuple] = {}  # path -> ("b", key) | ("c", n, key, reply) | ("other",): another configuration's file
        self.blocks: dict[bytes, str] = {}
        self.ckpts: dict[int, dict[bytes, tuple[str, bool]]] = {}  # n -> key -> (path, reply)
        self.total = 0
        self.pending: list[_Job] = []
        self.jobs: queue.Queue = queue.Queue()
        self.stats = dict(restores=0, restored_tokens=0, saves=0, written_bytes=0, errors=0)
        self._scan()
        self._evict()
        threading.Thread(target=self._run, name="session-writer", daemon=True).start()
        log(f"[sessions] {self.dir}: {len(self.files)} files, {self.total / 1e9:.1f} GB of {budget / 1e9:.0f} GB")

    # ------------------------------------------------------------------------------------------------ keys
    def _root(self, salt: str | None) -> bytes:
        return _h(b"colinfer-sessions", self.fp.encode(), b"\0", (salt or "").encode())

    def _chain(self, toks: np.ndarray, salt: str | None, nb: int) -> list[bytes]:
        """chain[j] names the tokens [0, jB): chain[j + 1] is the key of block j."""
        B, c = self.block, [self._root(salt)]
        for j in range(nb):
            c.append(_h(c[-1], toks[j * B:(j + 1) * B].tobytes()))
        return c

    def _ckey(self, chain: list[bytes], toks: np.ndarray, n: int) -> bytes:
        nb = n // self.block
        return _h(b"ckpt", chain[nb], toks[nb * self.block:n].tobytes())

    # ------------------------------------------------------------------------------------------------ index
    def _add(self, path: str, size: int, meta: tuple):
        with self.lock:
            self.files[path] = size
            self.total += size
            self.meta[path] = meta
            if meta[0] == "b":
                self.blocks[meta[1]] = path
            elif meta[0] == "c":
                self.ckpts.setdefault(meta[1], {})[meta[2]] = (path, meta[3])

    def _forget(self, path: str):
        """Drops a file from the index (the lock is held)."""
        self.total -= self.files.pop(path, 0)
        meta = self.meta.pop(path, None)
        if meta is not None and meta[0] == "b":
            self.blocks.pop(meta[1], None)
        elif meta is not None and meta[0] == "c":
            self.ckpts.get(meta[1], {}).pop(meta[2], None)
            if not self.ckpts.get(meta[1], True):
                del self.ckpts[meta[1]]

    def _touch(self, paths):
        """Marks files as used now (the lock is held). The mtime keeps the order across restarts."""
        now = time.time()
        for p in paths:
            if p in self.files:
                self.files.move_to_end(p)
                try:
                    os.utime(p, (now, now))
                except OSError:
                    pass

    def _scan(self):
        """Indexes the files under root, oldest first: this configuration's for lookups, all of them for the budget."""
        found = []
        for d in os.listdir(self.root):
            full = os.path.join(self.root, d)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                p = os.path.join(full, f)
                if f.endswith(".tmp"):  # a write that a crash interrupted
                    os.remove(p)
                elif f.endswith((".kv", ".ck")):
                    st = os.stat(p)
                    found.append((st.st_mtime, p, st.st_size, f if full == self.dir else None))
        for _, p, size, f in sorted(found):
            meta = ("other",)
            try:
                if f is not None and f.startswith("b-"):
                    meta = ("b", bytes.fromhex(f[2:-3]))
                elif f is not None and f.startswith("c-"):
                    n, r, key = f[2:-3].split("-")
                    meta = ("c", int(n), bytes.fromhex(key), r == "1")
            except ValueError:
                pass
            self._add(p, size, meta)

    def _evict(self):
        while True:
            with self.lock:
                if self.total <= self.budget or not self.files:
                    return
                path = next(iter(self.files))
                self._forget(path)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------------------------------------ the slot's bytes
    def _kv(self, b: int, r0: int, r1: int) -> list[torch.Tensor]:
        """Device views of the KV rows [r0, r1) of slot b: the target's, then the drafter's rows [r0 - 1, r1 - 1)."""
        out = []
        for i in self.layers:
            for t in (self.state.k[i], self.state.v[i]):
                out += [t[b, h, r0:r1] for h in range(t.shape[1])]
        d0, d1 = max(r0 - 1, 0), r1 - 1
        if self.mst is not None and d1 > d0:
            for t in (self.mst.k[0], self.mst.v[0]):
                out += [t[b, h, d0:d1] for h in range(t.shape[1])]
        return out

    def _gdn(self, b: int, h: torch.Tensor | None) -> list[torch.Tensor]:
        return ([self.state.conv[i][b] for i in self.gdn_layers] + [self.state.rec[i][b] for i in self.gdn_layers]
                + ([h] if h is not None else []))

    # ------------------------------------------------------------------------------------------------ save (engine thread)
    def save(self, b: int, tokens, salt: str | None, reply: bool, conv: list, rec: list, h: torch.Tensor | None):
        """Queues the checkpoint of the prefix `tokens` of slot b. conv / rec: its GDN state per layer (device tensors,
        copied now), h: the drafter's input hidden state or None. The writer reads the KV rows [0, n) from the slot
        later (see wait())."""
        toks = np.asarray(tokens, dtype=np.int64)  # image tokens are ids past 2^31 (engine/model/vision.py)
        n = len(toks)
        chain = self._chain(toks, salt, n // self.block)
        key = self._ckey(chain, toks, n)
        with self.cv:
            e = self.ckpts.get(n, {}).get(key)
            if e is not None:
                self._touch([e[0]] + [self.blocks[c] for c in chain[1:] if c in self.blocks])
                return
            if any(j.key == key for j in self.pending):
                return
            while len(self.pending) >= MAX_PENDING:
                self.cv.wait()
        gdn = torch.cat([_bytes(t) for t in list(conv) + list(rec) + ([h] if h is not None and self.mst is not None else [])])
        job = _Job(b, n, key, reply, chain, gdn, torch.cuda.Event())
        job.ready.record()
        with self.lock:
            self.pending.append(job)
        self.jobs.put(job)

    def wait(self, b: int, n0: int = 0):
        """Returns once no pending save has to read rows of slot b at or past n0: the caller is about to rewrite them (a
        prefill from n0 also rewrites the drafter's row n0 - 1, which a save of n tokens reads only if n > n0)."""
        with self.cv:
            while any(j.slot == b and j.n > n0 for j in self.pending):
                self.cv.wait()

    def flush(self):
        """Returns once the writer has written everything queued."""
        with self.cv:
            while self.pending:
                self.cv.wait()

    # ------------------------------------------------------------------------------------------------ the writer thread
    @torch.inference_mode()  # the buffers are inference tensors when the engine thread made them under inference mode
    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                self.wstream.wait_event(job.ready)
                self._write_job(job)
                self.stats["saves"] += 1
                self._evict()  # before the job leaves `pending`: flush() returns with the budget kept
            except Exception as e:  # noqa: BLE001  a disk error costs the cache, never the engine
                self.stats["errors"] += 1
                self.log(f"[sessions] save of {job.n} tokens failed: {type(e).__name__}: {e}")
            finally:
                job.gdn = None
                with self.cv:
                    self.pending.remove(job)
                    self.cv.notify_all()

    def _write_job(self, j: _Job):
        B, nb = self.block, j.n // self.block
        for i in range(nb):
            key = j.chain[i + 1]
            with self.lock:
                have = self.blocks.get(key)
                if have is not None:
                    self._touch([have])
            if have is None:
                path = os.path.join(self.dir, f"b-{key.hex()}.kv")
                self._add(path, self._write(path, {"kind": "block", "rows": [i * B, (i + 1) * B]}, self._kv(j.slot, i * B, (i + 1) * B)),
                          ("b", key))
        path = os.path.join(self.dir, f"c-{j.n}-{int(j.reply)}-{j.key.hex()}.ck")
        size = self._write(path, {"kind": "ckpt", "n": j.n, "rows": [nb * B, j.n]}, [j.gdn] + self._kv(j.slot, nb * B, j.n))
        self._add(path, size, ("c", j.n, j.key, j.reply))

    def _write(self, path: str, header: dict, pieces: list[torch.Tensor]) -> int:
        buf, off = self.wbuf, HEADER
        with torch.cuda.stream(self.wstream):
            for t in pieces:
                m = _nbytes(t)
                buf[off:off + m].copy_(_bytes(t), non_blocking=True)
                off += m
        hdr = json.dumps(dict(header, v=VERSION, fp=self.fp, bytes=off - HEADER)).encode().ljust(HEADER)
        buf[:HEADER].copy_(torch.frombuffer(bytearray(hdr), dtype=torch.uint8))
        self.wstream.synchronize()
        size = -(-off // ALIGN) * ALIGN
        mv = memoryview(buf.numpy())[:size]
        tmp = path + ".tmp"
        fd = _open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            done = 0
            while done < size:
                done += os.write(fd, mv[done:])
            os.fdatasync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        self.stats["written_bytes"] += size
        return size

    # ------------------------------------------------------------------------------------------------ restore (engine thread)
    def lookup(self, prompt, salt: str | None, min_len: int) -> Hit | None:
        """The longest checkpoint on disk, longer than min_len, that is a proper prefix of the prompt and has all of its
        blocks."""
        P = len(prompt)
        toks = np.asarray(prompt, dtype=np.int64)
        with self.lock:
            pend = [j for j in self.pending if min_len < j.n < P]
        if pend:  # a save of this prompt's prefix is still being written: wait for it
            chain = self._chain(toks, salt, max(j.n for j in pend) // self.block)
            if any(self._ckey(chain, toks, j.n) == j.key for j in pend):
                self.flush()
        with self.lock:
            lens = sorted((n for n in self.ckpts if min_len < n < P), reverse=True)
        if not lens:
            return None
        chain = self._chain(toks, salt, lens[0] // self.block)
        for n in lens:
            key = self._ckey(chain, toks, n)
            with self.lock:
                e = self.ckpts.get(n, {}).get(key)
                if e is None:
                    continue
                blocks = [self.blocks.get(c) for c in chain[1:n // self.block + 1]]
                if None in blocks:
                    continue
                self._touch(blocks + [e[0]])
            return Hit(n, e[0], e[1], blocks)
        return None

    def load(self, hit: Hit, b: int, h: torch.Tensor | None, kv: bool = True) -> bool:
        """Restores the checkpoint into slot b: its GDN state, the drafter's hidden (into h) and, with kv, the KV rows
        [0, n). kv=False: the slot already holds the KV of these tokens. False (and a slot in an unknown state) when a
        file is missing or wrong."""
        B = self.block
        files = [(p, i * B, (i + 1) * B, False) for i, p in enumerate(hit.blocks)] if kv else []
        files.append((hit.path, hit.n // B * B, hit.n, True))
        evs: list = [None, None]
        try:
            for i, (path, r0, r1, ck) in enumerate(files):
                buf = self.rbuf[i % 2]
                if evs[i % 2] is not None:
                    evs[i % 2].synchronize()  # the copies out of this buffer are done
                hdr = self._read(path, buf)
                pieces = self._kv(b, r0, r1)
                want = sum(_nbytes(t) for t in pieces) + (self.gdn_bytes + 2 * self.h_dim if ck else 0)
                if (hdr.get("v"), hdr.get("fp"), hdr.get("kind"), hdr.get("rows"), hdr.get("bytes")) != (
                        VERSION, self.fp, "ckpt" if ck else "block", [r0, r1], want):
                    raise ValueError(f"{path}: unexpected header {hdr}")
                if ck:
                    pieces = self._gdn(b, h if self.mst is not None else None) + (pieces if kv else [])
                off = HEADER
                for t in pieces:
                    m = _nbytes(t)
                    _bytes(t).copy_(buf[off:off + m], non_blocking=True)
                    off += m
                evs[i % 2] = torch.cuda.Event()
                evs[i % 2].record()
            torch.cuda.current_stream().synchronize()
        except Exception as e:  # noqa: BLE001  a missing or bad file: the caller prefills instead
            torch.cuda.current_stream().synchronize()
            self.stats["errors"] += 1
            self.log(f"[sessions] restore of {hit.n} tokens failed: {type(e).__name__}: {e}")
            with self.lock:
                for p in [p for p, *_ in files]:
                    if not os.path.exists(p):
                        self._forget(p)
            return False
        self.stats["restores"] += 1
        self.stats["restored_tokens"] += hit.n
        return True

    def _read(self, path: str, buf: torch.Tensor) -> dict:
        fd = _open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size > buf.numel() or size < HEADER:
                raise ValueError(f"{path}: {size} bytes")
            mv = memoryview(buf.numpy())[:size]
            got = 0
            while got < size:
                r = os.readv(fd, [mv[got:]])
                if r <= 0:
                    raise ValueError(f"{path}: short read")
                got += r
        finally:
            os.close(fd)
        return json.loads(bytes(buf[:HEADER].numpy()).decode())

    def status(self) -> dict:
        with self.lock:
            n_ck = sum(len(d) for d in self.ckpts.values())
            return dict(dir=self.dir, files=len(self.files), checkpoints=n_ck, blocks=len(self.blocks), bytes=self.total, budget=self.budget,
                        pending=len(self.pending), **self.stats)

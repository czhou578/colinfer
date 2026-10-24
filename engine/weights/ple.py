"""The n-gram table of the PLE layer of Qwen3.8-Flash-Next (qwen4_exp): the hash of the row ids and the row reads.

The PLE layer (engine/model/qwen4exp.py PLELayer) embeds each token with 16 hashed n-gram heads: 8 bigram heads and 8
trigram heads over the token and its two predecessors. Each head has a prime-sized vocabulary of about 20M rows, and
the heads share one table of 320,001,536 rows x 160 dims (FP8 e4m3 with one bf16 scale, 51.2 GB). The hash is the one
of transformers' Qwen4ExpTextNGramEmbedding: splitmix64 multipliers per layer, an XOR of the products, the remainder
by the head's prime, plus the head's offset. ngram_params computes the primes, offsets and multipliers. ngram_rows
computes the ids in numpy int64; the torch path of the model (NgramHash) computes the same ids.

The context of a row is the two tokens before it in the sequence. An eos token starts a new segment: the trigram
context of the token after an eos is (eos, eos), as transformers' _shift_right_ignore_eos gives it. A sequence starts
with the context (eos, eos).

NgramTable reads rows from the checkpoint file, model-fp8-mtp-ple.safetensors. The table stays on disk. The file holds
the table as shards `<prefix>.shard_<i>.weight` [rows_per_shard, 160]. Row r is in shard r // rows_per_shard. The file
does not store the shards in index order, so the reader takes each shard's offset from the file's own header. A row
read is an O_DIRECT preadv of the 4 KB page(s) that hold the row, on a thread pool: O_DIRECT keeps the page cache
from holding 4 KB per 160 B row, and preadv releases the GIL. The value of a row is bf16(e4m3 * scale) (dequant).
"""
from __future__ import annotations

import functools
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from engine.weights.direct import DTYPES, _open, header

PAGE = 4096  # the O_DIRECT unit: file offsets, lengths and buffer addresses
NGRAM_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"  # the table of the checkpoint's one PLE layer

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(vocab_size: int, ngram_size: int, ple_index: int, seed: int) -> list[int]:
    """The odd hash multiplier of each n-gram position, as transformers' _build_layer_multipliers: each is below
    (2^63 - 1) // vocab_size, so a token id times a multiplier stays below 2^63."""
    half = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base = seed + _PRIME_1 * ple_index
    return [2 * (_splitmix64((base + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram_size)]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


@functools.lru_cache(maxsize=None)
def _primes_after(start: int, count: int) -> tuple[int, ...]:
    """The first `count` primes above `start` (transformers' _find_nth_prime_after(start, k) is the k-th of them)."""
    out, p = [], start
    while len(out) < count:
        p += 1
        if _is_prime(p):
            out.append(p)
    return tuple(out)


@dataclass(frozen=True)
class NgramParams:
    """The hash parameters of one PLE layer."""
    ngram_size: int            # the largest n-gram: 3 (bigram and trigram heads)
    heads_per_ngram: int       # 8
    sizes: tuple[int, ...]     # the prime vocabulary of each head, bigram heads first
    offsets: tuple[int, ...]   # the first row of each head in the shared table
    multipliers: tuple[int, ...]  # one per n-gram position
    eos: int                   # the token that starts a new segment
    vocab_size: int            # token ids are below this
    padded_rows: int           # the table's rows: the sum of the sizes, padded to make_ngram_vocab_size_divisible_by

    @property
    def heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram


def ngram_params(cfg, ple_index: int = 0) -> NgramParams:
    """The hash parameters of the PLE layer with index `ple_index` (its place in cfg.ple_layer_ids). cfg: a
    Qwen4ExpConfig, or anything with its n-gram fields."""
    heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    primes = _primes_after(cfg.ngram_vocab_size_base - 1, (ple_index + 1) * heads)[ple_index * heads:]
    offsets = tuple(int(x) for x in np.cumsum((0,) + primes[:-1]))
    total = sum(primes)
    div = cfg.make_ngram_vocab_size_divisible_by
    return NgramParams(cfg.ngram_size, cfg.heads_per_ngram, primes, offsets,
                       tuple(layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_index, cfg.seed)),
                       int(cfg.eos_token_id), cfg.vocab_size, -(-total // div) * div)


def ngram_rows(tokens: np.ndarray, ctx: np.ndarray, p: NgramParams) -> np.ndarray:
    """The table rows of each token: tokens int [B, T] (or [T]), ctx int [B, 2] (or [2]): the two tokens before the
    first one, (eos, eos) at a sequence start -> int64 [B, T, heads] (or [T, heads]). The next call's context is
    concat(ctx, tokens)[..., -2:]."""
    one = tokens.ndim == 1
    tokens, ctx = np.atleast_2d(np.asarray(tokens, np.int64)), np.atleast_2d(np.asarray(ctx, np.int64))
    if tokens.size and (tokens.min() < 0 or tokens.max() >= p.vocab_size):
        raise ValueError(f"n-gram token ids must be in [0, {p.vocab_size}): map image tokens to the image token id first")
    c = p.ngram_size - 1
    if ctx.shape[-1] != c:
        raise ValueError(f"the n-gram context has {c} tokens, got {ctx.shape[-1]}")
    hist = np.concatenate([ctx, tokens], axis=1)
    T = tokens.shape[1]
    mixed = tokens * p.multipliers[0]
    blocks, broken = [], np.zeros_like(tokens, dtype=bool)
    for k in range(1, p.ngram_size):  # shift k: the token k back, or eos when an eos lies in the k - 1 tokens between
        prev = hist[:, c - k:c - k + T]
        mixed = mixed ^ (np.where(broken, p.eos, prev) * p.multipliers[k])
        broken = broken | (prev == p.eos)
        h0 = (k - 1) * p.heads_per_ngram
        sizes = np.array(p.sizes[h0:h0 + p.heads_per_ngram], np.int64)
        offs = np.array(p.offsets[h0:h0 + p.heads_per_ngram], np.int64)
        blocks.append(mixed[..., None] % sizes + offs)
    out = np.concatenate(blocks, axis=-1)
    return out[0] if one else out


# ------------------------------------------------------------------------------------------------------------ the table
class NgramTable:
    """The n-gram rows of a safetensors file, read on demand (see the module docstring).

    rows(ids) -> uint8 [n, dim] (the raw e4m3 bytes); dequant(raw, dtype) -> the values. Thread-safe: each call has its
    own buffer."""

    def __init__(self, path: str, prefix: str = NGRAM_PREFIX, threads: int = 16):
        h, base = header(path)
        names = {}
        for k, v in h.items():
            if k.startswith(prefix + ".shard_") and k.endswith(".weight"):
                names[int(k[len(prefix) + 7:-7])] = v
        if not names or sorted(names) != list(range(len(names))):
            raise ValueError(f"{path}: no contiguous set of shards {prefix}.shard_<i>.weight")
        shapes = {tuple(v["shape"]) for v in names.values()}
        dtypes = {v["dtype"] for v in names.values()}
        if len(shapes) != 1 or dtypes != {"F8_E4M3"}:
            raise ValueError(f"{path}: the n-gram shards must share one shape and be F8_E4M3, got {shapes} {dtypes}")
        (self.rows_per_shard, self.dim), = shapes
        self.n_shards = len(names)
        self.path, self.prefix = path, prefix
        self.shard_offsets = np.array([base + names[i]["data_offsets"][0] for i in range(self.n_shards)], np.int64)
        self.file_size = os.path.getsize(path)
        sc = h.get(prefix + ".weight_scale")
        if sc is None:
            raise ValueError(f"{path}: no {prefix}.weight_scale")
        with open(path, "rb") as f:
            f.seek(base + sc["data_offsets"][0])
            raw = f.read(sc["data_offsets"][1] - sc["data_offsets"][0])
        self.scale_tensor = torch.frombuffer(bytearray(raw), dtype=DTYPES[sc["dtype"]]).reshape(sc["shape"]).clone()
        if self.scale_tensor.numel() != 1:
            raise ValueError(f"{path}: the n-gram scale has {self.scale_tensor.numel()} values, expected 1")
        self.scale = float(self.scale_tensor.float())
        self.fd, self.direct = _open(path)
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="ngram-rows")
        self.threads = threads
        self._lock = threading.Lock()

    @property
    def n_rows(self) -> int:
        return self.n_shards * self.rows_per_shard

    def offsets(self, r: np.ndarray) -> np.ndarray:
        """The file offset of each row r (int64)."""
        r = np.asarray(r, np.int64)
        if r.size and (r.min() < 0 or r.max() >= self.n_rows):
            raise IndexError(f"n-gram rows must be in [0, {self.n_rows})")
        return self.shard_offsets[r // self.rows_per_shard] + (r % self.rows_per_shard) * self.dim

    def rows(self, ids, out: np.ndarray | None = None) -> np.ndarray:
        """ids int [n] (any shape; flattened) -> uint8 [n, dim]: the raw bytes of each row, in the order of ids.
        Repeated ids are read once. The reads cover whole pages, sorted by offset, and a run of rows that share
        pages is one read."""
        ids = np.asarray(ids, np.int64).reshape(-1)
        if out is None:
            out = np.empty((ids.size, self.dim), np.uint8)
        if ids.size == 0:
            return out
        uniq, inv = np.unique(ids, return_inverse=True)
        offs = self.offsets(uniq)
        order = np.argsort(offs, kind="stable")
        so = offs[order]
        first, last = so // PAGE, (so + self.dim - 1) // PAGE  # the pages of each row
        new = np.ones(so.size, bool)
        new[1:] = first[1:] > np.maximum.accumulate(last)[:-1]  # a row whose first page is past the run's last page starts a run
        run_id = np.cumsum(new) - 1
        starts = first[new]
        ends = np.maximum.reduceat(last, np.flatnonzero(new)) + 1  # one past the run's last page
        npages = ends - starts
        buf_page = np.concatenate([[0], np.cumsum(npages)[:-1]])  # each run's first page in the buffer
        raw = np.empty((int(npages.sum()) + 1) * PAGE, np.uint8)
        a = (-raw.ctypes.data) % PAGE
        buf = raw[a:a + int(npages.sum()) * PAGE]
        mv = memoryview(buf)

        def read(lo: int, hi: int):
            for j in range(lo, hi):
                pos, want = int(starts[j]) * PAGE, int(npages[j]) * PAGE
                dst, got = int(buf_page[j]) * PAGE, 0
                need = min(want, self.file_size - pos)  # the last page can end past the file
                while got < need:
                    r = os.preadv(self.fd, [mv[dst + got:dst + want]], pos + got)
                    if r <= 0:
                        raise OSError(f"{self.path}: short read at {pos + got}")
                    got += r
        nr = starts.size
        step = max(1, -(-nr // self.threads))
        for f in [self.pool.submit(read, lo, min(lo + step, nr)) for lo in range(0, nr, step)]:
            f.result()
        at = buf_page[run_id] * PAGE + (so - starts[run_id] * PAGE)  # each sorted row's place in the buffer
        got_rows = buf[at[:, None] + np.arange(self.dim)]
        rows_u = np.empty_like(got_rows)
        rows_u[order] = got_rows
        out[:] = rows_u[inv]
        return out

    def rows_into(self, ids, out: torch.Tensor) -> torch.Tensor:
        """rows(ids) into a uint8 CPU tensor [n, dim] (e.g. pinned, for a non-blocking copy to the device)."""
        if out.dtype != torch.uint8 or out.device.type != "cpu" or tuple(out.shape) != (np.size(ids), self.dim):
            raise ValueError(f"out must be a uint8 CPU tensor [{np.size(ids)}, {self.dim}], got {out.dtype} {tuple(out.shape)} {out.device}")
        self.rows(ids, out.numpy())
        return out

    def dequant(self, raw: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """uint8 [..., dim] -> dtype: (e4m3 * scale) in fp32, rounded once."""
        return (raw.view(torch.float8_e4m3fn).float() * self.scale).to(dtype)

    def close(self):
        if self.fd >= 0:
            self.pool.shutdown(wait=True)
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001  interpreter shutdown
            pass

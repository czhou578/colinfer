#!/usr/bin/env python3
"""ple_hotset.py -- how many rows of the PLE n-gram table (Qwen3.8-Flash-Next) real text touches, and how well a row
cache of a given size would serve them. The engine reads 16 rows of 160 B per token from a 51 GB table on disk
(engine/weights/ple.py); a bounded device row cache would hold the hot set. The tool needs no checkpoint and no GPU.

It tokenizes the cached WikiText-103 and TinyStories datasets with the 27B's tokenizer (the vocabulary is the same;
the tool checks the special ids), hashes every token with the real config (engine.weights.ple.ngram_rows), and
reports per corpus:
- the distinct rows after 1, 2, 5, ... million tokens (bigram and trigram heads apart)
- the hit rate of an LRU row cache of 1e5 .. 3.2e7 rows (16 MB .. 5.1 GB): exact LRU stack distances over a spatial
  sample of the rows (row % rate == 0), with the cache sizes scaled by the same rate (SHARDS sampling)
- the coverage of a static preload: the N most frequent rows of one corpus, on the other corpus

Documents are separated by eos, so each one starts with the context (eos, eos), as a new conversation does.

   uv run python tools/ple_hotset.py                       # 10M tokens of each corpus
   uv run python tools/ple_hotset.py --tokens 2000000 --rate 64
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from engine.model.qwen4exp import Qwen4ExpConfig
from engine.weights.ple import ngram_params, ngram_rows

HUB = os.path.expanduser("~/.cache/huggingface/hub")
CORPORA = {
    "wikitext": f"{HUB}/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/train-*.parquet",
    "tinystories": f"{HUB}/datasets--eminorhan--tinystories/snapshots/*/100M_1/train-*.parquet",
}
TOKENIZER = "nvidia/Qwen3.8-27B-NVFP4"
CACHE_ROWS = [100_000, 200_000, 400_000, 800_000, 1_600_000, 3_200_000, 6_400_000, 12_800_000, 32_000_000]
ROW_BYTES = 160


def documents(corpus: str):
    """The documents of a cached corpus: WikiText articles (split at the top-level " = Title = " lines) or stories."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(CORPORA[corpus]))
    if not files:
        raise FileNotFoundError(f"{corpus}: no cached files at {CORPORA[corpus]}")
    for f in files:
        col = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        if corpus != "wikitext":
            yield from (t for t in col if t.strip())
            continue
        doc: list[str] = []
        for line in col:
            if line.startswith(" = ") and not line.startswith(" = = ") and doc:
                yield "".join(doc)
                doc = []
            doc.append(line)
        if doc:
            yield "".join(doc)


def load_tokenizer(cfg: Qwen4ExpConfig, name: str = TOKENIZER):
    from tokenizers import Tokenizer

    from engine.weights.loader import resolve
    tok = Tokenizer.from_file(os.path.join(resolve(name), "tokenizer.json"))
    ids = {"<|endoftext|>": cfg.eos_token_id, "<|image_pad|>": cfg.image_token_id}
    for t, want in ids.items():
        if tok.token_to_id(t) != want:
            raise ValueError(f"{name}: {t} is {tok.token_to_id(t)}, the model's is {want}")
    if tok.get_vocab_size() > cfg.vocab_size:
        raise ValueError(f"{name}: {tok.get_vocab_size()} tokens > the model's {cfg.vocab_size}")
    return tok


def token_stream(corpus: str, tok, eos: int, max_tokens: int, batch: int = 256) -> np.ndarray:
    """The corpus's tokens, documents separated by eos, up to max_tokens."""
    out, n, docs = [], 0, []
    for d in documents(corpus):
        docs.append(d)
        if len(docs) < batch:
            continue
        for e in tok.encode_batch(docs, add_special_tokens=False):
            out.append(np.array(e.ids + [eos], np.int64))
            n += len(e.ids) + 1
        docs = []
        if n >= max_tokens:
            break
    else:
        for e in tok.encode_batch(docs, add_special_tokens=False):
            out.append(np.array(e.ids + [eos], np.int64))
    return np.concatenate(out)[:max_tokens]


def lru_hit_rates(accesses: np.ndarray, sizes: list[int]) -> list[float]:
    """The hit rate of an LRU cache of each size over the access sequence: the share of accesses whose stack distance
    (the distinct items since the item's last access) is below the size. Fenwick tree over the access times."""
    n = len(accesses)
    tree = [0] * (n + 1)
    last: dict[int, int] = {}
    dist = np.full(n, np.iinfo(np.int64).max, np.int64)

    def add(i: int, v: int):
        i += 1
        while i <= n:
            tree[i] += v
            i += i & -i

    def prefix(i: int) -> int:  # the markers at times [0, i]
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s
    for t, r in enumerate(accesses.tolist()):
        p = last.get(r)
        if p is not None:
            dist[t] = prefix(t - 1) - prefix(p)
            add(p, -1)
        add(t, 1)
        last[r] = t
    return [float((dist < s).mean()) if n else 0.0 for s in sizes]


def report(name: str, rows: np.ndarray, rate: int, log=print) -> np.ndarray:
    """rows int64 [T, 16]: prints the growth of the distinct rows and the LRU curve; returns the row counts."""
    T = rows.shape[0]
    log(f"\n== {name}: {T / 1e6:.1f}M tokens, {rows.size / 1e6:.0f}M row reads")
    for m in (1, 2, 5, 10, 20, 50, 100):
        if m * 1_000_000 > T:
            break
        r = rows[:m * 1_000_000]
        log(f"  after {m:>3}M tokens: {np.unique(r).size / 1e6:7.2f}M distinct rows "
            f"(bigram {np.unique(r[:, :8]).size / 1e6:.2f}M, trigram {np.unique(r[:, 8:]).size / 1e6:.2f}M), "
            f"{np.unique(r).size * ROW_BYTES / 1e9:.2f} GB")
    flat = rows.reshape(-1)
    sample = flat[flat % rate == 0]
    t0 = time.time()
    hits = lru_hit_rates(sample, [max(1, s // rate) for s in CACHE_ROWS])
    log(f"  LRU row cache (1/{rate} spatial sample, {sample.size / 1e6:.2f}M accesses, {time.time() - t0:.0f}s):")
    for s, h in zip(CACHE_ROWS, hits):
        log(f"    {s / 1e6:6.1f}M rows ({s * ROW_BYTES / 1e9:5.2f} GB): hit rate {h:6.1%}")
    u, c = np.unique(flat, return_counts=True)
    return u[np.argsort(-c, kind="stable")]


def coverage(hot: np.ndarray, rows: np.ndarray, name: str, log=print):
    flat = rows.reshape(-1)
    for s in CACHE_ROWS:
        log(f"    top {s / 1e6:6.1f}M rows of the other corpus on {name}: {np.isin(flat, hot[:s]).mean():6.1%} of reads")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", type=int, default=10_000_000, help="tokens per corpus")
    ap.add_argument("--rate", type=int, default=128, help="the spatial sample of the LRU simulation: rows with row %% rate == 0")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    a = ap.parse_args()
    cfg = Qwen4ExpConfig()
    p = ngram_params(cfg, 0)
    tok = load_tokenizer(cfg, a.tokenizer)
    hot, rows = {}, {}
    for c in CORPORA:
        t0 = time.time()
        ids = token_stream(c, tok, cfg.eos_token_id, a.tokens)
        t1 = time.time()
        rows[c] = ngram_rows(ids, np.array([p.eos, p.eos]), p)
        print(f"[hotset] {c}: {ids.size / 1e6:.1f}M tokens, tokenized in {t1 - t0:.0f}s, hashed in {time.time() - t1:.1f}s")
        hot[c] = report(c, rows[c], a.rate)
    names = list(CORPORA)
    print("\n== static preload, cross-corpus")
    for i, c in enumerate(names):
        coverage(hot[names[1 - i]], rows[c], c)


if __name__ == "__main__":
    main()

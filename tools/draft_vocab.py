#!/usr/bin/env python3
"""Build the frequency-ranked draft vocabulary for the MTP drafter (FR-Spec-style truncated lm_head).

The tool counts the token frequencies over a local corpus mix (WikiText prose, Python source, JSON, and chat-formatted
text), and forces the special / control tokens in. It writes the top ids (most frequent first) to
engine/spec/draft_vocab.npy (uint32). The drafter uses the first N of them.
  uv run python tools/draft_vocab.py [--top 65536]
"""
import argparse
import glob
import os
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=65536)
    ap.add_argument("--code-mb", type=float, default=24)
    ap.add_argument("--prose-mb", type=float, default=60, help="WikiText-103 train text")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from engine.weights.loader import resolve
    from tests.perplexity import wikitext_test
    tok = AutoTokenizer.from_pretrained(resolve("nvidia/Qwen3.8-27B-NVFP4"))
    V = len(tok)
    counts = np.zeros(V + 1024, dtype=np.int64)

    buf = []

    def add(text):
        buf.append(text)
        if sum(len(b) for b in buf) < 2_000_000:
            return 0
        return flush()

    def flush():
        if not buf:
            return 0
        ids = tok("\n".join(buf), add_special_tokens=False).input_ids
        buf.clear()
        np.add.at(counts, ids, 1)
        return len(ids)

    n = add(wikitext_test())
    import pyarrow.parquet as pq
    val = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*.parquet")))
    if val:
        n += add("\n\n".join(pq.read_table(val[0]).column("text").to_pylist()))
    train = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/train-*.parquet")))
    used = 0
    for f in train:
        for t in pq.read_table(f).column("text").to_pylist():
            if t.strip():
                n += add(t)
                used += len(t)
            if used > a.prose_mb * 1e6:
                break
        if used > a.prose_mb * 1e6:
            break
    rng = random.Random(0)
    site = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "lib")
    for pattern, budget in (("**/*.py", a.code_mb * 1e6), ("**/*.json", a.code_mb * 1e6 / 4)):
        files = glob.glob(os.path.join(site, pattern), recursive=True)
        rng.shuffle(files)
        used = 0
        for f in files:
            try:
                t = open(f, encoding="utf-8").read()
            except Exception:
                continue
            if len(t) > 200_000:
                continue
            n += add(t)
            used += len(t)
            if used > budget:
                break
    n += flush()
    chat = tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! How can I help?"}],
                                   tokenize=True)
    special = set(tok.all_special_ids) | set(chat if isinstance(chat, list) else chat["input_ids"])
    special |= {i for i, t in enumerate(tok.convert_ids_to_tokens(range(V))) if t and t.startswith("<") and t.endswith(">") and len(t) < 32}
    order = np.argsort(-counts[:V], kind="stable")
    top = [i for i in special if i < V] + [int(i) for i in order if int(i) not in special]
    top = np.array(top[: a.top], dtype=np.uint32)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "spec", "draft_vocab.npy")
    np.save(out, top)
    cov = {k: counts[top[:k]].sum() / counts.sum() for k in (8192, 16384, 32768, 65536) if k <= a.top}
    print(f"corpus {n} tokens; wrote {len(top)} ids to {out}; corpus coverage: " + ", ".join(f"top {k}: {v:.4f}" for k, v in cov.items()))


if __name__ == "__main__":
    main()

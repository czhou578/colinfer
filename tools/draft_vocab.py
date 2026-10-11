#!/usr/bin/env python3
"""Build the frequency-ranked draft vocabulary for the MTP drafter (FR-Spec-style truncated lm_head).

The tool counts the token frequencies over a local corpus mix, and forces the special / control tokens in. It writes
the top ids (most frequent first) to engine/spec/draft_vocab.npy (uint32). The drafter uses the first N of them.

The corpus is pinned, so a regeneration gives the same file: WikiText-103 prose from the HF cache, the Python sources
of the transformers package (its version is in uv.lock), the JSON files of the checkpoint (its configs and weight index,
not the tokenizer vocabularies), and a chat-formatted exchange.
  uv run python tools/draft_vocab.py [--top 65536] [--out engine/spec/draft_vocab.npy]
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
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "spec", "draft_vocab.npy"))
    a = ap.parse_args()
    import transformers

    from engine.server.chat import ChatFormat
    from engine.weights.loader import MODEL, resolve
    from tests.perplexity import wikitext_files, wikitext_lines, wikitext_test
    fmt = ChatFormat.from_checkpoint()
    tok = fmt.tok
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
    n += add("\n\n".join(wikitext_lines(wikitext_files("validation")[0])))
    used = 0
    for f in wikitext_files("train"):
        for t in wikitext_lines(f):
            if t.strip():
                n += add(t)
                used += len(t)
            if used > a.prose_mb * 1e6:
                break
        if used > a.prose_mb * 1e6:
            break
    rng = random.Random(0)
    code = sorted(glob.glob(os.path.join(os.path.dirname(transformers.__file__), "**", "*.py"), recursive=True))
    ckpt = resolve(MODEL)
    jsons = sorted(f for f in glob.glob(os.path.join(ckpt, "*.json")) if os.path.basename(f) not in ("tokenizer.json", "vocab.json"))
    for files, budget in ((code, a.code_mb * 1e6), (jsons, a.code_mb * 1e6 / 4)):
        rng.shuffle(files)
        used = 0
        for f in files:
            try:
                t = open(f, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):  # unreadable, or not text
                continue
            if len(t) > 200_000:
                continue
            n += add(t)
            used += len(t)
            if used > budget:
                break
    n += flush()
    chat = fmt.render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! How can I help?"}])  # the template's tokens
    special = set(tok.all_special_ids) | set(chat)
    special |= {i for i, t in enumerate(tok.convert_ids_to_tokens(range(V))) if t and t.startswith("<") and t.endswith(">") and len(t) < 32}
    order = np.argsort(-counts[:V], kind="stable")
    top = [i for i in special if i < V] + [int(i) for i in order if int(i) not in special]
    top = np.array(top[: a.top], dtype=np.uint32)
    out = a.out
    np.save(out, top)
    cov = {k: counts[top[:k]].sum() / counts.sum() for k in (8192, 16384, 32768, 65536) if k <= a.top}
    print(f"corpus {n} tokens; wrote {len(top)} ids to {out}; corpus coverage: " + ", ".join(f"top {k}: {v:.4f}" for k, v in cov.items()))


if __name__ == "__main__":
    main()

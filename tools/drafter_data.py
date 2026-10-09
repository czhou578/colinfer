#!/usr/bin/env python3
"""Training data for the drafter (PLAN.md Phase 6: better drafters): the own replies of the target model to varied
prompts, generated through a running server (engine/server) and saved as token ids.

The tool builds the prompts offline from local data:
- WikiText-103 train (article titles and paragraphs)
- TinyStories (story openings)
- Python sources of the installed packages (functions to explain / document / refactor)
- templates for questions and structured output

Mix: ~45% prose, 15% Q&A, 25% code, 15% structured, with thinking on for half. The server samples the replies (T=0.7,
top-p 0.95, top-k 20), so that they vary. The tool does not use the evaluation prompts (tests/scheduler_check.py).

   uv run python -m engine.server --port 8002 &
   uv run python tools/drafter_data.py --url http://127.0.0.1:8002 --n 1500 --out ~/.cache/colinfer/drafter/data.jsonl
"""
import argparse
import ast
import concurrent.futures as cf
import glob
import json
import os
import random
import threading
import time

import requests

TOPICS = ["volcanoes", "the history of tea", "a lighthouse in a storm", "learning to play the violin", "migrating birds",
          "a city on Mars", "the printing press", "friendship between rivals", "a robot gardener", "ancient Rome's water system",
          "deep-sea creatures", "a forgotten library", "the first winter in a new country", "coral reefs", "bread baking",
          "a detective on a train", "quantum computing for beginners", "the Silk Road", "a family recipe", "glaciers",
          "the immune system", "a chess prodigy", "urban beekeeping", "the moon landing", "a small-town election"]
QUESTIONS = ["What are the main causes of {}?", "How would you explain {} to a ten-year-old?", "What are common misconceptions about {}?",
             "Give me practical advice about {}.", "Compare the pros and cons of {}.", "Why does {} matter today?",
             "What should a beginner know about {}?", "Describe a typical day involving {}."]
PROSE = ["Write a short story about {}.", "Write a reflective essay about {}.", "Describe {} in vivid detail.",
         "Write a letter to a friend about {}.", "Write a blog post introducing {}.", "Write a poem in free verse about {}."]
STRUCT = ["Return a JSON object describing {} with fields name, summary, key_facts (list) and difficulty (1-5).",
          "Make a markdown table of five facts about {} with columns Fact and Why it matters.",
          "List ten bullet points about {}, each under fifteen words.",
          "Produce YAML with a title, three sections and two subsections each, outlining an article on {}."]
CODE = ["Explain what this Python function does, step by step:\n\n```python\n{}\n```",
        "Add type hints and a docstring to this function. Return only the code.\n\n```python\n{}\n```",
        "Refactor this function for readability without changing behavior. Return only the code.\n\n```python\n{}\n```",
        "Write pytest tests for this function:\n\n```python\n{}\n```",
        "Find possible bugs or edge cases in this code:\n\n```python\n{}\n```"]
CODE_GEN = ["Write a Python function that {}. Include a docstring.", "Write a Python module that {}, with tests.",
            "Write a small command-line tool in Python that {}."]
CODE_TASKS = ["parses a CSV file and prints column averages", "merges overlapping intervals", "implements a trie with insert and search",
              "computes the edit distance between two strings", "rate-limits function calls with a token bucket",
              "walks a directory tree and reports the largest files", "validates an email address with a regex",
              "implements Dijkstra's shortest path", "caches function results to disk", "serializes a binary tree to a string"]


def wiki_material(rng):
    import pyarrow.parquet as pq
    wt = os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1")
    files = sorted(glob.glob(wt + "/train-*.parquet"))
    lines = pq.read_table(files[0]).column("text").to_pylist()
    titles = [l.strip(" =\n") for l in lines if l.startswith(" = ") and not l.startswith(" = = ")]
    paras = [l.strip() for l in lines if len(l) > 600 and not l.startswith(" =")]
    rng.shuffle(titles)
    rng.shuffle(paras)
    return titles[:5000], paras[:5000]


def story_starts(rng):
    files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--eminorhan--tinystories/snapshots/*/**/*"), recursive=True))
    out = []
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            t = pq.read_table(f)
            col = t.column(t.column_names[0]).to_pylist()
            out += [s for s in col[:20000] if isinstance(s, str) and len(s) > 200]
        elif f.endswith((".json", ".jsonl", ".txt")) and os.path.getsize(f) < 2e9:
            with open(f, errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) > 200:
                        out.append(line)
                    if len(out) > 20000:
                        break
        if len(out) > 20000:
            break
    rng.shuffle(out)
    return [" ".join(s.split()[:40]) for s in out[:3000]]


def code_snippets(rng):
    import transformers
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(transformers.__file__)), "*", "**", "*.py"), recursive=True))
    rng.shuffle(files)
    out = []
    for f in files[:3000]:
        try:
            src = open(f, errors="replace").read()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.end_lineno and 8 <= node.end_lineno - node.lineno <= 40:
                out.append("\n".join(src.splitlines()[node.lineno - 1:node.end_lineno]))
        if len(out) > 3000:
            break
    rng.shuffle(out)
    return out


def build_prompts(n, rng):
    titles, paras = wiki_material(rng)
    stories = story_starts(rng)
    snippets = code_snippets(rng)
    prompts = []
    for i in range(n):
        r = rng.random()
        if r < 0.45:
            c = rng.random()
            if c < 0.3:
                p = rng.choice(PROSE).format(rng.choice(TOPICS + titles[:200]))
            elif c < 0.55:
                p = f"Explain {rng.choice(titles)} in a few paragraphs."
            elif c < 0.8:
                p = f"Summarize this passage and then discuss its main idea:\n\n{rng.choice(paras)}"
            else:
                p = f"Continue this story for a few paragraphs:\n\n{rng.choice(stories)}" if stories else rng.choice(PROSE).format(rng.choice(TOPICS))
            kind = "prose"
        elif r < 0.60:
            p, kind = rng.choice(QUESTIONS).format(rng.choice(TOPICS + titles[:500])), "qa"
        elif r < 0.85:
            p = rng.choice(CODE).format(rng.choice(snippets)) if rng.random() < 0.7 else rng.choice(CODE_GEN).format(rng.choice(CODE_TASKS))
            kind = "code"
        else:
            p, kind = rng.choice(STRUCT).format(rng.choice(TOPICS + titles[:500])), "struct"
        prompts.append((p, kind, rng.random() < 0.5))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/colinfer/drafter/data.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from engine.weights.loader import resolve
    tok = AutoTokenizer.from_pretrained(resolve("nvidia/Qwen3.8-27B-NVFP4"))
    rng = random.Random(a.seed)
    prompts = build_prompts(a.n, rng)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):  # resume
        for line in open(a.out):
            done.add(json.loads(line)["i"])
    lock, t0, ntok = threading.Lock(), time.time(), [0]
    fh = open(a.out, "a")

    def one(i):
        if i in done:
            return
        p, kind, think = prompts[i]
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        r = requests.post(a.url + "/v1/chat/completions", json={
            "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": a.seed * 100000 + i,
            "chat_template_kwargs": {"enable_thinking": think}, "return_token_ids": True}, timeout=3600).json()
        out = r["choices"][0]["token_ids"]
        assert r["usage"]["prompt_tokens"] == len(ids), "prompt rendering differs from the server's"
        with lock:
            fh.write(json.dumps({"i": i, "kind": kind, "think": think, "prompt": ids, "output": out}) + "\n")
            fh.flush()
            ntok[0] += len(out)
            if i % 25 == 0:
                print(f"[data] {i}/{a.n}: {ntok[0]} tokens, {ntok[0] / (time.time() - t0):.0f} tok/s", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, range(a.n)))
    print(f"[data] done: {a.out}")


if __name__ == "__main__":
    main()

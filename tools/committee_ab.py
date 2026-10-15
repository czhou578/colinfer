#!/usr/bin/env python3
"""A blind A/B of committee mode (docs/server.md, "Committee mode") against a single answer, on a running server.

For each prompt, the tool asks the server twice: the model as named, and <model>-committee. It writes the two answers of
each prompt in a random order, as A and B, to pairs.md, and which letter was the committee to key.json, so that you
judge the pairs before you know. results.json keeps the raw replies, the members' answers and the times.

   uv run python tools/committee_ab.py                            # the built-in prompts: design and review questions
   uv run python tools/committee_ab.py --prompts questions.md     # your own prompts, separated by blank lines
   uv run python tools/committee_ab.py --reveal ~/.cache/colinfer/committee_ab/<run>   # the key, after you judged
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

OUT = os.path.expanduser("~/.cache/colinfer/committee_ab")
PROMPTS = [
    "Review this Python function for bugs and edge cases, and give a corrected version:\n\n```python\n"
    "def merge_ranges(ranges):\n    ranges.sort()\n    out = [ranges[0]]\n    for lo, hi in ranges[1:]:\n"
    "        if lo <= out[-1][1]:\n            out[-1] = (out[-1][0], hi)\n        else:\n            out.append((lo, hi))\n    return out\n```",
    "A single-user LLM server on one machine runs HTTP handling and the GPU work in one process today. Should the HTTP layer move "
    "to a second process? Give a recommendation with the trade-offs.",
    "We want to add a disk cache for the KV state of past conversations, keyed by a hash chain of the token ids. What can go wrong "
    "with this design, and what would you check before shipping it?",
    "Write a short code review of this shell snippet, which is meant to delete build artifacts older than a week:\n\n"
    "```bash\nfind . -name '*.o' -mtime +7 | xargs rm\n```",
    "A test is flaky: it fails about one run in twenty with a timeout. Give a plan to find the cause, in order of what to try first.",
    "Explain whether a speculative decoder can ever change the output of a greedy model, and under what conditions the output is "
    "guaranteed to be identical.",
]


def parse_prompts(text: str) -> list[str]:
    """Prompts separated by blank lines; lines that start with # are comments."""
    out, cur = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("#"):
            continue
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            out.append("\n".join(cur).strip())
            cur = []
    return out


def ask(url: str, model: str, prompt: str, max_tokens: int, thinking: str, timeout: float = 3600.0) -> dict:
    """One chat completion without a stream: {text, seconds, reply}."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False}
    if thinking != "default":
        body["enable_thinking"] = thinking == "on"
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.load(r)
    return {"text": reply["choices"][0]["message"].get("content") or "", "seconds": time.time() - t0, "reply": reply}


def make_pairs(results: list[dict], seed: int = 0) -> list[dict]:
    """Each prompt's two answers as A and B in a random order; `committee` names the letter of the committee's."""
    rng = random.Random(seed)
    pairs = []
    for r in results:
        letter = rng.choice("AB")
        other = "B" if letter == "A" else "A"
        pairs.append({"prompt": r["prompt"], letter: r["committee"]["text"], other: r["single"]["text"], "committee": letter})
    return pairs


def render_pairs(pairs: list[dict]) -> str:
    """pairs.md: the prompts and their two answers, nothing that tells them apart."""
    L = ["# A/B pairs", "", "For each question, note which answer you prefer (A or B, or neither) before you look at key.json.", ""]
    for i, p in enumerate(pairs, 1):
        L += [f"## {i}. {p['prompt'].splitlines()[0][:80]}", "", p["prompt"], "", "### A", "", p["A"].strip(), "", "### B", "", p["B"].strip(), ""]
    return "\n".join(L)


def summary(results: list[dict]) -> str:
    s = sum(r["single"]["seconds"] for r in results) / len(results)
    c = sum(r["committee"]["seconds"] for r in results) / len(results)
    return f"single {s:.1f} s, committee {c:.1f} s per answer ({c / s:.1f}x)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="a blind A/B of committee mode against a single answer")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="the model name (default: the first of /v1/models); the committee is <model>-committee")
    ap.add_argument("--prompts", default=None, help="a text file of prompts separated by blank lines (default: the built-in questions)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default", help="enable_thinking of every request")
    ap.add_argument("--seed", type=int, default=0, help="the order of A and B")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--reveal", metavar="RUN_DIR", help="print the key of a finished run")
    a = ap.parse_args(argv)
    if a.reveal:
        with open(os.path.join(a.reveal, "key.json")) as f:
            key = json.load(f)
        for i, k in enumerate(key, 1):
            print(f"{i}. committee = {k['committee']}   {k['prompt'].splitlines()[0][:70]}")
        return 0
    prompts = PROMPTS
    if a.prompts:
        with open(a.prompts) as f:
            prompts = parse_prompts(f.read())
        if not prompts:
            print(f"{a.prompts}: no prompts", file=sys.stderr)
            return 2
    model = a.model
    if model is None:
        with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=10) as r:
            model = json.load(r)["data"][0]["id"]
    run = os.path.join(os.path.expanduser(a.out), time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(run)
    results = []
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {p.splitlines()[0][:70]}", flush=True)
        single = ask(a.url, model, p, a.max_tokens, a.thinking)
        print(f"  single    {single['seconds']:6.1f} s, {len(single['text'])} chars", flush=True)
        comm = ask(a.url, model + "-committee", p, a.max_tokens, a.thinking)
        print(f"  committee {comm['seconds']:6.1f} s, {len(comm['text'])} chars", flush=True)
        results.append({"prompt": p, "single": single, "committee": comm})
        with open(os.path.join(run, "results.json"), "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
    pairs = make_pairs(results, a.seed)
    with open(os.path.join(run, "pairs.md"), "w") as f:
        f.write(render_pairs(pairs))
    with open(os.path.join(run, "key.json"), "w") as f:
        json.dump([{"prompt": p["prompt"], "committee": p["committee"]} for p in pairs], f, indent=1, ensure_ascii=False)
    print(f"\n{summary(results)}\njudge {run}/pairs.md, then: tools/committee_ab.py --reveal {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

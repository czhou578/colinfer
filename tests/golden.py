#!/usr/bin/env python3
"""Golden-output check: the engine in its server configuration (INT6 / INT5 decode copies and the drafter files when
present, MTP speculation with suffix-match drafts, 3 slots) on a fixed request set must reproduce recorded tokens and
per-token logprobs exactly.

Requests (all submitted at once, so they also run batched): the first 16 prompts of the 40-prompt mix greedy for up to
128 tokens; 4 of them sampled (temperature 0.8, top-p 0.95, top-k 20, fixed seeds) for 64 tokens; a 20,000-token WikiText
prompt greedy for 32 tokens (FP8 prefill attention past 16k, the drafter's long prefill, decode at depth). Each output
token's logprob is compared bit for bit: it depends on all 248k logits, so any numeric change shows up even when the
tokens do not change.

The prompts are token ids frozen in tests/golden/prompts.json (also read by bench/perf.py and
bench/request_mix_bench.py): tools/drafter_data.py builds its code prompts from the Python files installed in
site-packages, so regenerating them in a different environment gives different prompts.

   uv run python tests/golden.py record          # writes tests/golden/outputs.json
   uv run python tests/golden.py check             # MTP speculation with suffix-match drafts (the default server)
   uv run python tests/golden.py check --suffix 0  # MTP drafts only: must match the same file
   uv run python tests/golden.py check --plain     # no speculation: the same file
   uv run python tests/golden.py prompts         # regenerate prompts.json (environment-dependent, see above)
"""
import argparse
import json
import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from engine.spec.suffix import MIN_MATCH  # noqa: E402

FILE = os.path.join(ROOT, "tests", "golden", "outputs.json")
PROMPTS = os.path.join(ROOT, "tests", "golden", "prompts.json")
EOS = (248046, 248044)


def build_engine(spec: bool, slots: int = 3, max_seq_len: int = 32768, checkpoints: int = 32, suffix_min: int = MIN_MATCH):
    """The server's engine (engine/server/api.py Worker._build) without HTTP: model, decode copies, drafter, scheduler."""
    from engine.model.fast import attach_decode_copies, decode_copies_paths, load_fast_model, to_fast
    from engine.runtime.scheduler import Scheduler
    from engine.weights.loader import resolve
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    model = to_fast(load_fast_model(path, verbose=False))
    files = decode_copies_paths(path)
    if all(os.path.exists(f) for f in files):
        attach_decode_copies(model, files)
    mtp = None
    if spec:
        from engine.spec.mtp import DRAFT_DIR, Mtp
        ft = os.path.join(DRAFT_DIR, "mtp_ft.safetensors")
        mtp = Mtp(model, path, weights=ft if os.path.exists(ft) else None)
    sched = Scheduler(model, n_slots=slots, max_seq_len=max_seq_len, n_checkpoints=checkpoints, mtp=mtp, k=7, selftest=False,
                      suffix_min=suffix_min)
    return path, sched


def freeze_prompts(path):
    """The 40-prompt mix (tools/drafter_data.py, seed 1, chat template, last 3,000 tokens) and a 20,000-token WikiText
    prompt, as token ids."""
    from transformers import AutoTokenizer

    from drafter_data import build_prompts
    from tests.perplexity import wikitext_test
    tok = AutoTokenizer.from_pretrained(path)
    mix = []
    for p, kind, think in build_prompts(40, random.Random(1)):
        x = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=think, tokenize=True)
        mix.append(dict(kind=kind, think=think, ids=list(x["input_ids"] if hasattr(x, "keys") else x)[-3000:]))
    long = tok(wikitext_test()[:400000], add_special_tokens=False).input_ids[:20000]
    json.dump(dict(mix=mix, long=long), open(PROMPTS, "w"), separators=(",", ":"))


def load_prompts():
    """{"mix": [{"kind", "think", "ids"}] x 40, "long": [ids]} (tests/golden/prompts.json)."""
    return json.load(open(PROMPTS))


def requests():
    from engine.runtime.scheduler import Request
    P = load_prompts()
    reqs = []
    for i, m in enumerate(P["mix"][:16]):
        x, kind = m["ids"], m["kind"]
        reqs.append((f"greedy-{i}-{kind}", Request(x, max_new_tokens=128, eos_ids=EOS, logprobs=0)))
        if i % 4 == 0:
            reqs.append((f"sampled-{i}-{kind}", Request(x, max_new_tokens=64, temperature=0.8, top_p=0.95, top_k=20, seed=1000 + i, eos_ids=EOS,
                                                        logprobs=0)))
    reqs.append(("long-20k", Request(P["long"], max_new_tokens=32, eos_ids=EOS, logprobs=0)))
    return reqs


def run(spec: bool, suffix_min: int = MIN_MATCH):
    _, sched = build_engine(spec, suffix_min=suffix_min)
    reqs = requests()
    t0 = time.perf_counter()
    sched.run([r for _, r in reqs])
    print(f"[golden] {len(reqs)} requests in {time.perf_counter() - t0:.1f} s ({'MTP' if spec else 'plain decode'}"
          + (f", suffix drafts >= {suffix_min}" if suffix_min else "") + ")")
    return {name: {"tokens": r.output, "logprobs": [lp for lp, _ in r.output_logprobs]} for name, r in reqs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("record", "check", "prompts"))
    ap.add_argument("--plain", action="store_true", help="run without speculation")
    ap.add_argument("--suffix", type=int, default=MIN_MATCH, help="suffix-match drafts of at least this match length (0: off)")
    a = ap.parse_args()
    if a.mode == "prompts":
        from engine.weights.loader import resolve
        freeze_prompts(resolve("nvidia/Qwen3.8-27B-NVFP4"))
        print(f"[golden] wrote {PROMPTS}")
        return
    with torch.inference_mode():
        got = run(spec=not a.plain, suffix_min=a.suffix)
    if a.mode == "record":
        os.makedirs(os.path.dirname(FILE), exist_ok=True)
        json.dump(got, open(FILE, "w"), separators=(",", ":"))
        print(f"[golden] recorded {sum(len(v['tokens']) for v in got.values())} tokens -> {FILE}")
        return
    want = json.load(open(FILE))
    bad = 0
    for name, w in want.items():
        g = got[name]
        if g["tokens"] != w["tokens"]:
            n = next((i for i, (x, y) in enumerate(zip(g["tokens"], w["tokens"])) if x != y), min(len(g["tokens"]), len(w["tokens"])))
            print(f"  {name}: tokens differ from position {n}")
            bad += 1
        elif g["logprobs"] != w["logprobs"]:
            d = max(abs(x - y) for x, y in zip(g["logprobs"], w["logprobs"]))
            print(f"  {name}: same tokens, logprobs differ (max {d:.2e})")
            bad += 1
    print(f"[golden] {len(want) - bad}/{len(want)} requests exact" + ("; GOLDEN CHECK PASSED" if not bad else "; GOLDEN CHECK FAILED"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

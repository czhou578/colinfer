#!/usr/bin/env python3
"""Speculative decoding check (PLAN.md Phase 4 exit: greedy outputs identical with and without spec; with the MTP
drafter, seeded sampled outputs are identical too).

Runs each prompt through plain greedy decode (CUDA graph, T=1) and through the speculative generator,
asserts identical token sequences, and reports tok/s and acceptance.
  uv run python tests/spec_check.py [--drafter ngram] [--k 3] [--max-new 256]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from engine.model.fast import load_fast_model, to_fast  # noqa: E402
from engine.runtime.fast_generate import FastGenerator  # noqa: E402
from engine.weights.loader import resolve  # noqa: E402

EOS = (248046, 248044)
CODE = '''def parse_config(path):
    with open(path) as f:
        data = json.load(f)
    config = Config()
    config.name = data["name"]
    config.version = data["version"]
    config.author = data["author"]
    config.license = data["license"]
    config.description = data["description"]
    return config
'''
PROMPTS = {
    "code-edit": "Rewrite this function so every field access uses data.get(key, default) with a sensible default, "
                 "and add type hints. Return only the code.\n\n" + CODE,
    "json": "Convert this list to a JSON array of objects with keys name, role and team: Alice engineer platform; "
            "Bob designer web; Carol manager platform; Dave engineer infra; Erin analyst data; Frank engineer web.",
    "code-gen": "Write a Python class LRUCache with get and put methods, O(1) each, using an OrderedDict. Include docstrings.",
    "prose": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
}


def chat(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="ngram", choices=["ngram", "mtp"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--mtp-bf16", action="store_true", help="draft with the BF16 MTP weights (default: FP8)")
    ap.add_argument("--draft-vocab", type=int, default=65536, help="0 = full vocabulary")
    ap.add_argument("--temperature", type=float, default=0.0, help="> 0: sampled (no identity check; speed and acceptance only)")
    a = ap.parse_args()
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    model = to_fast(load_fast_model(path), kv_fp8=True)
    if a.drafter == "ngram":
        from engine.spec.runner import SpecGenerator
        spec = SpecGenerator(model, max_seq_len=8192, k=a.k)
    else:
        from engine.spec.mtp import MtpGenerator
        spec = MtpGenerator(model, path, max_seq_len=8192, k=a.k, fp8=not a.mtp_bf16, draft_vocab=a.draft_vocab or None)
    all_same, tot_tok, tot_t = True, 0, 0.0
    for name, text in PROMPTS.items():
        ids = chat(tok, text)
        # baseline: the same generator (same prefill, same decode graph) with drafting off
        kw = dict(temperature=a.temperature, top_p=0.95, top_k=20, seed=1) if a.temperature > 0 else {}
        t0 = time.perf_counter(); ref = spec.generate(ids, a.max_new, eos_ids=EOS, use_spec=False, **kw); t_plain = time.perf_counter() - t0
        st0 = dict(spec.stats)
        t0 = time.perf_counter(); out = spec.generate(ids, a.max_new, eos_ids=EOS, **kw); t_spec = time.perf_counter() - t0
        d = {k: spec.stats[k] - st0[k] for k in spec.stats}
        exact = a.temperature == 0 or a.drafter == "mtp"  # MTP sampling is position-keyed: identical to plain sampling too
        same = out == ref if exact else True
        all_same &= same
        tot_tok += len(out); tot_t += t_spec
        acc = d["accepted"] / max(1, d["drafted"])
        tag = ("IDENTICAL" if same else "DIFFERENT") if exact else "sampled  "
        print(f"{name:10s} {tag}  {len(out):4d} tok  plain {len(ref) / t_plain:5.1f} tok/s  "
              f"spec {len(out) / t_spec:5.1f} tok/s  ({d['spec_steps']}/{d['steps']} steps drafted, acceptance {acc:.2f}, "
              f"{len(out) / max(1, d['steps']):.2f} tok/step)", flush=True)
        if a.temperature > 0:
            print("   ", repr(tok.decode(out[:30])))
        if not same:
            i = next(i for i in range(min(len(out), len(ref))) if out[i] != ref[i])
            print(f"   first difference at token {i}: {tok.decode(ref[max(0, i - 8):i + 4])!r} vs {tok.decode(out[max(0, i - 8):i + 4])!r}")
    print(f"overall spec {tot_tok / tot_t:.1f} tok/s;", "SPEC CHECK PASSED" if all_same else "SPEC CHECK FAILED")


if __name__ == "__main__":
    main()

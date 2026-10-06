#!/usr/bin/env python3
"""Speculative decoding check: outputs with MTP speculation must be identical to plain decoding (greedy, and seeded
sampling with --temperature), on the default decode weights (INT6 / INT5 copies when present).

Runs each prompt through plain decode (the CUDA graph, T = 1) and through the speculative cycle, asserts identical
token sequences, and reports tok/s and acceptance.
  uv run python tests/spec_check.py [--k 7] [--max-new 256] [--temperature 0.8]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from engine.model.fast import load_fast_model, to_fast  # noqa: E402
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
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--drafter-weights", default=None, help="fine-tuned MTP weights (tools/train_drafter.py)")
    ap.add_argument("--temperature", type=float, default=0.0, help="> 0: seeded sampling (top-p 0.95, top-k 20, seed 1)")
    a = ap.parse_args()
    from engine.model.fast import attach_decode_copies, decode_copies_paths
    from engine.spec.mtp import MtpGenerator
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    model = to_fast(load_fast_model(path))
    files = [f for f in decode_copies_paths(path) if os.path.exists(f)]
    print(f"INT6 / INT5 decode copies: {attach_decode_copies(model, files)} linears")
    spec = MtpGenerator(model, path, max_seq_len=8192, k=a.k, weights=a.drafter_weights)
    all_same, tot_tok, tot_t = True, 0, 0.0
    for name, text in PROMPTS.items():
        ids = chat(tok, text)
        # baseline: the same generator (same prefill, same decode graph) with drafting off
        kw = dict(temperature=a.temperature, top_p=0.95, top_k=20, seed=1) if a.temperature > 0 else {}
        t0 = time.perf_counter(); ref = spec.generate(ids, a.max_new, eos_ids=EOS, use_spec=False, **kw); t_plain = time.perf_counter() - t0
        st0 = dict(spec.stats)
        t0 = time.perf_counter(); out = spec.generate(ids, a.max_new, eos_ids=EOS, **kw); t_spec = time.perf_counter() - t0
        d = {k: spec.stats[k] - st0[k] for k in spec.stats}
        same = out == ref  # sampling is position-keyed, so sampled outputs must match too
        all_same &= same
        tot_tok += len(out); tot_t += t_spec
        acc = d["accepted"] / max(1, d["drafted"])
        print(f"{name:10s} {'IDENTICAL' if same else 'DIFFERENT'}  {len(out):4d} tok  plain {len(ref) / t_plain:5.1f} tok/s  "
              f"spec {len(out) / t_spec:5.1f} tok/s  ({d['steps']} cycles, acceptance {acc:.2f}, "
              f"{len(out) / max(1, d['steps']):.2f} tok/cycle)", flush=True)
        if a.temperature > 0:
            print("   ", repr(tok.decode(out[:30])))
        if not same:
            i = next(i for i in range(min(len(out), len(ref))) if out[i] != ref[i])
            print(f"   first difference at token {i}: {tok.decode(ref[max(0, i - 8):i + 4])!r} vs {tok.decode(out[max(0, i - 8):i + 4])!r}")
    print(f"overall spec {tot_tok / tot_t:.1f} tok/s;", "SPEC CHECK PASSED" if all_same else "SPEC CHECK FAILED")


if __name__ == "__main__":
    main()

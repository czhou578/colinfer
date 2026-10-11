#!/usr/bin/env python3
"""Drafter acceptance on fresh prompts (the mix of tools/drafter_data.py with another seed), greedy, through the real
speculative cycle of the engine (engine/spec/mtp.py MtpGenerator, fixed k). It reports the share of accepted drafts and
the tokens per cycle, per kind. It compares the MTP head of the checkpoint with fine-tuned ones
(tools/train_drafter.py). It measures no time, so it can share the GPU.

   uv run python tools/eval_drafter.py [--weights a.safetensors b.safetensors] [--n 40] [--k 3 7]
"""
import argparse
import collections
import os
import random

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--no-baseline", action="store_true", help="skip the checkpoint's MTP head")
    ap.add_argument("--full-head", action="store_true", help="score drafts with the full draft head, not the low-rank one")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from engine.runtime.build import load_model
    from engine.spec.mtp import MtpGenerator
    from tools.drafter_data import build_prompts
    path, model = load_model()  # the served configuration: the INT6 / INT5 decode copies when present
    tok = AutoTokenizer.from_pretrained(path)
    prompts = build_prompts(a.n, random.Random(1))
    ids = []
    for p, kind, think in prompts:
        x = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=think, tokenize=True)
        ids.append((list(x["input_ids"] if hasattr(x, "keys") else x)[-3000:], kind))
    eos = (248046, 248044)
    for w in ([] if a.no_baseline else [None]) + a.weights:
        for k in a.k:
            gen = MtpGenerator(model, path, max_seq_len=4096, k=k, weights=w, **({"lowrank": None} if a.full_head else {}))
            per = collections.defaultdict(lambda: [0, 0, 0, 0])  # drafted, accepted, tokens, cycles
            for x, kind in ids:
                s0 = dict(gen.stats)
                out = gen.generate(x, a.max_new, eos)
                d = per[kind]
                d[0] += gen.stats["drafted"] - s0["drafted"]
                d[1] += gen.stats["accepted"] - s0["accepted"]
                d[2] += len(out)
                d[3] += gen.stats["steps"] - s0["steps"]
            tot = [sum(v[i] for v in per.values()) for i in range(4)]
            name = os.path.basename(w) if w else "checkpoint MTP"
            kinds = "  ".join(f"{kd}: acc {v[1] / max(v[0], 1):.3f} tok/cycle {v[2] / max(v[3], 1):.2f}" for kd, v in sorted(per.items()))
            print(f"[{name} k={k}] " + kinds + f"  | all: acc {tot[1] / max(tot[0], 1):.3f} tok/cycle {tot[2] / max(tot[3], 1):.2f}", flush=True)
            del gen
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

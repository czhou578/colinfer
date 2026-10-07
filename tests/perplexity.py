#!/usr/bin/env python3
"""Perplexity on the WikiText test set (PLAN.md 4.7), per weight configuration.

It uses the cached Salesforce/wikitext wikitext-103-raw-v1 test split (identical to the test set of WikiText-2). The
script joins the text with "\\n\\n" as in the HF perplexity guide, and tokenizes it with the tokenizer of the
checkpoint. It scores the text in non-overlapping windows of --ctx tokens, each from a fresh state. It scores each
token except the first of each window.

  uv run python tests/perplexity.py --ckpt Qwen/Qwen3.8-27B                # BF16
  uv run python tests/perplexity.py --ckpt Qwen/Qwen3.8-27B-FP8            # FP8 dequantized to BF16
  uv run python tests/perplexity.py --ckpt nvidia/Qwen3.8-27B-NVFP4        # NVFP4/FP8 dequantized to BF16
Options: --ctx 2048 --max-tokens 65536 --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wikitext_test() -> str:
    import pyarrow.parquet as pq
    pat = os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/test-*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"WikiText test parquet not found: {pat}")
    return "\n\n".join(pq.read_table(files[0]).column("text").to_pylist())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--json")
    ap.add_argument("--override-filter", default=None, help="regex: only override linears whose name matches")
    ap.add_argument("--text", help="score this corpus instead of WikiText: a text file, or 'code' (Python standard library sources)")
    ap.add_argument("--emulate", help="quant emulation effects: act_nvfp4,act_fp8,fp8_requant or all")
    ap.add_argument("--engine", choices=["reference", "prefill"], default="reference",
                    help="reference: Phase 1 PyTorch model; prefill: Phase 3 W4A4 / W8A8 kernel prefill path")
    ap.add_argument("--override", help="reference engine: safetensors replacing the checkpoint's weights, comma-separated: "
                                       "tools/int6_requant.py --simulate outputs (.weight_deq) or NVFP4 tensors")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from engine.weights.loader import load_model, resolve
    path = resolve(args.ckpt)
    tok = AutoTokenizer.from_pretrained(path)
    if args.text == "code":
        import sysconfig
        lib = sysconfig.get_paths()["stdlib"]
        text = "\n\n".join(open(os.path.join(lib, f), errors="replace").read() for f in sorted(os.listdir(lib)) if f.endswith(".py"))
    elif args.text:
        text = open(args.text, errors="replace").read()
    else:
        text = wikitext_test()
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_total = ids.numel()
    n_win = min(args.max_tokens, n_total) // args.ctx
    print(f"[ppl] {args.ckpt}: {args.text or 'WikiText test'} {n_total} tokens; scoring {n_win} windows x {args.ctx}")

    if args.engine == "prefill":
        from engine.model.fast import load_fast_model, to_fast
        from engine.model.prefill import prefill
        model = to_fast(load_fast_model(path))
    else:
        model = load_model(path, emulate=args.emulate)
        if args.override:
            from safetensors import safe_open

            from engine.weights.loader import PREFIX, dequant_nvfp4
            n = 0
            for ov in args.override.split(","):  # several files: e.g. INT6 attention + INT5 GDN
              with safe_open(ov, "pt", device="cuda") as f:
                import re
                for k in f.keys():
                    if k.endswith(".weight_deq") and (not args.override_filter or re.search(args.override_filter, k)):
                        # dequantized weights of another format (tools/int6_requant.py --simulate)
                        model.get_submodule(k[len(PREFIX):-len(".weight_deq")]).weight.data.copy_(f.get_tensor(k))
                        n += 1
                        continue
                    if k.endswith(".weight") and (not args.override_filter or re.search(args.override_filter, k)):
                        base = k[: -len(".weight")]
                        w = dequant_nvfp4(f.get_tensor(k), f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_scale_2"))
                        model.get_submodule(base[len(PREFIX):]).weight.data.copy_(w)
                        n += 1
            print(f"[ppl] override: {n} linears from {ov}")
    nll, count = 0.0, 0
    t0 = time.time()
    for w in range(n_win):
        x = ids[w * args.ctx:(w + 1) * args.ctx].view(1, -1).cuda()
        state = model.new_state(1, args.ctx)
        if args.engine == "prefill":
            logits = prefill(model, x, state, all_logits=True)[:-1].float()
        else:
            logits = model(x, state)[0, :-1]
        loss = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum")
        nll += loss.item()
        count += x.shape[1] - 1
        del logits
        if (w + 1) % 8 == 0 or w == n_win - 1:
            print(f"[ppl] {w + 1}/{n_win} windows  running ppl {math.exp(nll / count):.4f}  ({time.time() - t0:.0f}s)")
    ppl = math.exp(nll / count)
    print(f"[ppl] RESULT text={args.text or 'wikitext'} ckpt={args.ckpt} engine={args.engine} emulate={args.emulate} override={args.override} filter={args.override_filter} "
          f"ctx={args.ctx} tokens={count} ppl={ppl:.4f} nll={nll / count:.5f}")
    if args.json:
        json.dump(dict(ckpt=args.ckpt, emulate=args.emulate, ctx=args.ctx, tokens=count, ppl=ppl, nll=nll / count), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

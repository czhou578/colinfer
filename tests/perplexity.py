#!/usr/bin/env python3
"""Perplexity on the WikiText test set (PLAN.md 4.7), per weight configuration.

Uses the cached Salesforce/wikitext wikitext-103-raw-v1 test split (identical to WikiText-2's test
set). Text is joined with "\\n\\n" as in the HF perplexity guide, tokenized with the checkpoint's
tokenizer, and scored in non-overlapping windows of --ctx tokens, each from a fresh state. Every
token but the first of each window is scored.

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
    ap.add_argument("--emulate", help="quant emulation effects: act_nvfp4,act_fp8,fp8_requant or all")
    ap.add_argument("--engine", choices=["reference", "prefill"], default="reference",
                    help="reference: Phase 1 PyTorch model; prefill: Phase 3 W4A4 / W8A8 kernel prefill path")
    ap.add_argument("--kv-fp8", action="store_true", help="prefill engine: fp8 KV cache")
    ap.add_argument("--override", help="reference engine: safetensors of NVFP4 weights replacing the checkpoint's "
                                       "(tools/requant_nvfp4.py output; 'requant' = its default path)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from engine.weights.loader import load_model, resolve
    path = resolve(args.ckpt)
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(wikitext_test(), return_tensors="pt").input_ids[0]
    n_total = ids.numel()
    n_win = min(args.max_tokens, n_total) // args.ctx
    print(f"[ppl] {args.ckpt}: WikiText test {n_total} tokens; scoring {n_win} windows x {args.ctx}")

    if args.engine == "prefill":
        from engine.model.fast import load_fast_model, to_fast
        from engine.model.prefill import prefill
        model = to_fast(load_fast_model(path), kv_fp8=args.kv_fp8)
    else:
        model = load_model(path, emulate=args.emulate)
        if args.override:
            from safetensors import safe_open

            from engine.weights.loader import PREFIX, dequant_nvfp4
            ov = args.override
            if ov == "requant":
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
                from requant_nvfp4 import default_out
                ov = default_out(path)
            n = 0
            with safe_open(ov, "pt", device="cuda") as f:
                for k in f.keys():
                    if k.endswith(".weight"):
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
    print(f"[ppl] RESULT ckpt={args.ckpt} engine={args.engine} emulate={args.emulate} override={args.override} kv_fp8={args.kv_fp8} "
          f"ctx={args.ctx} tokens={count} ppl={ppl:.4f} nll={nll / count:.5f}")
    if args.json:
        json.dump(dict(ckpt=args.ckpt, emulate=args.emulate, ctx=args.ctx, tokens=count, ppl=ppl, nll=nll / count), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

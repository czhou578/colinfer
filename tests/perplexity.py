#!/usr/bin/env python3
"""Perplexity on the WikiText test set (docs/history/PLAN.md 4.7), per weight configuration.

It uses the cached Salesforce/wikitext wikitext-103-raw-v1 test split (identical to the test set of WikiText-2). The
script joins the text with "\\n\\n" as in the HF perplexity guide, and tokenizes it with the tokenizer of the
checkpoint. It scores the text in non-overlapping windows of --ctx tokens, each from a fresh state. It scores each
token except the first of each window.

  uv run python tests/perplexity.py --ckpt Qwen/Qwen3.8-27B                # BF16
  uv run python tests/perplexity.py --ckpt nvidia/Qwen3.8-27B-NVFP4        # NVFP4/FP8 dequantized to BF16
Options: --ctx 2048 --max-tokens 65536 --json out.json

The qwen4_exp family (Qwen3.8-Flash-Next) runs its prefill (engine/model/prefill_qwen4exp.py) with --engine qwen4exp: one
call per --chunk tokens (the scheduler's way, with the two tokens before each chunk as prev2), the logits of every token
in bf16, an n-gram feeder with a row cache. --ctx takes several values (one model load). For each window it also prints the
NLL of each 2048-token segment, to compare with the 2048-token windows of tests/data/qwen4exp_golden.json (each from a
fresh state). It loads about 80 GB: run it under the GPU lock.

  flock -w 14400 ~/.cache/colinfer/gpu-model.lock uv run python tests/perplexity.py --engine qwen4exp --ckpt nvidia/Qwen3.8-Flash-Next-NVFP4 \
      --ctx 16384 2048 --max-tokens 16384
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

WIKITEXT = "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1"  # the HF cache of Salesforce/wikitext


def wikitext_files(split: str) -> list[str]:
    """The parquet shards of a split (test, validation or train) of WikiText-103 in the local HF cache, sorted."""
    files = sorted(glob.glob(os.path.expanduser(WIKITEXT) + f"/{split}-*.parquet"))
    if not files:
        sys.exit(f"WikiText {split} parquet not found under {WIKITEXT}")
    return files


def wikitext_lines(file: str) -> list[str]:
    """The text rows of one shard."""
    import pyarrow.parquet as pq
    return pq.read_table(file).column("text").to_pylist()


def wikitext_test() -> str:
    return "\n\n".join(wikitext_lines(wikitext_files("test")[0]))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048], help="the window length (several: one run per value)")
    ap.add_argument("--chunk", type=int, default=4096, help="--engine qwen4exp: the tokens per prefill call")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--json")
    ap.add_argument("--override-filter", default=None, help="regex: only override linears whose name matches")
    ap.add_argument("--text", help="score this corpus instead of WikiText: a text file, or 'code' (Python standard library sources)")
    ap.add_argument("--engine", choices=["reference", "prefill", "qwen4exp"], default="reference",
                    help="reference: Phase 1 PyTorch model; prefill: Phase 3 W4A4 / W8A8 kernel prefill path; qwen4exp: the prefill of "
                         "the qwen4_exp family")
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
    if args.engine == "qwen4exp":
        return qwen4exp(args, path, ids)
    args.ctx = args.ctx[0]
    n_total = ids.numel()
    n_win = min(args.max_tokens, n_total) // args.ctx
    print(f"[ppl] {args.ckpt}: {args.text or 'WikiText test'} {n_total} tokens; scoring {n_win} windows x {args.ctx}")

    if args.engine == "prefill":
        from engine.model.fast import load_fast_model, to_fast
        from engine.model.prefill import prefill
        model = to_fast(load_fast_model(path))
    else:
        model = load_model(path)
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
    print(f"[ppl] RESULT text={args.text or 'wikitext'} ckpt={args.ckpt} engine={args.engine} override={args.override} filter={args.override_filter} "
          f"ctx={args.ctx} tokens={count} ppl={ppl:.4f} nll={nll / count:.5f}")
    if args.json:
        json.dump(dict(ckpt=args.ckpt, ctx=args.ctx, tokens=count, ppl=ppl, nll=nll / count), open(args.json, "w"), indent=1)


def qwen4exp(args, path: str, ids: torch.Tensor):
    """--engine qwen4exp: the windows of each --ctx through the family's prefill, --chunk tokens per call."""
    from engine.model import prefill_qwen4exp as pq
    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from engine.weights.ple import PleFeeder
    _, model = load_fast_qwen4exp(path)
    ple = model.ple_layer()
    model.feeder = PleFeeder(ple.table, cache_rows=6_250_000, params=ple.params, image_token_id=model.cfg.image_token_id)
    eos = model.cfg.eos_token_id
    gold = None
    gfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qwen4exp_golden.json")
    if os.path.exists(gfile) and not args.text:
        gold = json.load(open(gfile)).get("wikitext")
    results = []
    for ctx in args.ctx:
        n_win = min(args.max_tokens, ids.numel()) // ctx
        print(f"[ppl] {args.ckpt}: {ids.numel()} tokens; scoring {n_win} windows x {ctx}, {args.chunk}-token prefill calls")
        state = model.new_state(1, -(-ctx // 16) * 16)
        nll, count, t0, t_pf = 0.0, 0, time.time(), 0.0
        for w in range(n_win):
            x = ids[w * ctx:(w + 1) * ctx].tolist()
            state.reset()
            tok_nll = []
            for c0 in range(0, ctx, args.chunk):
                c = torch.tensor([x[c0:c0 + args.chunk]], device="cuda")
                torch.cuda.synchronize()
                t1 = time.time()
                logits = pq.prefill(model, c, state, chunk=args.chunk, all_logits=True, prev2=([eos, eos] + x[:c0])[-2:])
                torch.cuda.synchronize()
                t_pf += time.time() - t1
                tgt = torch.tensor(x[c0 + 1:c0 + args.chunk + 1], device="cuda")  # the next token of each row (none for the window's last)
                for a in range(0, tgt.numel(), 1024):
                    lg = logits[a:min(a + 1024, tgt.numel())].float()
                    tok_nll.append(torch.nn.functional.cross_entropy(lg, tgt[a:a + 1024], reduction="none").double())
                del logits
            t = torch.cat(tok_nll)  # [ctx - 1]: the NLL of tokens 1 .. ctx - 1
            nll += float(t.sum())
            count += t.numel()
            seg = [float(t[max(0, s - 1):s + 2047].mean()) for s in range(0, ctx, 2048)]  # tokens s .. s + 2047 (the first token: none)
            ref = ", ".join(f"{math.exp(v):.3f}" for v in gold["window_nll"]) if gold and w == 0 and ctx > 2048 else None
            print(f"[ppl] window {w}: ppl {math.exp(float(t.mean())):.4f}; per 2048-token segment " + ", ".join(f"{math.exp(v):.3f}" for v in seg) +
                  (f" (the reference's 2048-token windows, each from a fresh state: {ref})" if ref else ""), flush=True)
        ppl = math.exp(nll / count)
        h, m, _, _ = model.feeder.stats()
        print(f"[ppl] RESULT engine=qwen4exp ctx={ctx} chunk={args.chunk} tokens={count} ppl={ppl:.4f} nll={nll / count:.5f} "
              f"(prefill {count / t_pf:.0f} tok/s with the logits; {time.time() - t0:.0f} s; feeder hit rate {h / max(1, h + m):.1%})" +
              (f"; the reference's {gold['windows']} windows of {gold['tokens']}: {gold['ppl']:.4f}, ratio {ppl / gold['ppl']:.4f}" if gold else ""),
              flush=True)
        results.append(dict(ckpt=args.ckpt, ctx=ctx, chunk=args.chunk, tokens=count, ppl=ppl, nll=nll / count))
        del state
        torch.cuda.empty_cache()
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

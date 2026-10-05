#!/usr/bin/env python3
"""Offline estimate of two draft chains (top-1 and top-2 first draft) vs one, replaying speculative cycles along
held-out replies with the drafter's teacher-forced unroll (identical to chained drafting on accepted prefixes).
The one-chain numbers reproduce tools/eval_drafter.py (2.76 / 3.41 vs 2.76 / 3.39 tokens per cycle). docs section 12.

   uv run python tools/twochain_sim.py
"""
import collections
import glob
import os
import random
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_drafter as td  # noqa: E402

torch.set_grad_enabled(False)
cfg, path, embed, lm_draft, remap = td.setup()
t = td.mtp_tensors(path)
with safe_open(os.path.expanduser("~/.cache/colinfer/drafter/mtp_ft.safetensors"), "pt", device="cuda") as f:
    for n in f.keys(): t[n] = f.get_tensor(n).to(torch.bfloat16)
head = td.Head(t, cfg).cuda().eval()
first = sorted(glob.glob(os.path.join(td.DIR, "feat", "shard_*.pt")))
refs = [(f, i) for f in first for i in range(len(torch.load(f, mmap=True)))]
random.Random(0).shuffle(refs)
val = sorted(refs[:max(20, len(refs) // 20)])
K = 7

def simulate(hit, hit2, p, end, kk, two):
    tok = cyc = 0
    while p + kk + 1 < end:
        nA = 0
        while nA < kk and hit[nA, p + nA]: nA += 1
        n = nA
        if two and not hit[0, p] and hit2[p]:
            nB = 1
            while nB < kk and hit[nB, p + nB]: nB += 1
            n = max(n, nB)
        tok += n + 1; cyc += 1; p += n + 1
    return tok, cyc

agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
cache = {}
for f, i in val:
    if f not in cache: cache = {f: torch.load(f, mmap=True)}
    it = cache[f][i]
    ids = it["ids"].long().cuda(); T, P = ids.numel(), it["P"]
    e = embed[torch.cat([ids[1:], ids[-1:]])]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outs = head.unroll(e, it["H"].cuda(), K)
    nxt = remap[it["ti"].cuda()[:, 0].long()].roll(-1)   # row r's drafts predict x_{r+2}: the target argmax at ti[r + 1]
    preds = [(g @ lm_draft.t()).float() for g in outs]
    hit = torch.stack([p.argmax(-1) == nxt for p in preds]).cpu()
    hit2 = (preds[0].topk(2, -1).indices[:, 1] == nxt).cpu()
    for kk in (3, 7):
        for two in (False, True):
            tk, cy = simulate(hit, hit2, max(P - 2, 0), T - 2, kk, two)
            a = agg[it["kind"]][(kk, two)]; a[0] += tk; a[1] += cy
            a = agg["all"][(kk, two)]; a[0] += tk; a[1] += cy
print(f"{'kind':8s} {'k=3: 1 chain':>12s} {'2 chains':>9s} {'gain':>6s} | {'k=7: 1 chain':>12s} {'2 chains':>9s} {'gain':>6s}")
for kind, d in sorted(agg.items()):
    r = {key: v[0] / max(v[1], 1) for key, v in d.items()}
    print(f"{kind:8s} {r[(3, False)]:12.2f} {r[(3, True)]:9.2f} {r[(3, True)] / r[(3, False)] - 1:+6.1%} | "
          f"{r[(7, False)]:12.2f} {r[(7, True)]:9.2f} {r[(7, True)] / r[(7, False)] - 1:+6.1%}")

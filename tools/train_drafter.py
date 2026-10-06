#!/usr/bin/env python3
"""Fine-tune the checkpoint's MTP head as a multi-step drafter (docs/history/phase6_progress.md sections 7, 12, 13).

The MTP head ships trained for one step: (embedding of x_{i+1}, target hidden h_i) -> x_{i+2}. The engine chains it
(engine/spec/mtp.py): step s feeds the head its own previous output instead of a target hidden state, so drafts 2..k
see inputs the head never trained on, and acceptance falls with depth (prose: ~0.45 per token). This tool fine-tunes it
the way EAGLE-3 trains its drafter ("training-time test"): every training row is unrolled to depth D exactly as
drafting runs, on the target model's own replies (tools/drafter_data.py).

Depth s, row i (the draft for x_{i+2} made s-1 chain steps after a catch-up row at p = i - s + 1):
    input   token x_{i+1}, hidden = target h_i (s = 1) or the head's own normed output of depth s-1 at row i-1
    attends depth-1 rows at positions <= p (the cache of catch-up rows) and the depth-t row at position p + t - 1
            for t = 2..s (the chain's own earlier steps), exactly the MTP KV cache at inference
    target  the target model's next-token distribution at position i+1 (top-32, restricted to the 64k draft vocabulary
            the engine drafts from), soft cross-entropy
Rows are scored only where x_{i+2} lies in the reply. The embedding and lm_head stay frozen (shared with the target).

   uv run python tools/train_drafter.py --extract      # target hidden states + top-k distributions (once per --data file)
   uv run python tools/train_drafter.py --train        # fine-tune; writes ~/.cache/colinfer/drafter/mtp_ft.safetensors
                                                       # (prints held-out per-depth agreement with the target before / after)
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.weights.loader import resolve  # noqa: E402

DIR = os.path.expanduser("~/.cache/colinfer/drafter")
TOPK = 32
MAXLEN = 2048


def mtp_tensors(path):
    from safetensors import safe_open
    wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    t = {}
    for name, f in wm.items():
        if name.startswith("mtp."):
            with safe_open(os.path.join(path, f), framework="pt", device="cuda") as sf:
                t[name] = sf.get_tensor(name).to(torch.bfloat16)
    return t


# ------------------------------------------------------------------------------------------------ 1. features
@torch.inference_mode()
def extract(a):
    from engine.model.fast import load_fast_model, to_fast
    from engine.model.prefill import prefill
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    model = to_fast(load_fast_model(path))
    src = os.path.expanduser(a.data)
    rows = [json.loads(l) for l in open(src)]
    stem = os.path.splitext(os.path.basename(src))[0]
    prefix = "" if stem == "data" else stem + "_"  # data.jsonl -> shard_*.pt, data2.jsonl -> data2_shard_*.pt
    feat = os.path.join(DIR, a.feat)
    os.makedirs(feat, exist_ok=True)
    t0 = time.time()
    shard, n = [], 0
    for j, r in enumerate(rows):
        ids = (r["prompt"] + r["output"])[-MAXLEN:]
        P = max(0, len(ids) - len(r["output"]))  # reply starts here
        st = model.new_state(1, len(ids))
        logits, H = prefill(model, torch.tensor([ids], device="cuda"), st, all_logits=True, return_hidden=True)
        lp = torch.log_softmax(logits.float(), -1)
        tv, ti = lp.topk(TOPK, -1)
        item = dict(ids=torch.tensor(ids, dtype=torch.int32), P=P, H=H.cpu(), tv=tv.half().cpu(), ti=ti.int().cpu(), kind=r["kind"])
        shard.append(item)
        if len(shard) == 100 or j == len(rows) - 1:
            torch.save(shard, os.path.join(feat, f"{prefix}shard_{n:03d}.pt"))
            shard, n = [], n + 1
            print(f"[extract] {j + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ 2. the head
class Head(torch.nn.Module):
    """The MTP head in fp32 with a training-time-test unroll (one decoder layer, as the engine runs it)."""
    def __init__(self, t, cfg):
        super().__init__()
        P = lambda n: torch.nn.Parameter(t["mtp." + n].float())  # noqa: E731
        self.fc = P("fc.weight")
        self.pre_e, self.pre_h, self.norm = P("pre_fc_norm_embedding.weight"), P("pre_fc_norm_hidden.weight"), P("norm.weight")
        L = "layers.0."
        self.in_ln, self.post_ln = P(L + "input_layernorm.weight"), P(L + "post_attention_layernorm.weight")
        self.q, self.k, self.v, self.o = (P(L + f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = P(L + "self_attn.q_norm.weight"), P(L + "self_attn.k_norm.weight")
        self.gate, self.up, self.down = (P(L + f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.eps, self.H, self.Hq, self.Hkv, self.D = cfg.rms_norm_eps, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        d = cfg.rotary_dim
        self.register_buffer("inv_freq", 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)), persistent=False)

    def rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + w)).to(x.dtype)

    def rope(self, x, pos):  # x [H, T, D], partial rotary on the first R dims
        R = self.inv_freq.numel() * 2
        ang = pos[:, None].float() * self.inv_freq[None]
        cos, sin = torch.cat([ang.cos()] * 2, -1).to(x.dtype), torch.cat([ang.sin()] * 2, -1).to(x.dtype)
        xr, xp = x[..., :R], x[..., R:]
        rot = torch.cat([-xr[..., R // 2:], xr[..., :R // 2]], -1)
        return torch.cat([xr * cos + rot * sin, xp], -1)

    def _layer(self, x, kv, s, pos, T, G, scale):
        """The decoder layer at draft depth s; kv: (keys, values) lists, one entry per depth so far."""
        prm = dict(in_ln=self.in_ln, post_ln=self.post_ln, q=self.q, k=self.k, v=self.v, o=self.o, qn=self.qn, kn=self.kn,
                   gate=self.gate, up=self.up, down=self.down)
        ks, vs = kv
        h1 = self.rms(x, prm["in_ln"])
        qg = (h1 @ prm["q"].t().to(h1.dtype)).view(T, self.Hq, 2 * self.D)
        q, gate = qg[..., :self.D], qg[..., self.D:].reshape(T, -1)
        q = self.rope(self.rms(q, prm["qn"]).transpose(0, 1), pos)                                # [Hq, T, D]
        k = self.rope(self.rms((h1 @ prm["k"].t().to(h1.dtype)).view(T, self.Hkv, self.D), prm["kn"]).transpose(0, 1), pos)
        v = (h1 @ prm["v"].t().to(h1.dtype)).view(T, self.Hkv, self.D).transpose(0, 1)
        ks.append(k.repeat_interleave(G, 0))
        vs.append(v.repeat_interleave(G, 0))
        # depth-1 keys at positions <= i - s + 1, plus the chain's own rows: depth t at position i - s + t
        s1 = (q @ ks[0].transpose(1, 2)).float() * scale                                       # [Hq, T, T]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-(s - 1))
        s1 = s1.masked_fill(~mask, float("-inf"))
        diag = []
        for t in range(2, s + 1):
            sh = s - t  # key row i - sh
            kt = torch.cat([torch.zeros_like(ks[t - 1][:, :sh]), ks[t - 1][:, :T - sh]], 1) if sh else ks[t - 1]
            dsc = (q * kt).sum(-1).float() * scale                                              # [Hq, T]
            if sh:
                dsc[:, :sh] = float("-inf")
            diag.append(dsc)
        allsc = torch.cat([s1] + [d[..., None] for d in diag], -1)
        p = torch.softmax(allsc, -1)
        p = torch.nan_to_num(p)  # rows with no visible key (i < s - 1): never scored
        att = p[..., :T].to(v.dtype) @ vs[0]
        for idx, t in enumerate(range(2, s + 1)):
            sh = s - t
            vt = torch.cat([torch.zeros_like(vs[t - 1][:, :sh]), vs[t - 1][:, :T - sh]], 1) if sh else vs[t - 1]
            att = att + p[..., T + idx:T + idx + 1].to(v.dtype) * vt
        att = att.transpose(0, 1).reshape(T, -1) * torch.sigmoid(gate)
        x = x + att @ prm["o"].t().to(att.dtype)
        m = self.rms(x, prm["post_ln"])
        x = x + (F.silu(m @ prm["gate"].t().to(m.dtype)) * (m @ prm["up"].t().to(m.dtype))) @ prm["down"].t().to(m.dtype)
        return x

    def unroll(self, e, h0, depth):
        """e [T, H]: embeddings of x_{i+1}; h0 [T, H]: target hidden h_i. Returns the normed outputs per depth [T, H]."""
        T = e.shape[0]
        pos = torch.arange(T, device=e.device)
        outs = []
        kv = ([], [])  # keys / values per depth
        hid = h0
        scale = self.D ** -0.5
        G = self.Hq // self.Hkv
        for s in range(1, depth + 1):
            if s > 1:  # row i takes the previous depth's output at row i - 1
                hid = torch.cat([torch.zeros_like(outs[-1][:1]), outs[-1][:-1]])
            x = torch.cat([self.rms(e, self.pre_e), self.rms(hid, self.pre_h)], -1) @ self.fc.t().to(e.dtype)
            x = self._layer(x, kv, s, pos, T, G, scale)
            outs.append(self.rms(x, self.norm))
        return outs

    def export(self):
        L = "mtp.layers.0."
        t = {"mtp.fc.weight": self.fc, "mtp.pre_fc_norm_embedding.weight": self.pre_e, "mtp.pre_fc_norm_hidden.weight": self.pre_h,
             "mtp.norm.weight": self.norm, L + "input_layernorm.weight": self.in_ln, L + "post_attention_layernorm.weight": self.post_ln,
             L + "self_attn.q_norm.weight": self.qn, L + "self_attn.k_norm.weight": self.kn}
        for n, p in zip("qkvo", (self.q, self.k, self.v, self.o)):
            t[L + f"self_attn.{n}_proj.weight"] = p
        for n, p in zip(("gate", "up", "down"), (self.gate, self.up, self.down)):
            t[L + f"mlp.{n}_proj.weight"] = p
        return {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in t.items()}


def shards(feat="feat"):
    return sorted(glob.glob(os.path.join(DIR, feat, "*shard_*.pt")))


def setup():
    from engine.model.qwen35 import Qwen35Config
    from engine.weights.loader import dequant_nvfp4
    from safetensors import safe_open
    import numpy as np
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    cfg = Qwen35Config.from_checkpoint(path)
    wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def get(n):
        with safe_open(os.path.join(path, wm[n]), framework="pt", device="cuda") as f:
            return f.get_tensor(n)
    P = "model.language_model."
    embed = get(P + "embed_tokens.weight").to(torch.bfloat16)
    vocab = torch.tensor(np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "spec",
                                              "draft_vocab.npy"))[:65536].astype(np.int64), device="cuda")
    lm = dequant_nvfp4(get("lm_head.weight"), get("lm_head.weight_scale"), get("lm_head.weight_scale_2"), torch.bfloat16)
    lm_draft = lm[vocab].contiguous()
    del lm
    remap = torch.full((cfg.vocab_size,), -1, dtype=torch.long, device="cuda")
    remap[vocab] = torch.arange(vocab.numel(), device="cuda")
    return cfg, path, embed, lm_draft, remap


def seq_loss(head, item, embed, lm_draft, remap, depth, train=True):
    """Soft cross-entropy per depth over reply rows; also top-1 agreement with the target (draft-vocab argmax)."""
    ids = item["ids"].long().cuda()
    T, P = ids.numel(), item["P"]
    H = item["H"].cuda()
    e = embed[torch.cat([ids[1:], ids[-1:]])]  # row i consumes x_{i+1} (the last row is never scored)
    outs = head.unroll(e, H, depth)
    rows = torch.arange(max(P - 2, 0), T - 2, device="cuda")  # x_{i+2} in the reply
    tv, ti = item["tv"].cuda()[rows + 1].float(), item["ti"].cuda()[rows + 1].long()
    idx = remap[ti]
    pt = torch.where(idx >= 0, tv.exp(), torch.zeros_like(tv))
    pt = pt / pt.sum(-1, keepdim=True).clamp_min(1e-9)
    tgt_top1 = idx[:, 0]  # the target's argmax (in the draft vocabulary, else -1: unreachable)
    losses, agree = [], []
    for s, g in enumerate(outs, start=1):
        ok = rows >= s - 1  # a chain needs its catch-up row p = i - s + 1 >= 0
        logits = (g[rows] @ lm_draft.t()).float()
        logp = torch.log_softmax(logits, -1)
        l = -(pt * logp.gather(1, idx.clamp_min(0))).sum(-1)
        losses.append((l * ok).sum() / ok.sum().clamp_min(1))
        agree.append(((logits.argmax(-1) == tgt_top1) & ok).sum().item() / max(ok.sum().item(), 1))
    return losses, agree, rows.numel()


def train(a):
    from safetensors import safe_open
    from safetensors.torch import save_file
    cfg, path, embed, lm_draft, remap = setup()
    t = mtp_tensors(path)
    if a.init:  # start from an earlier fine-tune (e.g. mtp_ft.safetensors)
        with safe_open(os.path.expanduser(a.init), framework="pt", device="cuda") as f:
            for n in f.keys():
                t[n] = f.get_tensor(n).to(torch.bfloat16)
    head = Head(t, cfg).cuda()
    files = shards(a.feat)
    first = [f for f in files if os.path.basename(f).startswith("shard_")]  # data.jsonl: the held-out split comes from here only,
    sizes = {f: len(torch.load(f, mmap=True)) for f in files}               # so it stays the same as more data files are added
    refs = [(f, i) for f in first for i in range(sizes[f])]
    rng = random.Random(0)
    rng.shuffle(refs)  # the same permutation as shuffling the items themselves (it depends only on the length)
    nval = max(20, len(refs) // 20)
    vset = set(refs[:nval])
    val = []
    for f in first:
        items = torch.load(f)
        val += [(f, i, items[i]) for i in range(len(items)) if (f, i) in vset]
    order = {r: k for k, r in enumerate(refs[:nval])}
    val = [it for _, _, it in sorted(val, key=lambda x: order[(x[0], x[1])])]
    ntr = sum(sizes.values()) - nval
    print(f"[train] {ntr} train / {len(val)} held-out replies, depth {a.depth}")
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = a.epochs * ntr // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(st, total) / total)))

    def evaluate():
        head.eval()
        agg = [0.0] * a.depth
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for it in val:
                _, ag, _ = seq_loss(head, it, embed, lm_draft, remap, a.depth, False)
                agg = [x + y for x, y in zip(agg, ag)]
        head.train()
        return [round(x / len(val), 4) for x in agg]

    def stream(rng):
        """Training replies, one shard in memory at a time: shard order and order within a shard shuffled."""
        fs = list(files)
        rng.shuffle(fs)
        for f in fs:
            items = torch.load(f)
            idx = [i for i in range(len(items)) if (f, i) not in vset]
            rng.shuffle(idx)
            for i in idx:
                yield items[i]
            del items
    print(f"[train] held-out top-1 agreement per depth before: {evaluate()}", flush=True)
    t0, step = time.time(), 0
    w = [a.decay ** (s - 1) for s in range(1, a.depth + 1)]
    for ep in range(a.epochs):
        for j, it in enumerate(stream(rng)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, _, _ = seq_loss(head, it, embed, lm_draft, remap, a.depth)
                loss = sum(wi * l for wi, l in zip(w, losses)) / sum(w)
            (loss / a.accum).backward()
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"[train] epoch {ep} step {step}/{total} loss {loss.item():.4f} per depth {[round(l.item(), 3) for l in losses]} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[train] held-out top-1 agreement per depth after epoch {ep}: {evaluate()}", flush=True)
    out = a.out
    save_file(head.export(), out)
    print(f"[train] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(DIR, "mtp_ft.safetensors"))
    ap.add_argument("--data", default=os.path.join(DIR, "data.jsonl"), help="--extract: replies (tools/drafter_data.py output)")
    ap.add_argument("--decay", type=float, default=0.9, help="loss weight decay per depth")
    ap.add_argument("--feat", default="feat", help="feature directory under ~/.cache/colinfer/drafter")
    ap.add_argument("--init", default="", help="--train: start from these mtp.* weights instead of the checkpoint's")
    a = ap.parse_args()
    if a.extract:
        extract(a)
    if a.train:
        train(a)


if __name__ == "__main__":
    main()

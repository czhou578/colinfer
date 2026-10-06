#!/usr/bin/env python3
"""Offline acceptance of the DFlash2 block drafter (z-lab/Qwen3.8-27B-DFlash2) on our NVFP4 target, against the MTP
head, replaying greedy speculative cycles along held-out replies (docs/phase6_progress.md section 16).

DFlash2 (a PyTorch port of SGLang's sglang/srt/models/dflash.py, DFlash2DraftModel): the target's residual stream after
layers 5, 19, 33, 47, 61 -> fc -> RMSNorm is the context; every draft layer's K / V of a context position is its k / v
projection of that feature (context positions never run through the layers). A cycle drafts one block of 8 at positions
p .. p+7: the verified anchor token x_p, then 7 mask tokens (id 248070), embedded with the target's embedding; 5
Qwen3-style layers (non-causal inside the block, 2048-token sliding window; each sublayer wrapped in a dynamic 2-tap
grouped convolution over block positions). Slots 1..7 go through the target's lm_head: top-16 candidates per slot; the
selector scores transitions unary[c] + <A[pred] * P h, B[c]> and walks the argmax path (greedy).

A cycle with anchor p accepts drafts d_1.. while d_j equals the target's argmax for position p + j, then takes the
target's own token: p advances by accepted + 1. The MTP numbers on the same replies come from tools/twochain_sim.py's
one-chain replay (2.76 / 3.41 tokens per cycle at k = 3 / 7 over all kinds).

   uv run python tools/dflash_sim.py [--n 75]          # held-out sampled replies
   uv run python tools/dflash_sim.py --greedy [--both] # the engine's greedy replies to the eval prompts (+ best of both)
"""
import argparse
import collections
import glob
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAPS = (5, 19, 33, 47, 61)


class DFlash2(torch.nn.Module):
    def __init__(self, path, embed, lm_head):
        super().__init__()
        cfg = json.load(open(os.path.join(path, "config.json")))
        dc = cfg["dflash_config"]
        self.B, self.mask_id, self.top_k = dc["block_size"], dc["mask_token_id"], dc["selector_top_k"]
        self.taps_conv, self.group = dc["conv_kernel_size"], dc["conv_group_size"]
        self.H, self.Hq, self.Hkv, self.D = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps, self.window, self.n_layers = cfg["rms_norm_eps"], cfg["sliding_window"], cfg["num_hidden_layers"]
        theta = cfg["rope_parameters"]["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.D, 2, dtype=torch.float32, device="cuda") / self.D))
        with safe_open(os.path.join(path, "model.safetensors"), "pt", device="cuda") as f:
            self.w = {k: f.get_tensor(k) for k in f.keys()}
        self.embed, self.lm_head = embed, lm_head  # target's, bf16

    def rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w.float()).to(torch.bfloat16)

    def rope(self, x, pos):  # x [T, heads, D], neox rotate-half over the full head
        ang = pos[:, None].float() * self.inv_freq[None]
        cos, sin = torch.cat([ang.cos()] * 2, -1)[:, None], torch.cat([ang.sin()] * 2, -1)[:, None]
        xf = x.float()
        rot = torch.cat([-xf[..., self.D // 2:], xf[..., :self.D // 2]], -1)
        return (xf * cos + rot * sin).to(torch.bfloat16)

    def lin(self, x, name):
        return x @ self.w[name].t()

    def context(self, feats):
        """feats [T, 5 * H] (target residuals after the tapped layers) -> per layer (K [T, Hkv, D], V [T, Hkv, D])."""
        h = self.rms(self.lin(feats, "fc.weight"), self.w["hidden_norm.weight"])
        T = h.shape[0]
        pos = torch.arange(T, device="cuda")
        out = []
        for l in range(self.n_layers):
            p = f"layers.{l}.self_attn."
            k = self.rms(self.lin(h, p + "k_proj.weight").view(T, self.Hkv, self.D), self.w[p + "k_norm.weight"])
            out.append((self.rope(k, pos), self.lin(h, p + "v_proj.weight").view(T, self.Hkv, self.D)))
        return out

    def conv(self, x, delta, base):
        """Dynamic grouped 2-tap conv over block positions: x [B, H], delta [B, taps, groups], base [taps, H]."""
        G = self.H // self.group
        blocks = x.float().view(self.B, G, self.group)
        coef = base.float().view(1, self.taps_conv, G, self.group) + delta.float().unsqueeze(-1)
        out = coef[:, 0] * blocks
        for tap in range(1, self.taps_conv):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coef[:, tap] * shifted  # rows < tap get the zero padding
        return out.reshape(self.B, self.H).to(torch.bfloat16)

    def conv_prepare(self, x, name):
        coef = self.lin(x, name + ".kernel_projection.weight").view(self.B, 2, self.taps_conv, self.H // self.group)
        return self.conv(x, coef[:, 0], self.w[name + ".base_kernel"][0]), coef[:, 1]

    def draft(self, anchor_tok, p, ctx):
        """One block at positions p .. p+7 (anchor x_p, masks); ctx: per-layer context K / V of positions < p.
        Returns the 7 draft tokens (greedy selector path)."""
        B = self.B
        ids = torch.full((B,), self.mask_id, dtype=torch.long, device="cuda")
        ids[0] = anchor_tok
        pos = torch.arange(p, p + B, device="cuda")
        x = self.embed[ids]
        residual = None
        lo = max(0, p + B - 1 - self.window + 1)  # sliding window: the block's last position sees back 2048
        for l in range(self.n_layers):
            L = f"layers.{l}."
            residual = x if residual is None else x + residual
            h = self.rms(residual, self.w[L + "input_layernorm.weight"])
            h, kern = self.conv_prepare(h, L + "attention_conv")
            a = L + "self_attn."
            q = self.rope(self.rms(self.lin(h, a + "q_proj.weight").view(B, self.Hq, self.D), self.w[a + "q_norm.weight"]), pos)
            k = self.rope(self.rms(self.lin(h, a + "k_proj.weight").view(B, self.Hkv, self.D), self.w[a + "k_norm.weight"]), pos)
            v = self.lin(h, a + "v_proj.weight").view(B, self.Hkv, self.D)
            ck, cv = ctx[l]
            K = torch.cat([ck[lo:p], k]).transpose(0, 1).repeat_interleave(self.Hq // self.Hkv, 0)  # [Hq, S, D]
            V = torch.cat([cv[lo:p], v]).transpose(0, 1).repeat_interleave(self.Hq // self.Hkv, 0)
            att = F.scaled_dot_product_attention(q.transpose(0, 1)[None].float(), K[None].float(), V[None].float())[0]
            o = self.lin(att.transpose(0, 1).reshape(B, -1).to(torch.bfloat16), a + "o_proj.weight")
            o = self.conv(o, kern, self.w[L + "attention_conv.base_kernel"][1])
            residual = o + residual
            h = self.rms(residual, self.w[L + "post_attention_layernorm.weight"])
            h, kern = self.conv_prepare(h, L + "mlp_conv")
            m = L + "mlp."
            h = self.lin(F.silu(self.lin(h, m + "gate_proj.weight")) * self.lin(h, m + "up_proj.weight"), m + "down_proj.weight")
            x = self.conv(h, kern, self.w[L + "mlp_conv.base_kernel"][1])
        hs = self.rms(x + residual, self.w["norm.weight"])[1:]                       # slots 1..7
        vals, cand = (hs @ self.lm_head.t()).float().topk(self.top_k, -1)           # [7, K]
        sel = "candidate_selector."
        hp = self.lin(hs, sel + "hidden_projection.weight").float()                 # [7, r]
        succ, pred = self.w[sel + "successor_codebook"], self.w[sel + "predecessor_codebook"]
        a_vec = pred[anchor_tok].float() * hp[0]  # slot 0: its predecessor is the verified anchor
        idx = int((vals[0] + (succ[cand[0]].float() @ a_vec)).argmax())
        path = [int(cand[0, idx])]
        for e in range(1, B - 1):
            a_vec = pred[cand[e - 1, idx]].float() * hp[e]
            idx = int((vals[e] + succ[cand[e]].float() @ a_vec).argmax())
            path.append(int(cand[e, idx]))
        return path


def simulate(dm, ctx, toks, p0):
    """Greedy cycles along toks (the target's greedy output from p0 on): (tokens, cycles)."""
    p, tok, cyc = p0, 0, 0
    T = len(toks)
    while p + dm.B < T:
        d = dm.draft(toks[p], p, ctx)
        n = 0
        while n < len(d) and d[n] == toks[p + 1 + n]:
            n += 1
        tok += n + 1
        cyc += 1
        p += n + 1
    return tok, cyc


def eval_greedy(a, model, path, dm):
    """The engine's greedy replies to tools/eval_drafter.py's prompts (MTP k=7 measures itself on the way), then DFlash2
    replayed on exactly those sequences."""
    from transformers import AutoTokenizer

    from drafter_data import build_prompts
    from engine.model.prefill import prefill
    from engine.spec.mtp import MtpGenerator
    tok = AutoTokenizer.from_pretrained(path)
    gen = MtpGenerator(model, path, max_seq_len=4096, k=7, weights=os.path.expanduser("~/.cache/colinfer/drafter/mtp_ft.safetensors"))
    res = collections.defaultdict(lambda: [0, 0, 0, 0])  # kind -> mtp tokens, mtp cycles, dflash tokens, dflash cycles
    both = collections.defaultdict(lambda: [0, 0])
    if a.both:
        import train_drafter as td
        from safetensors import safe_open as so
        cfg, _, embed, lm_draft, remap = td.setup()
        t = td.mtp_tensors(path)
        with so(os.path.expanduser("~/.cache/colinfer/drafter/mtp_ft.safetensors"), "pt", device="cuda") as f:
            for nme in f.keys():
                t[nme] = f.get_tensor(nme).to(torch.bfloat16)
        head = td.Head(t, cfg).cuda().eval()
        inv = torch.nonzero(remap >= 0)[:, 0][remap[remap >= 0].argsort()]  # draft-vocab index -> token id

        def mtp_hits(seq, H):
            """[7, T] bool: the MTP chain's depth-s draft made at row r (consuming x_{r+1}, h_r) equals x_{r+s+1}."""
            ids = torch.tensor(seq, device="cuda")
            e = embed[torch.cat([ids[1:], ids[-1:]])]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = head.unroll(e, H, 7)
            T = ids.numel()
            hit = torch.zeros(7, T, dtype=torch.bool)
            for s_, g in enumerate(outs, start=1):
                pred = inv[(g @ lm_draft.t()).argmax(-1)]
                r = torch.arange(T - s_ - 1, device="cuda")
                ok = torch.zeros(T, dtype=torch.bool, device="cuda")
                ok[: T - s_ - 1] = pred[r] == ids[r + s_ + 1]
                hit[s_ - 1] = ok.cpu()
            return hit
    for j, (prompt, kind, think) in enumerate(build_prompts(a.n_eval, random.Random(1))):
        x = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=think, tokenize=True)
        x = list(x["input_ids"] if hasattr(x, "keys") else x)[-3000:]
        s0 = dict(gen.stats)
        out = gen.generate(x, a.max_new, (248046, 248044))
        res[kind][0] += len(out)
        res[kind][1] += gen.stats["steps"] - s0["steps"]
        seq = x + out
        st = model.new_state(1, len(seq))
        _, Hf, Lh = prefill(model, torch.tensor([seq], device="cuda"), st, return_hidden=True, return_layers=TAPS)
        ctx = dm.context(Lh.reshape(len(seq), -1))
        tk, cy = simulate(dm, ctx, seq, len(x))
        res[kind][2] += tk
        res[kind][3] += cy
        if a.both:  # upper bound of two chains, one from each drafter: per cycle the longer accepted prefix
            hit = mtp_hits(seq, Hf)
            p, tk2, cy2 = len(x), 0, 0
            while p + dm.B < len(seq):
                d = dm.draft(seq[p], p, ctx)
                nd = 0
                while nd < len(d) and d[nd] == seq[p + 1 + nd]:
                    nd += 1
                nm = 0
                while nm < 7 and hit[nm, p - 1 + nm]:
                    nm += 1
                n = max(nd, nm)
                tk2 += n + 1; cy2 += 1; p += n + 1
            both[kind][0] += tk2; both[kind][1] += cy2
        if j % 10 == 9:
            print(f"[greedy] {j + 1}/{a.n_eval}", flush=True)
    print("tokens per cycle on the engine's greedy replies (MTP k=7 fine-tuned head | DFlash2 block 8):")
    tot = [0, 0, 0, 0]
    for kind, v in sorted(res.items()):
        print(f"  {kind:8s} MTP {v[0] / max(v[1], 1):.2f} | DFlash2 {v[2] / max(v[3], 1):.2f}")
        tot = [x + y for x, y in zip(tot, v)]
    print(f"  {'all':8s} MTP {tot[0] / tot[1]:.2f} | DFlash2 {tot[2] / tot[3]:.2f}")
    if a.both:
        print("best of both chains per cycle (upper bound):")
        tb = [0, 0]
        for kind, v in sorted(both.items()):
            print(f"  {kind:8s} {v[0] / max(v[1], 1):.2f}")
            tb = [x + y for x, y in zip(tb, v)]
        print(f"  {'all':8s} {tb[0] / tb[1]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--greedy", action="store_true", help="the engine's greedy replies to the eval prompts instead")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--both", action="store_true", help="--greedy: also the per-cycle best of an MTP chain and a DFlash2 chain")
    a = ap.parse_args()
    import train_drafter as td
    from engine.model.fast import load_fast_model, to_fast
    from engine.model.prefill import prefill
    from engine.weights.loader import PREFIX, dequant_nvfp4, resolve
    torch.set_grad_enabled(False)
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    model = to_fast(load_fast_model(path), kv_fp8=True)
    wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def get(n):
        with safe_open(os.path.join(path, wm[n]), "pt", device="cuda") as f:
            return f.get_tensor(n)
    embed = get(PREFIX + "embed_tokens.weight").to(torch.bfloat16)
    lm = dequant_nvfp4(get("lm_head.weight"), get("lm_head.weight_scale"), get("lm_head.weight_scale_2"), torch.bfloat16)
    dpath = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*"))[0]
    dm = DFlash2(dpath, embed, lm)
    if a.greedy:
        return eval_greedy(a, model, path, dm)
    # the 75 held-out replies of tools/train_drafter.py (same selection as twochain_sim.py)
    first = sorted(glob.glob(os.path.join(td.DIR, "feat", "shard_*.pt")))
    refs = [(f, i) for f in first for i in range(len(torch.load(f, mmap=True)))]
    random.Random(0).shuffle(refs)
    val = sorted(refs[:max(20, len(refs) // 20)])[: a.n]
    agg = collections.defaultdict(lambda: [0, 0])
    t0, cache = time.time(), {}
    for n_done, (f, i) in enumerate(val):
        if f not in cache:
            cache = {f: torch.load(f, mmap=True)}
        it = cache[f][i]
        ids = it["ids"].long()
        T, P = ids.numel(), it["P"]
        st = model.new_state(1, T)
        _, _, Lh = prefill(model, ids.view(1, -1).cuda(), st, return_hidden=True, return_layers=TAPS)
        ctx = dm.context(Lh.reshape(T, -1))
        tgt = it["ti"][:, 0].long()  # row r: the target's argmax for position r + 1
        p, tok, cyc = max(P, 1), 0, 0  # anchor = the reply's first token; drafts predict p + 1 ..
        while p + dm.B < T:
            d = dm.draft(int(ids[p]), p, ctx)
            n = 0
            while n < len(d) and d[n] == int(tgt[p + n]):
                n += 1
            tok += n + 1
            cyc += 1
            p += n + 1
        agg[it["kind"]][0] += tok; agg[it["kind"]][1] += cyc
        agg["all"][0] += tok; agg["all"][1] += cyc
        if n_done % 10 == 9:
            print(f"[dflash] {n_done + 1}/{len(val)} ({time.time() - t0:.0f}s): all {agg['all'][0] / agg['all'][1]:.2f} tokens/cycle", flush=True)
    print("DFlash2 (block 8, 7 drafts) tokens per cycle on the held-out replies:")
    for kind, (tk, cy) in sorted(agg.items()):
        print(f"  {kind:8s} {tk / cy:.2f}  ({cy} cycles)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""prefill_chunk_bench.py -- the prefill chunk of Qwen3.8-Flash-Next (engine/model/prefill_qwen4exp.py) by block, on
random weights at the real dims, with no checkpoint (every allocation stays well under 8 GB: no model lock needed).

  --part layers  the time of one layer of each block at T = 2048, 4096, 8192 tokens: the MoE (KernelMoE.prefill, 512
                 experts), a GDN layer, a QSA layer at the prefix 0 and 32k, a hyper-connection (hc_pre + hc_post), the
                 PLE layer; the sum over the 48 layers of a chunk (36 GDN, 12 QSA, 48 MoE, 96 hyper-connections, PLE),
                 the implied tok/s, and the peak memory of each block over its inputs
  --part attn    the sparse attention of a 2048-row chunk at the prefix 0, 16k, 128k: the selection (prefill_select),
                 then each path of qsa_kernels.prefill_attend (union, mask, list, and causal where it applies)
  --part ple     the n-gram feeder (engine.weights.ple.PleFeeder) on the real n-gram file: cold and warm rows of 16
                 chunks of 2048 tokens, the warm path at T = 1 and 16 (decode), and the hit rate of a row cache over the
                 WikiText token stream of tools/ple_hotset.py (CPU and drive only)

  --part real    the real checkpoint (about 80 GB: run it under the GPU lock): the WikiText prompt of --real-tokens tokens
                 prefilled in chunks of 4096 and of 2048 (one call each) and in one call per 4096-token chunk (the
                 scheduler's way): the state of the runs compared (GDN conv / rec, PLE conv, K / V, indexer keys, block keys)
                 and 16 greedy tokens after each; the time of each call per chunk (tok/s); then a WikiText prefill to
                 --long-tokens tokens in 4096-token calls, with the tok/s of each chunk, and, at the chunks that start at
                 --probe prefixes, the union size of the sparse attention (keys per 4-row group) and the time of the union
                 and the mask paths on each attention layer's real data

   ./py bench/prefill_chunk_bench.py --part layers attn
   flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py bench/prefill_chunk_bench.py --part real
   ./py bench/prefill_chunk_bench.py --part ple --ple-file <snapshot>/model-fp8-mtp-ple.safetensors
"""
import argparse
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from bench.timing import timed

DEV = "cuda"
H, C = 2560, 10240


def _rnd(g, *shape, scale=1.0):
    return (torch.randn(*shape, device=DEV, generator=g) * scale).bfloat16()


def _lin(g, n, k):
    return _rnd(g, n, k, scale=1 / math.sqrt(k))


def _mem(fn) -> float:
    """GB of the peak allocation of fn() over the memory before it."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e9


def gdn_weights(g):
    conv = SimpleNamespace(weight=_rnd(g, 10240, 1, 4, scale=0.3))
    return SimpleNamespace(qkvz=SimpleNamespace(w=_lin(g, 16384, H), sizes=[10240, 6144]), w_ba=_lin(g, 96, H), conv1d=conv,
                           A_log=torch.rand(48, device=DEV, generator=g), dt_bias=torch.randn(48, device=DEV, generator=g) * 0.5,
                           norm=SimpleNamespace(weight=_rnd(g, 128, scale=0.2) + 1, eps=1e-6), out_proj=SimpleNamespace(w=_lin(g, H, 6144)))


def qsa_weights(g):
    from engine.model.qwen4exp import Qwen4ExpConfig
    from engine.model.qwen35 import rope_inv_freq
    n = lambda d: SimpleNamespace(weight=_rnd(g, d, scale=0.2))  # noqa: E731
    return SimpleNamespace(qkvi=SimpleNamespace(w=_lin(g, 13952, H), sizes=[12288, 512, 512, 640]), q_norm=n(256), k_norm=n(256),
                           indexer=SimpleNamespace(q_layernorm=n(128), k_layernorm=n(128)), o_proj=SimpleNamespace(w=_lin(g, H, 6144)),
                           inv_freq=rope_inv_freq(Qwen4ExpConfig(), DEV), mrope=(11, 10))


def ple_weights(g):
    n = lambda: SimpleNamespace(weight=_rnd(g, C, scale=0.2), eps=1e-6)  # noqa: E731
    return SimpleNamespace(kv_proj=SimpleNamespace(w=_lin(g, 12800, H), sizes=[10240, 2560]), norm_key=n(), norm_query=n(), norm_conv=n(),
                           conv1d=torch.nn.Conv1d(C, C, 4, groups=C, dilation=3, bias=False).to(DEV, torch.bfloat16), scale=0.0078125)


def hc_weights(g, inj=True):
    from engine.model.hc_kernels import KernelGatedResidual
    return KernelGatedResidual(_rnd(g, C, scale=0.2), _lin(g, 320, C), _lin(g, 4, C) if inj else None, _lin(g, C, 320))


def state(L, g, prefix=0):
    f8 = torch.float8_e4m3fn
    st = SimpleNamespace(k={3: (torch.randn(1, 2, L, 256, device=DEV, generator=g) * 2).to(f8)},
                         v={3: torch.randn(1, 2, L, 256, device=DEV, generator=g).to(f8)},
                         ik={3: _rnd(g, 1, L, 128)}, ikb={3: torch.zeros(1, L // 4, 128, device=DEV, dtype=torch.bfloat16)},
                         rope_pos=torch.arange(L, device=DEV, dtype=torch.int32)[None, :, None].expand(1, L, 3).contiguous(),
                         conv={0: torch.zeros(1, 10240, 3, device=DEV, dtype=torch.bfloat16)}, rec={0: torch.zeros(1, 48, 128, 128, device=DEV)},
                         ple_conv={1: torch.zeros(1, C, 9, device=DEV, dtype=torch.bfloat16)}, pos_t=torch.zeros(1, device=DEV, dtype=torch.int32),
                         rope_delta=torch.zeros(1, device=DEV, dtype=torch.int32))
    if prefix:
        from engine.model import qsa_kernels as qk
        qk.qsa_block_keys(st.ik[3], st.ikb[3], _rnd(g, 128, scale=0.2), qsa_weights(g).inv_freq, st.pos_t, prefix, 1e-6, rope_pos=st.rope_pos)
    return st


def layers(Ts=(2048, 4096, 8192), prefixes=(0, 32768)):
    from engine.model import prefill_qwen4exp as pq
    from engine.model.moe_kernels import KernelMoE, random_moe_weights
    g = torch.Generator(device=DEV).manual_seed(0)
    moe = KernelMoE(**random_moe_weights(512, H, 640, generator=g))
    gdn, qsa, ple, hc = gdn_weights(g), qsa_weights(g), ple_weights(g), hc_weights(g)
    L = max(prefixes) + max(Ts) + 64
    sts = {p: state(L, g, p) for p in prefixes}
    print(f"{'T':>5s} {'block':<16s} {'ms/layer':>9s} {'layers':>6s} {'ms/chunk':>9s} {'peak GB':>8s}")
    for T in Ts:
        x4 = _rnd(g, T, C)
        x = _rnd(g, T, H)
        rows = torch.randint(0, 256, (T, 16, 160), dtype=torch.uint8, device=DEV, generator=g)
        steps = {"moe": (48, lambda: pq.moe_layer(x, moe)),
                 "gdn": (36, lambda: pq.gdn_layer(x, gdn, sts[0], 0)),
                 "hc (pre+post)": (96, lambda: _hc(pq, x4, hc, x)),
                 "ple": (1, lambda: pq.ple_layer(x4, rows, ple, sts[0], 1))}
        for p in prefixes:
            def run(p=p):
                sts[p].pos_t.fill_(p)
                return pq.qsa_layer(x, qsa, sts[p], 3, p)
            steps[f"qsa @{p // 1024}k"] = (12, run)
        tot = {p: 0.0 for p in prefixes}
        for name, (n, fn) in steps.items():
            dt = timed(fn, n=5, warmup=2)
            gb = _mem(fn)
            print(f"{T:5d} {name:<16s} {dt * 1e3:9.2f} {n:6d} {dt * n * 1e3:9.1f} {gb:8.2f}", flush=True)
            for p in prefixes:
                if not name.startswith("qsa") or name == f"qsa @{p // 1024}k":
                    tot[p] += dt * n
        for p in prefixes:
            print(f"{T:5d} chunk at the prefix {p // 1024}k: {tot[p] * 1e3:7.1f} ms, {T / tot[p]:6.0f} tok/s", flush=True)


def _hc(pq, x4, hc, o):
    y, inj = pq.hc_pre(x4, hc)
    pq.hc_post(x4, o, inj)


def attn(T=2048, prefixes=(0, 16384, 131072)):
    from engine.model import qsa_kernels as qk
    g = torch.Generator(device=DEV).manual_seed(1)
    for p in prefixes:
        L = p + T + 64
        kc = (torch.randn(1, 2, L, 256, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn)
        vc = torch.randn(1, 2, L, 256, device=DEV, generator=g).to(torch.float8_e4m3fn)
        q = _rnd(g, 1, 24, T, 256)
        gate = _rnd(g, T, 24 * 512)
        q_idx = _rnd(g, 1, T, 4, 128)
        ikb = _rnd(g, 1, L // 4, 128)
        sel, nsel = qk.prefill_select(q_idx, ikb, p, T)
        res = [f"select {timed(lambda: qk.prefill_select(q_idx, ikb, p, T), n=5, warmup=2) * 1e3:.2f} ms"]
        ul, uc = qk.prefill_union(sel, nsel, p, T)
        res.append(f"union size {uc.float().mean().item() * 4:.0f} keys / 4 rows")
        for path in ["union", "list", "mask"] + (["causal"] if p + T <= qk.DENSE_LEN else []):
            dt = timed(lambda: qk.prefill_attend(q, kc, vc, sel, nsel, p, 256 ** -0.5, gate, path), n=3, warmup=1)
            res.append(f"{path} {dt * 1e3:.2f} ms")
        print(f"attention T={T} at the prefix {p // 1024}k: " + ", ".join(res), flush=True)


def ple(path: str, tokens: int, cache_gb: float):
    from engine.model.qwen4exp import Qwen4ExpConfig
    from engine.weights.ple import ROW_BYTES, NgramTable, PleFeeder
    from tools.ple_hotset import load_tokenizer, token_stream
    cfg = Qwen4ExpConfig()
    tok = load_tokenizer(cfg)
    ids = token_stream("wikitext", tok, cfg.eos_token_id, tokens)
    table = NgramTable(path)
    eos = (cfg.eos_token_id,) * 2
    chunks = [ids[i * 2048:(i + 1) * 2048] for i in range(16)]
    f = PleFeeder(table, cache_rows=int(cache_gb * 1e9 / ROW_BYTES))
    t0 = time.perf_counter()
    for c in chunks:
        f.rows_for(c, eos)
    cold = time.perf_counter() - t0
    reads = f.gather_rows
    t0 = time.perf_counter()
    for c in chunks:
        f.rows_for(c, eos)
    warm = time.perf_counter() - t0
    n = 16 * 2048 * 16
    print(f"feeder, 16 chunks of 2048 tokens ({n / 1e6:.2f}M row slots, {reads / 1e6:.2f}M distinct rows read from the drive): "
          f"cold {cold * 1e3:.0f} ms ({n / cold / 1e6:.2f}M rows/s, {reads / cold / 1e3:.0f}k reads/s, {32768 / cold:.0f} tok/s), "
          f"warm {warm * 1e3:.0f} ms ({n / warm / 1e6:.2f}M rows/s, {32768 / warm:.0f} tok/s)", flush=True)
    for T in (1, 16):
        out = torch.empty(T, 16, 160, dtype=torch.uint8, pin_memory=True)
        c = ids[:T]
        f.rows_for(c, eos, out=out)
        t0 = time.perf_counter()
        for _ in range(200):
            f.rows_for(c, eos, out=out)
        print(f"feeder warm path T={T}: {(time.perf_counter() - t0) / 200 * 1e3:.3f} ms per call", flush=True)
    f.close()
    # the hit rate of the cache over the token stream, chunk by chunk (a fresh cache)
    f = PleFeeder(table, cache_rows=int(cache_gb * 1e9 / ROW_BYTES))
    t0 = time.perf_counter()
    prev = eos
    for i in range(0, ids.size, 2048):
        c = ids[i:i + 2048]
        f.rows_for(c, prev)
        prev = tuple(np.concatenate([prev, c])[-2:].tolist())
    hits, misses, held, cap = f.stats()
    print(f"row cache {cache_gb:.1f} GB ({cap / 1e6:.1f}M rows) over {ids.size / 1e6:.1f}M WikiText tokens: hit rate "
          f"{hits / (hits + misses):.1%} of the distinct rows of each chunk ({held / 1e6:.1f}M rows held), {time.perf_counter() - t0:.0f} s, "
          f"{ids.size / (time.perf_counter() - t0):.0f} tok/s", flush=True)
    f.close()
    table.close()


# ------------------------------------------------------------------------------------------------------------ the real model
class AttendProbe:
    """Wraps qsa_kernels.prefill_attend (the function that prefill_qwen4exp.qsa_layer calls). At a chunk whose pos0 is in
    `prefixes`, it records for each attention layer the union size (keys per group of 4 rows), the time of the union and the
    mask paths (CUDA events, median of 3) and their relative difference; the chunk's output is the default path's."""

    def __init__(self, prefixes):
        from engine.model import qsa_kernels as qk
        self.qk, self.orig, self.prefixes, self.rows = qk, qk.prefill_attend, set(prefixes), []
        qk.prefill_attend = self

    def close(self):
        self.qk.prefill_attend = self.orig

    def __call__(self, q, kc, vc, sel, nsel, pos0, scale, gate, path=None):
        if pos0 not in self.prefixes:
            return self.orig(q, kc, vc, sel, nsel, pos0, scale, gate, path)
        T = q.shape[2]
        ul, uc = self.qk.prefill_union(sel, nsel, pos0, T)
        res, outs = dict(pos0=pos0, T=T, keys_per_group=float(uc.float().mean()) * 4, keys_per_group_max=int(uc.max()) * 4,
                         selected_per_row=float(nsel.float().mean()) * 4), {}
        for path_ in ("union", "mask"):
            ts = []
            for _ in range(3):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                outs[path_] = self.orig(q, kc, vc, sel, nsel, pos0, scale, gate, path_)
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b))
            res[f"{path_}_ms"] = sorted(ts)[1]
        res["rel_union_mask"] = float((outs["union"].float() - outs["mask"].float()).norm() / outs["mask"].float().norm())
        self.rows.append(res)
        return outs[path or self.qk.PREFILL_PATH]


def _state_diff(a, b, n: int) -> dict:
    """The largest relative L2 difference (and max |delta|) per kind of state tensor of two one-slot FastState4 over the
    first n positions."""
    def rel(x, y):
        x, y = x.float(), y.float()
        return float((x - y).norm() / y.norm().clamp_min(1e-30)), float((x - y).abs().max())
    kinds = {"gdn conv": [(a.conv[i], b.conv[i]) for i in a.conv], "gdn rec": [(a.rec[i], b.rec[i]) for i in a.rec],
             "ple conv": [(a.ple_conv[i], b.ple_conv[i]) for i in a.ple_conv],
             "k": [(a.k[i][:, :, :n], b.k[i][:, :, :n]) for i in a.k], "v": [(a.v[i][:, :, :n], b.v[i][:, :, :n]) for i in a.v],
             "ik": [(a.ik[i][:, :n], b.ik[i][:, :n]) for i in a.ik], "ikb": [(a.ikb[i][:, :n // 4], b.ikb[i][:, :n // 4]) for i in a.ikb]}
    def same(x, y):
        return torch.equal(x.view(torch.uint8), y.view(torch.uint8)) if x.dtype == torch.float8_e4m3fn else torch.equal(x, y)
    out = {}
    for k, pairs in kinds.items():
        r = [rel(x, y) for x, y in pairs]
        out[k] = dict(rel=max(v[0] for v in r), max_abs=max(v[1] for v in r), equal=all(same(x, y) for x, y in pairs),
                      per_layer=[v[0] for v in r])
    out["rope_pos"] = dict(equal=torch.equal(a.rope_pos[:, :n], b.rope_pos[:, :n]))
    return out


def _greedy(model, st, logits, ids, n: int) -> list[int]:
    """n greedy tokens after a prefill: eager decode steps on st, the n-gram rows from the host (model.host_rows)."""
    ctx, out = list(ids[-2:]), []
    for _ in range(n):
        t = int(logits.argmax())
        out.append(t)
        rows = model.host_rows([[t]], [ctx]).cuda()
        ctx = [ctx[1], t]
        logits = model(torch.tensor([[t]], device=DEV), st, rows)[0]
    return out


def _calls(pq, model, st, ids, chunk, log=None) -> list[float]:
    """Prefill ids in one call per chunk (prev2 from the prompt: the scheduler's way); the seconds of each call."""
    eos = model.cfg.eos_token_id
    ts, out = [], None
    for c0 in range(0, len(ids), chunk):
        x = torch.tensor([ids[c0:c0 + chunk]], device=DEV)
        prev2 = ([eos, eos] + ids[:c0])[-2:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pq.prefill(model, x, st, chunk=chunk, prev2=prev2)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        if log:
            log(c0, x.shape[1], ts[-1])
    return ts, out


def real(name: str, n_tok: int, n_long: int, probes, out: str | None = None):
    import json

    from transformers import AutoTokenizer

    from engine.model import prefill_qwen4exp as pq
    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from engine.weights.loader import resolve
    from engine.weights.ple import PleFeeder
    from tests.perplexity import wikitext_test
    path, model = load_fast_qwen4exp(name)
    ple = model.ple_layer()
    model.feeder = PleFeeder(ple.table, cache_rows=6_250_000, params=ple.params, image_token_id=model.cfg.image_token_id)
    tok = AutoTokenizer.from_pretrained(resolve(path))
    all_ids = tok(wikitext_test(), return_tensors="pt").input_ids[0][:max(3 * n_tok, n_long)].tolist()
    ids = all_ids[:n_tok]
    eos = model.cfg.eos_token_id
    L = -(-(n_tok + 64) // 16) * 16
    res = {}
    # 1. the same prompt in chunks of 4096 and 2048 (one call each), and in one call per 4096-token chunk
    states, first = {}, {}
    for tag, chunk, per_call in (("4096", 4096, False), ("2048", 2048, False), ("4096 per call", 4096, True)):
        st = model.new_state(1, L)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if per_call:
            _, logits = _calls(pq, model, st, ids, chunk)
        else:
            logits = pq.prefill(model, torch.tensor([ids], device=DEV), st, chunk=chunk)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        h, m, _, _ = model.feeder.stats(reset=True)
        print(f"[real] prefill of {n_tok} WikiText tokens, chunk {tag}: {dt:.2f} s ({n_tok / dt:.0f} tok/s; feeder {h} hits, {m} misses)",
              flush=True)
        states[tag], first[tag] = st, logits[0].clone()
        res[f"prefill_s {tag}"] = dt
    for tag in ("2048", "4096 per call"):
        d = _state_diff(states[tag], states["4096"], n_tok)
        lg = (first[tag] - first["4096"]).abs().max().item()
        res[f"state {tag} vs 4096"] = d
        print(f"[real] state after chunk {tag} against chunk 4096: " +
              "; ".join(f"{k} " + ("equal" if v.get("equal") else f"rel {v['rel']:.2e} max {v['max_abs']:.3g}") for k, v in d.items()) +
              f"; last logits max |delta| {lg:.4f}", flush=True)
        for k, v in d.items():
            if "per_layer" in v and not v["equal"]:
                print(f"[real]     {k} rel by layer: " + " ".join(f"{x:.1e}" for x in v["per_layer"]), flush=True)
    toks = {tag: _greedy(model, states[tag], first[tag], ids, 16) for tag in states}
    res["greedy"] = toks
    for tag in ("2048", "4096 per call"):
        same = toks[tag] == toks["4096"]
        i = next((j for j, (a, b) in enumerate(zip(toks[tag], toks["4096"])) if a != b), None)
        print(f"[real] 16 greedy tokens after chunk {tag}: {'equal to' if same else f'differ at token {i} from'} chunk 4096", flush=True)
    del states, first
    torch.cuda.empty_cache()
    # 2. the time of each call (warm row cache: the same prompt again)
    for chunk in (4096, 2048):
        st = model.new_state(1, L)
        ts, _ = _calls(pq, model, st, ids, chunk)
        res[f"calls {chunk}"] = ts
        print(f"[real] one call per {chunk}-token chunk, warm rows: " +
              ", ".join(f"@{c0 // 1024}k {chunk / t:.0f} tok/s ({t * 1e3:.0f} ms)" for c0, t in zip(range(0, n_tok, chunk), ts)), flush=True)
        del st
    # 2b. cold rows, one call per 4096-token chunk: without and with next_ids (two regions of the text that no run read yet)
    if len(all_ids) >= 3 * n_tok:
        for tag, a0, ahead in (("without next_ids", n_tok, False), ("with next_ids", 2 * n_tok, True)):
            part = all_ids[a0:a0 + n_tok]
            st = model.new_state(1, L)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for c0 in range(0, n_tok, 4096):
                nxt = part[c0 + 4096:c0 + 8192] if ahead else None
                pq.prefill(model, torch.tensor([part[c0:c0 + 4096]], device=DEV), st, prev2=([eos, eos] + part[:c0])[-2:], next_ids=nxt)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            res[f"cold {tag}"] = dt
            print(f"[real] {n_tok} tokens with cold rows, one call per 4096-token chunk {tag}: {dt:.2f} s ({n_tok / dt:.0f} tok/s)", flush=True)
            del st
    # 3. a long WikiText prefill in 4096-token calls, the attention probe at the given prefixes
    if n_long:
        st = model.new_state(1, -(-(n_long + 64) // 16) * 16)
        probe = AttendProbe(probes)
        rows = []

        def log(c0, T, t):
            rows.append((c0, T, t))
            if c0 // 4096 % 4 == 0 or c0 in probes:
                print(f"[real]   long prefill: the chunk at {c0 // 1024}k: {T / t:.0f} tok/s ({t * 1e3:.0f} ms){' (probed)' if c0 in probes else ''}",
                      flush=True)
        try:
            _calls(pq, model, st, all_ids[:n_long], 4096, log)
        finally:
            probe.close()
        res["long_calls"] = rows
        res["probe"] = probe.rows
        for p in sorted(probes):
            r = [x for x in probe.rows if x["pos0"] == p]
            if not r:
                continue
            mean = {k: sum(x[k] for x in r) / len(r) for k in ("keys_per_group", "selected_per_row", "union_ms", "mask_ms")}
            kmax, rmax = max(x["keys_per_group_max"] for x in r), max(x["rel_union_mask"] for x in r)
            print(f"[real] sparse attention at the prefix {p // 1024}k (T = {r[0]['T']}, {len(r)} layers): union {mean['keys_per_group']:.0f} keys "
                  f"per 4-row group (max {kmax}; one row selects {mean['selected_per_row']:.0f}); union {mean['union_ms']:.2f} ms, mask "
                  f"{mean['mask_ms']:.2f} ms per layer; rel(union, mask) up to {rmax:.1e}", flush=True)
            for x in r:
                print(f"[real]     {x['keys_per_group']:7.0f} keys/group  union {x['union_ms']:6.2f} ms  mask {x['mask_ms']:7.2f} ms  "
                      f"rel {x['rel_union_mask']:.1e}")
    h, m, held, cap = model.feeder.stats()
    print(f"[real] feeder after the long prefill: {held} rows held of {cap}; {model.feeder.gather_rows} rows read from the file in all")
    if out:
        json.dump(res, open(out, "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", nargs="+", default=["layers", "attn"], choices=["layers", "attn", "ple", "real"])
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--real-tokens", type=int, default=12288, help="--part real: the prompt of the chunk comparison")
    ap.add_argument("--long-tokens", type=int, default=65536 + 4096, help="--part real: the long prefill (0: skip it)")
    ap.add_argument("--json", help="--part real: write the results here")
    ap.add_argument("--probe", type=int, nargs="*", default=[8192, 65536], help="--part real: the prefixes of the attention probe")
    ap.add_argument("--ple-file", default=None, help="the checkpoint's model-fp8-mtp-ple.safetensors")
    ap.add_argument("--tokens", type=int, default=4_000_000, help="--part ple: WikiText tokens of the hit-rate run")
    ap.add_argument("--cache-gb", type=float, default=2.0)
    a = ap.parse_args()
    if "layers" in a.part:
        layers()
    if "attn" in a.part:
        attn()
    if "ple" in a.part:
        ple(a.ple_file, a.tokens, a.cache_gb)
    if "real" in a.part:
        real(a.model, a.real_tokens, a.long_tokens, a.probe, a.json)


if __name__ == "__main__":
    with torch.inference_mode():
        main()

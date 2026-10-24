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

   ./py bench/prefill_chunk_bench.py --part layers attn
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", nargs="+", default=["layers", "attn"], choices=["layers", "attn", "ple"])
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


if __name__ == "__main__":
    with torch.inference_mode():
        main()

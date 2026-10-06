"""MTP drafter (PLAN.md 4.5 item 2): the checkpoint's multi-token-prediction block (one gated full-attention
decoder layer + fc, BF16, excluded from quantization) run k times per step, with its own KV cache.

Row semantics (as vLLM's Qwen3_5MTP / EAGLE proposer): MTP row at position i takes (embed(x_{i+1}),
h_i) where h_i is the target's post-final-norm hidden state at position i, and predicts x_{i+2}:
    x = fc([pre_fc_norm_embedding(embed(x_{i+1})), pre_fc_norm_hidden(h_i)]) -> decoder layer -> mtp.norm
    -> shared lm_head.  Chained steps feed the MTP's own normed output as the next hidden.

MtpCycle captures one CUDA graph per decode cycle for B slots (k drafts each):
  1. verify [y, d1..dk] of every slot on the target (hidden states kept); acceptance (drafts that equal the
     target's argmax, or its position-keyed sample for slots with temperature > 0, so the output is exactly what
     plain decoding would emit); the accepted length is cut after the first accepted stop token
     so the state never runs past the end of a reply; commit n[b] tokens per slot (0 for inactive slots);
  2. MTP catch-up: rows for the n newly committed positions with the true target hidden states (rows past
     n are padding, overwritten later); the last valid row yields the next d1;
  3. k-1 chained MTP steps yield d2..dk; the next cycle's input [y', d1'..dk'] is written in place.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from engine.kernels import ops
from engine.model.fast import GEMV_M, DecodeGraph, fp8_rows, FastQwen35, FastState, KernelAttention, KernelRMSNorm, fast_layer_forward
from engine.model.prefill import ATTN_FP8, ATTN_FP8_BN, ATTN_FP8_MIN_CTX, prefill, prepare_prefill
from engine.model.qwen35 import DecoderLayer, RMSNorm


class Bf16Linear(nn.Module):
    """BF16 weights; decode-sized inputs go through a GEMV (BF16, or FP8 with per-row scales after
    to_fp8(), halving the weight stream), long inputs (the prompt pass) through cuBLAS BF16."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.register_buffer("w", w.contiguous(), persistent=False)
        self.out_features, self.in_features = w.shape
        self.w8 = self.rs = None

    def to_fp8(self):
        rs = (self.w.float().abs().amax(1) / 448.0).clamp_min(1e-12)
        self.w8 = (self.w.float() / rs[:, None]).to(torch.float8_e4m3fn).contiguous()
        self.rs = rs.contiguous()
        return self

    def to_lowbit(self, fmt: str):
        """Decode copy in INT6 / INT5 / NVFP4 (block-16 e4m3 scales, round-to-nearest from the BF16 weights) for the
        tensor-core skinny GEMM: the drafter re-reads its weights every draft step, so its bytes are cycle time."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
        from engine.model.fast import Int6Linear, Nvfp4Linear
        w = self.w.float()
        if fmt == "nvfp4":
            from requant_nvfp4 import quantize
            gs = float(w.abs().max()) / (448.0 * 6.0)
            packed, sf = quantize(w, gs)
            self.low = Nvfp4Linear(packed, sf, gs)
        else:
            from int6_requant import pack5, pack6, quantize_int
            bits = {"int6": 6, "int5": 5}[fmt]
            gs = float(w.abs().max()) / (448.0 * (2 ** (bits - 1) - 1))
            codes, sf = quantize_int(w, gs, bits)
            lo, hi = (pack6 if bits == 6 else pack5)(codes)
            self.low = Int6Linear(lo, hi, sf, gs)
        return self

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        if x2.shape[0] <= 16:
            out = torch.empty(x2.shape[0], self.out_features, device=x.device, dtype=torch.bfloat16)
            if getattr(self, "low", None) is not None:
                self.low.rows(x2, None, out)
            elif self.w8 is not None:
                fp8_rows(x2, self.w8, 1.0, None, out, self.rs)
            else:
                for i in range(0, x2.shape[0], GEMV_M):
                    ops().bf16_gemv(x2[i:i + GEMV_M], self.w, out[i:i + GEMV_M])
        else:
            out = F.linear(x2, self.w)
        out = out.view(*shp[:-1], self.out_features)
        return out if residual is None else out + residual


class Bf16MLP(nn.Module):
    def __init__(self, g, u, d):
        super().__init__()
        self.gate, self.up, self.down = Bf16Linear(g), Bf16Linear(u), Bf16Linear(d)

    def forward(self, x, residual=None):
        return self.down(F.silu(self.gate(x)) * self.up(x), residual)


class MtpState(FastState):
    """KV cache of the single MTP attention layer (index 0) for `batch` slots plus their device positions.
    active: share the target state's mask so a graph step never writes the MTP cache of an inactive slot.
    kv_fp8: e4m3 cache like the target's (default; COLINFER_MTP_KV_FP8=0 for bf16). Each of the k draft steps of a cycle
    reads the drafter's whole cache (512 MB per step per slot at 128k in bf16), so fp8 makes long-context cycles ~10%
    faster (126 vs 140 ms, k=7 at 128k) at unchanged acceptance (tools/eval_drafter.py). Drafts change speed, never outputs."""

    def __init__(self, cfg, max_seq_len, device, batch: int = 1, active: torch.Tensor | None = None, kv_fp8: bool | None = None):
        if kv_fp8 is None:
            kv_fp8 = os.environ.get("COLINFER_MTP_KV_FP8", "1") != "0"
        self.cfg, self.max_seq_len, self.kv_fp8, self.kv_fp4, self.pos = cfg, max_seq_len, kv_fp8, False, 0
        self.conv, self.rec = {}, {}
        self.k = {0: torch.zeros(batch, cfg.num_key_value_heads, max_seq_len, cfg.head_dim, device=device,
                                 dtype=torch.float8_e4m3fn if kv_fp8 else torch.bfloat16)}
        self.v = {0: torch.zeros_like(self.k[0])}
        self.pos_t = torch.zeros(batch, dtype=torch.int32, device=device)
        self.active = active if active is not None else torch.ones(batch, dtype=torch.int32, device=device)
        self.arange = torch.arange(16, device=device)

    def slot(self, b: int) -> FastState:
        """Single-slot view with its own all-ones mask, for eager (non-graph) MTP work on one slot."""
        v = self.view(b, b + 1)
        v.active = torch.ones(1, dtype=torch.int32, device=self.pos_t.device)
        return v

class Mtp(nn.Module):
    """fp8: MTP linears stream FP8 (per-row scales) instead of BF16 when drafting.
    draft_vocab: draft only among the N most frequent tokens (engine/spec/draft_vocab.npy): the drafting
    lm_head reads N rows instead of 248k. Neither affects correctness, only acceptance."""

    def __init__(self, target: FastQwen35, path: str, fp8: bool = False, draft_vocab: int | None = None, prompt_slots: int = 4096,
                 weights: str | None = None):
        """weights: optional safetensors with mtp.* tensors replacing the checkpoint's (tools/train_drafter.py)."""
        super().__init__()
        cfg = target.cfg
        wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        t = {}
        for name, f in wm.items():
            if name.startswith("mtp."):
                with safe_open(os.path.join(path, f), framework="pt", device="cuda") as sf:
                    t[name[4:]] = sf.get_tensor(name).to(torch.bfloat16)
        if weights:
            with safe_open(weights, framework="pt", device="cuda") as sf:
                for name in sf.keys():
                    assert name.startswith("mtp.") and name[4:] in t, name
                    t[name[4:]] = sf.get_tensor(name).to(torch.bfloat16)
        P = "layers.0."
        self.fc = Bf16Linear(t["fc.weight"])
        self.pre_e = self._norm(t["pre_fc_norm_embedding.weight"], cfg)
        self.pre_h = self._norm(t["pre_fc_norm_hidden.weight"], cfg)
        self.norm = self._norm(t["norm.weight"], cfg)
        import dataclasses
        with torch.device("meta"):
            layer = DecoderLayer(dataclasses.replace(cfg, layer_types=["full_attention"], num_hidden_layers=1), 0)
        layer.input_layernorm = self._norm(t[P + "input_layernorm.weight"], cfg)
        layer.post_attention_layernorm = self._norm(t[P + "post_attention_layernorm.weight"], cfg)
        a = layer.self_attn
        a.q_proj, a.k_proj, a.v_proj, a.o_proj = (Bf16Linear(t[P + f"self_attn.{n}_proj.weight"]) for n in "qkvo")
        a.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps).to("cuda")
        a.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps).to("cuda")
        with torch.no_grad():
            a.q_norm.weight = nn.Parameter(t[P + "self_attn.q_norm.weight"], requires_grad=False)
            a.k_norm.weight = nn.Parameter(t[P + "self_attn.k_norm.weight"], requires_grad=False)
        a.__class__ = KernelAttention
        d = cfg.rotary_dim
        a.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device="cuda") / d))
        layer.mlp = Bf16MLP(t[P + "mlp.gate_proj.weight"], t[P + "mlp.up_proj.weight"], t[P + "mlp.down_proj.weight"])
        layer.forward = fast_layer_forward.__get__(layer)
        self.layer = layer
        self.embed, self.lm_head, self.cfg = target.embed_tokens, target.lm_head, cfg
        self.nbytes = sum(v.numel() * 2 for v in t.values())
        if fp8:
            # the draft steps' weight stream: nvfp4 (default; k=7 cycle 103.8 -> 98.4 ms, acceptance 3.45 -> 3.41 tokens per
            # cycle, docs/phase6_progress.md section 17) | int6 | int5 | fp8 (per-row scales)
            fmt = os.environ.get("COLINFER_MTP_FORMAT", "nvfp4")
            for mod in [self.fc, a.q_proj, a.k_proj, a.v_proj, a.o_proj, layer.mlp.gate, layer.mlp.up, layer.mlp.down]:
                mod.to_fp8() if fmt == "fp8" else mod.to_lowbit(fmt)
        self.vocab_ids = None
        self.lm_draft = target.lm_head
        if draft_vocab:
            import numpy as np
            from engine.model.fast import Nvfp4Linear
            ids = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab.npy"))[:draft_vocab]
            # static frequent tokens + `prompt_slots` rows rewritten per request with the prompt's other tokens
            self.n_static = len(ids)
            self.static_set = set(int(i) for i in ids)
            allids = np.concatenate([ids, np.full(prompt_slots, ids[0], dtype=ids.dtype)])
            self.vocab_ids = torch.tensor(allids.astype(np.int64), device="cuda")
            lm = target.lm_head
            self.lm_draft = Nvfp4Linear(lm.w[self.vocab_ids].contiguous(), lm.sf[self.vocab_ids].contiguous(), lm.gscale, out_fp32=True)

    def set_prompt_vocab(self, prompt: list[int]):
        """Fill the per-request draft-vocab rows with prompt tokens missing from the static set."""
        if self.vocab_ids is None:
            return
        slots = self.vocab_ids.numel() - self.n_static
        extra = [t for t in dict.fromkeys(prompt) if t not in self.static_set][:slots]
        extra += [int(self.vocab_ids[0])] * (slots - len(extra))
        ids = torch.tensor(extra, dtype=torch.long, device="cuda")
        self.vocab_ids[self.n_static:] = ids
        lm = self.lm_head
        self.lm_draft.w[self.n_static:] = lm.w[ids]
        self.lm_draft.sf[self.n_static:] = lm.sf[ids]

    def draft(self, g: torch.Tensor, prob: bool = False):
        """Greedy draft token ids from MTP outputs g [..., H] -> [...]; prob: also the drafter's probability of each
        (softmax over the draft vocabulary)."""
        lg = self.lm_draft(g)
        idx = lg.argmax(-1)
        ids = idx if self.vocab_ids is None else self.vocab_ids[idx]
        if not prob:
            return ids
        return ids, torch.exp(lg.amax(-1) - torch.logsumexp(lg, -1))

    @staticmethod
    def _norm(w, cfg):
        n = RMSNorm(w.numel(), cfg.rms_norm_eps)
        n.weight = nn.Parameter(w, requires_grad=False)
        return KernelRMSNorm(n)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor, st: MtpState):
        """tokens [B, T], hidden [B, T, H] at positions st.pos_t + t (B * T <= 16): MTP output (normed) [B, T, H]."""
        x = self.fc(torch.cat([self.pre_e(self.embed(tokens)), self.pre_h(hidden)], -1))
        return self.norm(self.layer(x, None, None, st))

    @torch.inference_mode()
    def prefill(self, tokens: torch.Tensor, hidden: torch.Tensor, st: MtpState, chunk: int = 2048) -> torch.Tensor:
        """Long-T MTP pass from position st.pos_t (prompt rows); returns the normed output of the last row [1, H]."""
        import flashinfer
        a, layer = self.layer.self_attn, self.layer
        T_all = tokens.shape[0]
        g = None
        for c0 in range(0, T_all, chunk):
            tk, hd = tokens[c0:c0 + chunk], hidden[c0:c0 + chunk]
            T = tk.shape[0]
            x = self.fc(torch.cat([self.pre_e(self.embed(tk)), self.pre_h(hd)], -1))
            h = layer.input_layernorm(x)
            qp, kp, vp = a.q_proj(h), a.k_proj(h), a.v_proj(h)
            q = torch.empty(1, a.num_heads, T, a.head_dim, device=x.device, dtype=torch.bfloat16)
            ops().attn_prologue(qp, kp, vp, a.q_norm.weight, a.k_norm.weight, a.inv_freq, st.pos_t, st.k[0], st.v[0], q, a.q_norm.eps)
            p0 = int(st.pos_t)
            L = p0 + T
            kk, vv = st.k[0][0:1], st.v[0][0:1]
            if kk.dtype == torch.float8_e4m3fn and ATTN_FP8 and L > ATTN_FP8_MIN_CTX:  # as the target's prefill
                o = torch.empty(T, a.num_heads, a.head_dim, device=x.device, dtype=torch.bfloat16)
                ops().attn_prefill_fp8(q, kk, vv, o.view(T, -1), p0, a.head_dim ** -0.5, ATTN_FP8_BN)
            else:
                kk, vv = kk[0, :, :L], vv[0, :, :L]
                if kk.dtype == torch.float8_e4m3fn:  # FlashInfer reads a bf16 copy
                    kk, vv = kk.to(torch.bfloat16), vv.to(torch.bfloat16)
                o = flashinfer.single_prefill_with_kv_cache(q[0].transpose(0, 1), kk, vv, causal=True, kv_layout="HND",
                                                            sm_scale=a.head_dim ** -0.5)
            gate = qp.view(T, a.num_heads, 2 * a.head_dim)[:, :, a.head_dim:]
            x = a.o_proj((o * torch.sigmoid(gate)).reshape(T, -1), x)
            x = layer.mlp(layer.post_attention_layernorm(x), x)
            g = self.norm(x[-1:])
            st.pos_t += T
        return g

    @torch.inference_mode()
    def first_drafts(self, tokens: torch.Tensor, hidden: torch.Tensor, st: FastState, k: int) -> list[int]:
        """MTP rows for (tokens[i], hidden[i]) from position st.pos_t (eager, one slot), then k-1 chained steps:
        the k drafts that follow the last token. st: a single-slot view with an all-ones mask (MtpState.slot)."""
        g = self.prefill(tokens, hidden, st)
        d = [self.draft(g).view(1, 1)]
        gp = g.view(1, 1, -1)
        for _ in range(k - 1):
            gp = self(d[-1], gp, st)
            d.append(self.draft(gp).view(1, 1))
            st.pos_t += 1
        return torch.cat(d, 1)[0].tolist()


# Draft early exit: once the product of the drafter's probabilities of a cycle's drafts so far is below this for every
# active slot, the remaining draft steps skip their GEMMs (their drafts are junk that verify rejects; outputs are
# unchanged). Simulated on the k = 7 cycle mix: 35.3 -> 36.6 tok/s at 0.2 (an oracle stop: 38.8). 0 disables.
DRAFT_STOP = float(os.environ.get("COLINFER_DRAFT_STOP", "0.1"))

# GDN commit concurrent with the MTP drafting (a second branch of the cycle graph); COLINFER_COMMIT_OVERLAP=0: serial
COMMIT_OVERLAP = os.environ.get("COLINFER_COMMIT_OVERLAP", "1") != "0"


class MtpCycle:
    """CUDA graph of one speculative cycle with the MTP drafter for B slots (k drafts each).

    tok [B, k+1] (the next cycle's input [y, d1..dk] per slot), stop_ids [B, S] (int64, -1 = unused) and
    params may be views into buffers shared by the graphs of every batch width, so a slot keeps its pending
    input when the width changes. params: SamplerParams -> sampled cycle (slots with temperature > 0 accept drafts
    that equal the target's position-keyed sample, so the output is exactly plain sampling; greedy for the rest;
    engine/spec/accept.py); None -> greedy cycle.
    After replay: out_tok [B, k+1] (first n[b] valid), n [B] int32, logits [B, k+1, V] fp32 (verify rows,
    raw), H [B, k+1, hidden] (target post-norm hidden of the verify rows)."""

    def __init__(self, model: FastQwen35, mtp: Mtp, state: FastState, mst: MtpState, k: int = 3, params=None,
                 tok: torch.Tensor | None = None, stop_ids: torch.Tensor | None = None):
        self.model, self.mtp, self.state, self.mst, self.k, self.params = model, mtp, state, mst, k, params
        dev = state.pos_t.device
        B = state.pos_t.shape[0]
        self.tok = tok if tok is not None else torch.zeros(B, k + 1, dtype=torch.long, device=dev)
        self.stop_ids = stop_ids if stop_ids is not None else torch.full((B, 1), -1, dtype=torch.long, device=dev)
        assert self.tok.shape == (B, k + 1) and self.stop_ids.shape[0] == B
        self.idx = torch.arange(k + 1, device=dev)
        self._side = torch.cuda.Stream()
        self.skip = torch.zeros(1, dtype=torch.int32, device=dev)
        state.reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.out_tok, self.n, self.logits, self.H = self._body()
        state.reset()
        mst.pos_t.zero_()

    def _body(self):
        k, m, mtp, st = self.k, self.model, self.mtp, self.state
        tok = self.tok
        B = tok.shape[0]
        logits, H = m.verify(tok, st, return_hidden=True)            # [B, k+1, V], [B, k+1, H]
        if self.params is None:
            pred = logits.argmax(-1)                                 # [B, k+1]
        else:  # the target's own sample at every row, keyed by position (engine/spec/accept.py): output = plain sampling
            from engine.spec.accept import draw
            pred = draw(logits, self.params, st.pos_t + 1)           # row i predicts position pos + i + 1
        match = (pred[:, :k] == tok[:, 1:]).int()
        n = (1 + match.cumprod(-1).sum(-1)).int()                    # [B] accepted inputs, 1..k+1
        bonus = pred.gather(1, (n.long() - 1)[:, None])[:, 0]        # [B]
        # cut after the first accepted stop token: drafts d1..dk are out positions 0..k-1, accepted while < n-1
        d = tok[:, 1:]
        hit = (d[:, :, None] == self.stop_ids[:, None, :]).any(-1) & (self.idx[None, :k] < (n - 1)[:, None])
        first = hit.int().argmax(-1)
        has = hit.any(-1)
        n = torch.where(has, (first + 1).int(), n) * st.active
        bonus = torch.where(has, d.gather(1, first[:, None])[:, 0], bonus)
        p0 = st.pos_t.clone()
        if COMMIT_OVERLAP:  # the drafting below never reads the GDN state: commit it on a parallel graph branch
            side = self._side
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                m.commit(st, n)
        else:
            m.commit(st, n)
        nl = n.long()
        out_tok = torch.cat([d, bonus[:, None]], 1)                  # accepted drafts are d[:n-1]; the bonus is out_tok[n-1]
        out_tok = torch.where(self.idx[None] == nl[:, None] - 1, bonus[:, None], out_tok)
        # MTP catch-up: rows at positions p0 .. p0+k with tokens x_{i+1} (= out_tok[i] for i < n) and true hidden H[i]
        self.mst.pos_t.copy_(p0)
        g = mtp(out_tok, H, self.mst)                                # [B, k+1, H]
        last = (nl - 1).clamp_min(0)
        gp = g.gather(1, last[:, None, None].expand(B, 1, g.shape[-1]))
        stop = DRAFT_STOP > 0 and k > 1
        d, cp = mtp.draft(gp, prob=True) if stop else (mtp.draft(gp), None)
        dr = [d]                                                     # [B, 1]
        self.mst.pos_t.copy_(p0 + n)
        if stop:
            self.skip.zero_()
        for _ in range(k - 1):
            if stop:  # every slot's chain is unlikely to survive this far: the remaining steps skip their GEMMs
                self.skip.copy_(torch.maximum(self.skip, ((cp[:, 0] < DRAFT_STOP) | (st.active == 0)).all().int().view(1)))
                ops().skinny_skip(self.skip)
            try:
                gp = mtp(dr[-1], gp, self.mst)
                if stop:
                    d, p = mtp.draft(gp, prob=True)
                    cp = cp * p
                else:
                    d = mtp.draft(gp)
            finally:
                if stop:
                    ops().skinny_skip(None)
            dr.append(d)
            self.mst.pos_t += 1
        nxt = torch.cat([bonus[:, None]] + dr, 1)                    # [B, k+1] = [y', d1'..dk']
        self.tok.copy_(torch.where(st.active[:, None] > 0, nxt, tok))
        if COMMIT_OVERLAP:
            torch.cuda.current_stream().wait_stream(self._side)
        return out_tok, n, logits, H


class MtpGenerator:
    """Generation with MTP speculation for one slot; greedy output identical to plain greedy decode."""

    def __init__(self, model: FastQwen35, path: str, max_seq_len: int = 32768, k: int = 3, fp8: bool = True, draft_vocab: int | None = 65536,
                 weights: str | None = None):
        prepare_prefill(model)
        self.model, self.k = model, k
        self.mtp = Mtp(model, path, fp8=fp8, draft_vocab=draft_vocab, weights=weights)
        self.state = model.new_state(1, max_seq_len)
        self.mst = MtpState(model.cfg, max_seq_len, "cuda", active=self.state.active)
        from engine.runtime.sampler import SamplerParams
        self.plain = DecodeGraph(model, self.state)
        self.params = SamplerParams(1, model.cfg.vocab_size, "cuda")
        self.cycle = MtpCycle(model, self.mtp, self.state, self.mst, k)
        self.cycle_s = MtpCycle(model, self.mtp, self.state, self.mst, k, params=self.params, tok=self.cycle.tok)
        self.stats = dict(steps=0, spec_steps=0, tokens=0, drafted=0, accepted=0)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, eos_ids=(), use_spec: bool = True, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, seed: int = 0):
        from engine.runtime.sampler import sample
        st, mst, k = self.state, self.mst, self.k
        self.params.set(0, temperature, top_k, top_p, min_p, seed)
        self.plain.params.set(0, temperature, top_k, top_p, min_p, seed)
        cyc = self.cycle_s if temperature > 0 else self.cycle
        st.reset(); st.pos = 0
        mst.pos_t.zero_()
        self.mtp.set_prompt_vocab(list(input_ids))
        logits, H = prefill(self.model, torch.tensor([input_ids], device="cuda"), st, return_hidden=True)
        y = int(sample(logits, self.params, st.pos_t)) if temperature > 0 else int(logits.argmax(-1))
        out = [y]
        eos = set(eos_ids)
        if not use_spec:
            while len(out) < max_new_tokens and y not in eos:
                y = int(self.plain.step(torch.tensor([y], device="cuda")))
                out.append(y)
            return out
        # MTP over the prompt: rows (x_{i+1}, h_i), i = 0..T-1, with x_T = y; the last row drafts d1
        toks = torch.tensor(list(input_ids[1:]) + [y], device="cuda")
        cyc.tok.copy_(torch.tensor([[y] + self.mtp.first_drafts(toks, H, mst, k)], device="cuda"))
        while len(out) < max_new_tokens and y not in eos:
            cyc.graph.replay()
            n = int(cyc.n)
            new = cyc.out_tok[0, :n].tolist()
            self.stats["steps"] += 1; self.stats["spec_steps"] += 1
            self.stats["drafted"] += k; self.stats["accepted"] += n - 1
            for t in new:
                out.append(t)
                if t in eos or len(out) >= max_new_tokens:
                    break
            y = out[-1]
        self.stats["tokens"] += len(out)
        return out

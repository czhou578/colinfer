"""MTP drafter (PLAN.md 4.5 item 2): the checkpoint's multi-token-prediction block (one gated full-attention
decoder layer + fc, BF16, excluded from quantization) run k times per step, with its own KV cache.

Row semantics (as vLLM's Qwen3_5MTP / EAGLE proposer): MTP row at position i takes (embed(x_{i+1}),
h_i) where h_i is the target's post-final-norm hidden state at position i, and predicts x_{i+2}:
    x = fc([pre_fc_norm_embedding(embed(x_{i+1})), pre_fc_norm_hidden(h_i)]) -> decoder layer -> mtp.norm
    -> shared lm_head.  Chained steps feed the MTP's own normed output as the next hidden.

MtpCycle captures one CUDA graph per decode cycle (k = 3):
  1. verify [y, d1..dk] on the target (hidden states kept), greedy acceptance, commit n tokens;
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
from engine.model.fast import DecodeGraph, FastQwen35, FastState, KernelAttention, KernelRMSNorm, fast_layer_forward
from engine.model.prefill import prefill, prepare_prefill
from engine.model.qwen35 import DecoderLayer, RMSNorm

MAX_M = 4


class Bf16Linear(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.register_buffer("w", w.contiguous(), persistent=False)
        self.out_features, self.in_features = w.shape

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        if x2.shape[0] <= 16:
            out = torch.empty(x2.shape[0], self.out_features, device=x.device, dtype=torch.bfloat16)
            for i in range(0, x2.shape[0], MAX_M):
                ops().bf16_gemv(x2[i:i + MAX_M], self.w, out[i:i + MAX_M])
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
    """KV cache of the single MTP attention layer (index 0) plus its own device position."""

    def __init__(self, cfg, max_seq_len, device):
        self.cfg, self.max_seq_len, self.kv_fp8, self.pos = cfg, max_seq_len, False, 0
        self.conv, self.rec = {}, {}
        self.k = {0: torch.zeros(1, cfg.num_key_value_heads, max_seq_len, cfg.head_dim, device=device, dtype=torch.bfloat16)}
        self.v = {0: torch.zeros_like(self.k[0])}
        self.pos_t = torch.zeros(1, dtype=torch.int32, device=device)
        self.active = torch.ones(1, dtype=torch.int32, device=device)
        self.arange = torch.arange(16, device=device)


class Mtp(nn.Module):
    def __init__(self, target: FastQwen35, path: str):
        super().__init__()
        cfg = target.cfg
        wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        t = {}
        for name, f in wm.items():
            if name.startswith("mtp."):
                with safe_open(os.path.join(path, f), framework="pt", device="cuda") as sf:
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

    @staticmethod
    def _norm(w, cfg):
        n = RMSNorm(w.numel(), cfg.rms_norm_eps)
        n.weight = nn.Parameter(w, requires_grad=False)
        return KernelRMSNorm(n)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor, st: MtpState):
        """tokens [1, T], hidden [1, T, H] at positions st.pos_t + t (T <= 16): MTP output (normed) [1, T, H]."""
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
            L = int(st.pos_t) + T
            o = flashinfer.single_prefill_with_kv_cache(q[0].transpose(0, 1), st.k[0][0, :, :L], st.v[0][0, :, :L], causal=True,
                                                        kv_layout="HND", sm_scale=a.head_dim ** -0.5)
            gate = qp.view(T, a.num_heads, 2 * a.head_dim)[:, :, a.head_dim:]
            x = a.o_proj((o * torch.sigmoid(gate)).reshape(T, -1), x)
            x = layer.mlp(layer.post_attention_layernorm(x), x)
            g = self.norm(x[-1:])
            st.pos_t += T
        return g


class MtpCycle:
    """CUDA graph of one speculative cycle with the MTP drafter (single slot, k drafts)."""

    def __init__(self, model: FastQwen35, mtp: Mtp, state: FastState, mst: MtpState, k: int = 3):
        self.model, self.mtp, self.state, self.mst, self.k = model, mtp, state, mst, k
        dev = state.pos_t.device
        self.tok = torch.zeros(1, k + 1, dtype=torch.long, device=dev)  # [y, d1..dk]; rewritten by each replay
        self.idx = torch.arange(k + 1, device=dev)
        state.reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.out_tok, self.n = self._body()
        state.reset()

    def _body(self):
        k, m, mtp = self.k, self.model, self.mtp
        tok = self.tok
        logits, H = m.verify(tok, self.state, return_hidden=True)   # [1, k+1, V], [1, k+1, H]
        pred = logits.argmax(-1)                                     # [1, k+1]
        match = (pred[:, :k] == tok[:, 1:]).int()
        n = (1 + match.cumprod(-1).sum(-1)).int()                    # [1] accepted inputs, 1..k+1
        p0 = self.state.pos_t.clone()
        m.commit(self.state, n)
        nl = n.long()
        bonus = pred.gather(1, (nl - 1)[:, None])                    # [1, 1]
        out_tok = torch.cat([tok[:, 1:], bonus], 1)                  # accepted drafts are tok[1..n-1]; the bonus is out_tok[n-1]
        out_tok = torch.where(self.idx[None] == nl[:, None] - 1, bonus, out_tok)
        # MTP catch-up: rows at positions p0 .. p0+k with tokens x_{i+1} (= out_tok[i] for i < n) and true hidden H[i]
        self.mst.pos_t.copy_(p0)
        g = mtp(out_tok, H, self.mst)                                # [1, k+1, H]
        g1 = g.gather(1, (nl - 1)[:, None, None].expand(1, 1, g.shape[-1]))
        d = [mtp.lm_head(g1).float().argmax(-1)]                     # [1, 1]
        gp = g1
        self.mst.pos_t.copy_(p0 + n)
        for _ in range(k - 1):
            gp = mtp(d[-1], gp, self.mst)
            d.append(mtp.lm_head(gp).float().argmax(-1))
            self.mst.pos_t += 1
        nxt = torch.cat([bonus] + d, 1)                              # [1, k+1] = [y', d1'..dk']
        self.tok.copy_(nxt)
        return out_tok, n


class MtpGenerator:
    """Greedy generation with MTP speculation for one slot; output identical to plain greedy decode."""

    def __init__(self, model: FastQwen35, path: str, max_seq_len: int = 32768, k: int = 3):
        prepare_prefill(model)
        self.model, self.k = model, k
        self.mtp = Mtp(model, path)
        self.state = model.new_state(1, max_seq_len)
        self.mst = MtpState(model.cfg, max_seq_len, "cuda")
        self.plain = DecodeGraph(model, self.state)
        self.cycle = MtpCycle(model, self.mtp, self.state, self.mst, k)
        self.stats = dict(steps=0, spec_steps=0, tokens=0, drafted=0, accepted=0)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, eos_ids=(), use_spec: bool = True):
        st, mst, k = self.state, self.mst, self.k
        st.reset(); st.pos = 0
        mst.pos_t.zero_()
        logits, H = prefill(self.model, torch.tensor([input_ids], device="cuda"), st, return_hidden=True)
        y = int(logits.argmax(-1))
        out = [y]
        eos = set(eos_ids)
        if not use_spec:
            while len(out) < max_new_tokens and y not in eos:
                y = int(self.plain.step(torch.tensor([y], device="cuda")))
                out.append(y)
            return out
        # MTP over the prompt: rows (x_{i+1}, h_i), i = 0..T-1, with x_T = y; the last row drafts d1
        toks = torch.tensor(list(input_ids[1:]) + [y], device="cuda")
        g = self.mtp.prefill(toks, H, mst)
        d = [self.mtp.lm_head(g).float().argmax(-1).view(1, 1)]
        gp = g.view(1, 1, -1)
        for _ in range(k - 1):
            gp = self.mtp(d[-1], gp, mst)
            d.append(self.mtp.lm_head(gp).float().argmax(-1).view(1, 1))
            mst.pos_t += 1
        self.cycle.tok.copy_(torch.cat([torch.tensor([[y]], device="cuda")] + d, 1))
        while len(out) < max_new_tokens and y not in eos:
            self.cycle.graph.replay()
            n = int(self.cycle.n)
            new = self.cycle.out_tok[0, :n].tolist()
            self.stats["steps"] += 1; self.stats["spec_steps"] += 1
            self.stats["drafted"] += k; self.stats["accepted"] += n - 1
            for t in new:
                out.append(t)
                if t in eos or len(out) >= max_new_tokens:
                    break
            y = out[-1]
        self.stats["tokens"] += len(out)
        return out

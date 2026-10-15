"""MTP drafter: the multi-token-prediction block of the checkpoint (one gated full-attention decoder layer + fc, BF16 in
the checkpoint). It runs once per draft, with its own fp8 KV cache. Fine-tuned weights can replace it
(tools/train_drafter.py). The --drafter-weights auto option of the server picks
~/.cache/colinfer/drafter/mtp_ft.safetensors.

Row semantics (as the Qwen3_5MTP / EAGLE proposer of vLLM): the MTP row at position i takes (embed(x_{i+1}), h_i). h_i
is the post-final-norm hidden state of the target at position i, and the row predicts x_{i+2}:
    x = fc([pre_fc_norm_embedding(embed(x_{i+1})), pre_fc_norm_hidden(h_i)]) -> decoder layer -> mtp.norm -> draft head.
Chained steps feed the own normed output of the MTP as the next hidden.

The cost per draft step (the engine streams the drafter again at each step, so its bytes are cycle time):
  * Its linears run on NVFP4 weights (round-to-nearest from BF16) on the skinny GEMM; the prompt pass dequantizes
    them to BF16 for each call (csrc/prefill_ops.cu dequant_nvfp4) instead of keeping a BF16 copy.
  * The draft head scores DRAFT_VOCAB frequent tokens (engine/spec/draft_vocab.npy) plus up to PROMPT_SLOTS tokens of
    the current prompts. It does not score the 248k vocabulary.
  * With ~/.cache/colinfer/drafter/draft_head_pca.safetensors (tools/lowrank_draft_head.py), the head is a rank-1024
    approximation. The engine rescores its top LOWRANK_CANDS candidates exactly. This gives the same drafts for ~1/5 of
    the bytes.
None of this changes the outputs, only how many drafts the verify accepts.

MtpCycle captures one CUDA graph per speculative cycle for B slots (k drafts each):
  1. Verify [y, d1..dk] of each slot on the target, and keep the hidden states. Accept the leading drafts that are
     equal to the argmax of the target, or to its position-keyed sample for slots with temperature > 0
     (engine/spec/accept.py). Thus the output is exactly what plain decoding would emit. Cut the accepted length after
     the first accepted stop token, so the state never goes past the end of a reply. Commit n[b] tokens per slot (0
     for inactive slots) on a parallel branch.
  2. MTP catch-up: rows for the newly committed positions, with the true hidden states of the target. The last valid
     row gives the next d1.
  3. k-1 chained MTP steps give d2..dk. They stop early (DRAFT_STOP) when the verify is unlikely to accept the drafts.
     Then the cycle writes the input of the next cycle [y', d1'..dk'] in place.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from engine.kernels import ops
from engine.model.fast import (
    MAX_ROWS,
    FastDecoderLayer,
    FastQwen35,
    FastState,
    KernelAttention,
    KernelRMSNorm,
    LinearGroup,
    Nvfp4Linear,
    RowsLinear,
    capture,
)
from engine.model.prefill import attend_cached, prefill, prepare_prefill
from engine.model.qwen35 import DecoderLayer, RMSNorm, rope_inv_freq
from engine.runtime.sampler import apply_mask
from engine.spec.accept import draw
from engine.weights.direct import read_tensors, tensor_names
from engine.weights.loader import dequant_nvfp4, weight_map
from engine.weights.quantize import nvfp4_global_scale, quantize

DRAFT_DIR = os.path.expanduser("~/.cache/colinfer/drafter")
DRAFT_VOCAB = 65536   # static draft vocabulary (a smaller one loses as much acceptance as it saves time)
PROMPT_SLOTS = 4096   # draft-head rows rewritten per request with prompt tokens outside the static vocabulary
LOWRANK_CANDS = 256   # low-rank head: candidates rescored exactly (64: 0.6% of drafts change, 256: none measured)
DEQUANT_ROWS = 16384  # lm_head rows dequantized to fp32 at a time for the low-rank head (16384 x 5120 x 4 B = 336 MB)
# Draft early exit: once the product of the drafter's probabilities of a cycle's drafts so far is below this for every
# active slot, the remaining draft steps skip their GEMMs (a skipped step costs ~0.35 ms instead of ~1.8 ms; its junk
# drafts are rejected by verify, outputs are unchanged). 40-request mix: 37.2 -> 38.1 tok/s.
DRAFT_STOP = 0.1


class DraftLinear(RowsLinear):
    """A drafter linear on NVFP4 weights (round-to-nearest from BF16): the draft steps stream them (skinny GEMM, up to
    MAX_ROWS rows), and the prompt pass (cuBLAS, many rows) runs on the same weights, dequantized to BF16 for the call
    (0.1-0.2 GB for a moment) rather than on a BF16 copy kept beside them (0.85 GB for the drafter). The residual is
    added after the GEMM, not in its epilogue."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.out_features, self.in_features = w.shape
        wf = w.float()
        gs = nvfp4_global_scale(wf)
        packed, sf = quantize(wf, gs)
        self.low = Nvfp4Linear(packed, sf, gs)

    @classmethod
    def from_parts(cls, packed: torch.Tensor, sf: torch.Tensor, gs: float) -> "DraftLinear":
        """From quantized parts (the drafter's cache, Mtp): what __init__ makes of the BF16 weight."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.out_features, self.in_features = packed.shape[0], packed.shape[1] * 2
        self.low = Nvfp4Linear(packed, sf, gs)
        return self

    def rows(self, x2, r2, out):
        if x2.shape[0] <= MAX_ROWS:
            self.low.rows(x2, None, out)
        else:
            out.copy_(F.linear(x2, ops().dequant_nvfp4(self.low.w, self.low.sf, self.low.gscale)))
        if r2 is not None:
            out += r2


class DraftMLP(nn.Module):
    def __init__(self, g, u, d):
        super().__init__()
        self.gate, self.up, self.down = g, u, d

    def forward(self, x, residual=None):
        return self.down(F.silu(self.gate(x)) * self.up(x), residual)


class MtpState(FastState):
    """The fp8 KV cache of the single MTP attention layer (index 0) for `batch` slots plus their device positions.
    active: shares the target state's mask, so a graph step never writes the MTP cache of an inactive slot. rope_delta:
    shares the target state's (the drafter's rows rotate as the target's at the same positions)."""

    def __init__(self, cfg, max_seq_len, device, batch: int = 1, active: torch.Tensor | None = None, rope_delta: torch.Tensor | None = None):
        super().__init__(dataclasses.replace(cfg, layer_types=["full_attention"], num_hidden_layers=1), batch, max_seq_len, device)
        if active is not None:
            self.active = active
        if rope_delta is not None:
            self.rope_delta = rope_delta

    def slot(self, b: int) -> FastState:
        """Single-slot view with its own all-ones mask, for eager (non-graph) MTP work on one slot."""
        v = self.view(b, b + 1)
        v.active = torch.ones(1, dtype=torch.int32, device=self.pos_t.device)
        return v


class Mtp(nn.Module):
    """The MTP drafter for `target` (checkpoint at `path`). weights: optional safetensors with mtp.* tensors replacing the
    checkpoint's (tools/train_drafter.py). lowrank: the PCA basis of the low-rank draft head (tools/lowrank_draft_head.py),
    used when the file exists. cache: keep the quantized linears and the low-rank head in DRAFT_DIR, keyed by the files
    they come from, so the next start reads them (~0.4 GB) instead of quantizing again (~3 s)."""

    LINEARS = ("fc", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj",
               "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "layers.0.mlp.down_proj")

    def __init__(self, target: FastQwen35, path: str, weights: str | None = None,
                 lowrank: str | None = os.path.join(DRAFT_DIR, "draft_head_pca.safetensors"), cache: bool = True):
        super().__init__()
        cfg = target.cfg
        dev = self.dev = target.embed_tokens.weight.device
        self.weights = weights  # None: the checkpoint's
        files = {name: f for name, f in weight_map(path).items() if name.startswith("mtp.")}
        lowrank = lowrank if lowrank and os.path.exists(lowrank) else None
        self.cache_file = self._cache_path(path, files, weights, lowrank) if cache else None
        q = read_tensors(self.cache_file, device=dev) if self.cache_file and os.path.exists(self.cache_file) else None
        self.cached = q is not None
        big = {n + ".weight" for n in self.LINEARS} if self.cached else set()  # quantized in the cache: not read as BF16
        t = {}
        if weights:
            for name, w in read_tensors(weights, [n for n in tensor_names(weights) if n[4:] not in big], dev).items():
                if name not in files:
                    raise ValueError(f"{weights}: {name} is not a tensor of the checkpoint's MTP block (tools/train_drafter.py writes them)")
                t[name[4:]] = w.to(torch.bfloat16)
        rest = {}  # the checkpoint's tensors that `weights` did not replace, read once per file
        for name, f in files.items():
            if name[4:] not in t and name[4:] not in big:
                rest.setdefault(f, []).append(name)
        for f, names in rest.items():
            for name, w in read_tensors(os.path.join(path, f), names, dev).items():
                t[name[4:]] = w.to(torch.bfloat16)
        made: dict[str, torch.Tensor] = {}  # what a new cache gets

        def lin(name):
            if q is not None:
                return DraftLinear.from_parts(q[name + ".w"], q[name + ".sf"], float(q[name + ".gs"]))
            d = DraftLinear(t.pop(name + ".weight"))
            made.update({name + ".w": d.low.w, name + ".sf": d.low.sf, name + ".gs": torch.tensor(d.low.gscale, dtype=torch.float64)})
            return d
        P = "layers.0."
        self.fc = lin("fc")
        self.pre_e = self._norm(t["pre_fc_norm_embedding.weight"], cfg)
        self.pre_h = self._norm(t["pre_fc_norm_hidden.weight"], cfg)
        self.norm = self._norm(t["norm.weight"], cfg)
        with torch.device("meta"):
            layer = DecoderLayer(dataclasses.replace(cfg, layer_types=["full_attention"], num_hidden_layers=1), 0)
        layer.input_layernorm = self._norm(t[P + "input_layernorm.weight"], cfg)
        layer.post_attention_layernorm = self._norm(t[P + "post_attention_layernorm.weight"], cfg)
        a = layer.self_attn
        a.q_proj, a.k_proj, a.v_proj, a.o_proj = (lin(P + f"self_attn.{n}_proj") for n in "qkvo")
        a.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps).to(dev)
        a.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps).to(dev)
        with torch.no_grad():
            a.q_norm.weight = nn.Parameter(t[P + "self_attn.q_norm.weight"], requires_grad=False)
            a.k_norm.weight = nn.Parameter(t[P + "self_attn.k_norm.weight"], requires_grad=False)
        # q / k / v separate: each keeps its own NVFP4 scales
        KernelAttention.adopt(a, LinearGroup([a.q_proj, a.k_proj, a.v_proj]), rope_inv_freq(cfg, dev), cfg.mrope_section[1:])
        layer.mlp = DraftMLP(lin(P + "mlp.gate_proj"), lin(P + "mlp.up_proj"), lin(P + "mlp.down_proj"))
        layer.__class__ = FastDecoderLayer
        self.layer = layer
        self.embed, self.lm_head, self.cfg = target.embed_tokens, target.lm_head, cfg
        # draft head: the static frequent tokens, then PROMPT_SLOTS rows rewritten per request (set_prompt_vocab)
        ids = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab.npy"))[:DRAFT_VOCAB]
        self.n_static = len(ids)
        self.static = np.zeros(cfg.vocab_size, dtype=bool)  # token id -> in the static set
        self.static[ids] = True
        allids = np.concatenate([ids, np.full(PROMPT_SLOTS, ids[0], dtype=ids.dtype)]).astype(np.int64)
        self.vocab_ids = torch.tensor(allids, device=dev)
        self._lm_draft = None  # the draft vocabulary's rows of lm_head (200 MB), only for the full head (lm_draft)
        self.lr_A = self.lr_B = None
        if lowrank:
            if q is not None:
                self._set_lowrank(q["lr_A.w"], q["lr_A.sf"], float(q["lr_A.gs"]), q["lr_full.w"], q["lr_full.sf"], float(q["lr_full.gs"]))
            else:
                self._init_lowrank(lowrank)
                made.update({"lr_A.w": self.lr_A.w, "lr_A.sf": self.lr_A.sf, "lr_A.gs": torch.tensor(self.lr_A.gscale, dtype=torch.float64),
                             "lr_full.w": self.lr_full[0], "lr_full.sf": self.lr_full[1],
                             "lr_full.gs": torch.tensor(self.lr_B.gscale, dtype=torch.float64)})
        if self.cache_file and q is None:
            try:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                host = {}  # through pinned memory: a pageable copy out of the GPU runs at ~0.1 GB/s on this machine
                for k, v in made.items():
                    v = v.detach().contiguous()
                    pin = torch.empty(v.numel() * v.element_size(), dtype=torch.uint8, pin_memory=True)
                    pin.copy_(v.view(-1).view(torch.uint8))
                    host[k] = pin.view(v.dtype).view(v.shape)
                torch.cuda.synchronize()
                tmp = self.cache_file + ".tmp"
                save_file(host, tmp)
                os.replace(tmp, self.cache_file)
            except OSError as e:
                print(f"[mtp] drafter cache not written ({e})", flush=True)

    @staticmethod
    def _cache_path(path: str, files: dict, weights: str | None, lowrank: str | None) -> str:
        """The cache of the quantized drafter: named by the files the quantization depends on (the shards that hold
        mtp.*, the fine-tuned weights, the low-rank basis), their sizes and modification times."""
        def fid(f):
            st = os.stat(f)
            return [os.path.basename(f), st.st_size, int(st.st_mtime)]
        key = {"v": 2, "checkpoint": os.path.basename(path), "shards": [fid(os.path.join(path, f)) for f in sorted(set(files.values()))],
               "weights": fid(weights) if weights else None, "lowrank": fid(lowrank) if lowrank else None}
        h = hashlib.blake2b(json.dumps(key, sort_keys=True).encode(), digest_size=8).hexdigest()
        return os.path.join(DRAFT_DIR, f"quant-{h}.safetensors")

    def _set_lowrank(self, wa, sa, ga, wb, sb, gb):
        self.lr_A = Nvfp4Linear(wa, sa, ga)
        self.lr_full = (wb, sb)                                                   # [V_target, r]
        self.lr_B = Nvfp4Linear(wb[self.vocab_ids].contiguous(), sb[self.vocab_ids].contiguous(), gb, out_fp32=True)

    def _init_lowrank(self, file: str):
        """Low-rank draft head: with U [H, r] the top-r principal directions of the drafter's outputs, the draft logits are
        approximated by (g U)(W U)^T (two NVFP4 GEMMs streaming ~1/5 of the head's bytes) and the top LOWRANK_CANDS
        candidates are rescored exactly against the real NVFP4 rows. W U is kept for the whole target vocabulary (143 MB),
        so a request's prompt rows are a gather."""
        with safe_open(file, framework="pt", device=str(self.dev)) as sf:
            U = sf.get_tensor("U").float()                                        # [H, r]
        lm = self.lm_head
        gs = torch.tensor(lm.gscale)
        WU = torch.empty(lm.w.shape[0], U.shape[1], device=self.dev)
        for i in range(0, lm.w.shape[0], DEQUANT_ROWS):
            WU[i:i + DEQUANT_ROWS] = dequant_nvfp4(lm.w[i:i + DEQUANT_ROWS], lm.sf[i:i + DEQUANT_ROWS], gs, out_dtype=torch.float32) @ U
        gb = nvfp4_global_scale(WU)
        wb, sb = quantize(WU, gb)
        del WU
        ga = nvfp4_global_scale(U)
        wa, sa = quantize(U.T.contiguous(), ga)
        self._set_lowrank(wa, sa, ga, wb, sb, gb)

    def set_prompt_vocab(self, prompt):
        """Fill the per-request draft-head rows with prompt tokens missing from the static set, in order of first
        occurrence. prompt: token ids (a list or an int64 array)."""
        slots = self.vocab_ids.numel() - self.n_static
        p = np.asarray(prompt, dtype=np.int64)
        u, first = np.unique(p[~self.static[p]], return_index=True)
        extra = u[np.argsort(first)][:slots].tolist()
        extra += [int(self.vocab_ids[0])] * (slots - len(extra))
        ids = torch.tensor(extra, dtype=torch.long, device=self.dev)
        self.vocab_ids[self.n_static:] = ids
        if self._lm_draft is not None:
            self._lm_draft.w[self.n_static:] = self.lm_head.w[ids]
            self._lm_draft.sf[self.n_static:] = self.lm_head.sf[ids]
        if self.lr_B is not None:
            self.lr_B.w[self.n_static:] = self.lr_full[0][ids]
            self.lr_B.sf[self.n_static:] = self.lr_full[1][ids]

    @property
    def lm_draft(self) -> Nvfp4Linear:
        """The full draft head: lm_head's rows of the draft vocabulary (static, then the prompt's). The low-rank head
        rescores its candidates against lm_head itself, so this copy exists only without it (and for the tools)."""
        if self._lm_draft is None:
            lm = self.lm_head
            self._lm_draft = Nvfp4Linear(lm.w[self.vocab_ids].contiguous(), lm.sf[self.vocab_ids].contiguous(), lm.gscale, out_fp32=True)
        return self._lm_draft

    def draft(self, g: torch.Tensor):
        """Greedy drafts from MTP outputs g [..., H] -> (token ids [...], the drafter's probability of each: softmax over
        the draft vocabulary; with the low-rank head, the exact logit over the approximation's logsumexp)."""
        if self.lr_B is not None:  # low-rank scores, the top candidates rescored exactly
            g2 = g.reshape(-1, g.shape[-1]).contiguous()
            ap = self.lr_B(self.lr_A(g2))                                           # [N, V] fp32
            cand = ap.topk(LOWRANK_CANDS, -1).indices                               # [N, C] into the draft vocabulary
            lm = self.lm_head  # the candidates' rows of lm_head itself: the bytes lm_draft would hold
            ex = ops().rescore_nvfp4(g2, lm.w, lm.sf, lm.gscale, self.vocab_ids[cand])
            best = ex.argmax(-1, keepdim=True)
            ids = self.vocab_ids[cand.gather(-1, best)].view(g.shape[:-1])
            p = torch.exp(ex.gather(-1, best) - torch.logsumexp(ap, -1, keepdim=True)).clamp_max(1.0)
            return ids, p.view(g.shape[:-1])
        lg = self.lm_draft(g)
        return self.vocab_ids[lg.argmax(-1)], torch.exp(lg.amax(-1) - torch.logsumexp(lg, -1))

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
    def prefill(self, tokens: torch.Tensor, hidden: torch.Tensor, st: MtpState, chunk: int = 2048,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        """MTP rows for a prompt (tokens [T], hidden [T, H]) from position st.pos_t of a single-slot view, writing the
        drafter's KV; returns the normed output of the last row [1, H]. Attention as in the target's prefill. rope_pos:
        the rows' (t, h, w) rotary positions [T, 3] (images), else their cache position + st.rope_delta."""
        a, layer = self.layer.self_attn, self.layer
        g = None
        for c0 in range(0, tokens.shape[0], chunk):
            tk, hd = tokens[c0:c0 + chunk], hidden[c0:c0 + chunk]
            T = tk.shape[0]
            x = self.fc(torch.cat([self.pre_e(self.embed(tk)), self.pre_h(hd)], -1))
            h = layer.input_layernorm(x)
            qp, kp, vp = a.q_proj(h), a.k_proj(h), a.v_proj(h)
            q = torch.empty(1, a.num_heads, T, a.head_dim, device=x.device, dtype=torch.bfloat16)
            a.prologue(qp, kp, vp, st, st.k[0], st.v[0], q, rope_pos=rope_pos[c0:c0 + T] if rope_pos is not None else None)
            o = attend_cached(q, st.k[0], st.v[0], int(st.pos_t), a.head_dim ** -0.5).view(T, a.num_heads, a.head_dim)
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
        d = [self.draft(g)[0].view(1, 1)]
        gp = g.view(1, 1, -1)
        for _ in range(k - 1):
            gp = self(d[-1], gp, st)
            d.append(self.draft(gp)[0].view(1, 1))
            st.pos_t += 1
        return torch.cat(d, 1)[0].tolist()


class MtpCycle:
    """CUDA graph of one speculative cycle with the MTP drafter for B slots (k drafts each).

    tok [B, k+1] (the next cycle's input [y, d1..dk] per slot), stop_ids [B, S] (int64, -1 = unused) and params may be
    views into buffers shared by the graphs of every draft length, so a slot keeps its pending input when k changes.
    params: SamplerParams -> sampled cycle (slots with temperature > 0 accept drafts that equal the target's
    position-keyed sample, so the output is exactly plain sampling; greedy for the rest; engine/spec/accept.py);
    None -> greedy cycle. pool: a graph memory pool shared with the other cycles (fast.py capture: the outputs are
    valid until the next replay). drafts: MTP drafts made for the next cycle (default k; fewer when k is a suffix-match draft
    length, engine/spec/suffix.py, whose drafts come from the host). mask (optional, int32 [B, k+1, words]): the token
    bitmask of each verify row (constrained decoding, engine/runtime/grammar.py); the target's token at a row is chosen
    among the masked logits, so a draft outside its row's mask is never accepted.
    After replay: out_tok [B, k+1] (first n[b] valid), n [B] int32, logits [B, k+1, V] fp32 (verify rows, raw),
    H [B, k+1, hidden] (target post-norm hidden of the verify rows)."""

    def __init__(self, model: FastQwen35, mtp: Mtp, state: FastState, mst: MtpState, k: int = 3, params=None,
                 tok: torch.Tensor | None = None, stop_ids: torch.Tensor | None = None, drafts: int | None = None,
                 mask: torch.Tensor | None = None, pool=None):
        self.model, self.mtp, self.state, self.mst, self.k, self.params, self.mask = model, mtp, state, mst, k, params, mask
        self.drafts = drafts or k
        dev = state.pos_t.device
        B = state.pos_t.shape[0]
        self.tok = tok if tok is not None else torch.zeros(B, k + 1, dtype=torch.long, device=dev)
        self.stop_ids = stop_ids if stop_ids is not None else torch.full((B, 1), -1, dtype=torch.long, device=dev)
        assert self.tok.shape == (B, k + 1) and self.stop_ids.shape[0] == B
        self.idx = torch.arange(k + 1, device=dev)
        self._side = torch.cuda.Stream()  # the GDN commit's branch
        self.skip = torch.zeros(1, dtype=torch.int32, device=dev)  # draft early exit flag (skinny_skip)
        state.reset(kv=False)  # the capture's warm-up runs write the first rows; the rest of the KV is never read
        self.graph, (self.out_tok, self.n, self.logits, self.H) = capture(self._body, pool)
        state.reset(kv=False)
        mst.pos_t.zero_()

    def _body(self):
        k, m, mtp, st = self.k, self.model, self.mtp, self.state
        tok = self.tok
        B = tok.shape[0]
        logits, H = m.verify(tok, st)                                # [B, k+1, V], [B, k+1, H]
        sel = apply_mask(logits, self.mask) if self.mask is not None else logits  # the logits stay raw for logprobs
        if self.params is None:
            pred = sel.argmax(-1)                                    # [B, k+1]
        else:  # the target's own sample at every row, keyed by position (engine/spec/accept.py): output = plain sampling
            pred = draw(sel, self.params, st.pos_t + 1)              # row i predicts position pos + i + 1
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
        # the drafting below never reads the GDN state: commit it on a parallel graph branch
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            m.commit(st, n)
        nl = n.long()
        out_tok = torch.cat([d, bonus[:, None]], 1)                  # accepted drafts are d[:n-1]; the bonus is out_tok[n-1]
        out_tok = torch.where(self.idx[None] == nl[:, None] - 1, bonus[:, None], out_tok)
        # MTP catch-up: rows at positions p0 .. p0+k with tokens x_{i+1} (= out_tok[i] for i < n) and true hidden H[i]
        self.mst.pos_t.copy_(p0)
        g = mtp(out_tok, H, self.mst)                                # [B, k+1, H]
        last = (nl - 1).clamp_min(0)
        gp = g.gather(1, last[:, None, None].expand(B, 1, g.shape[-1]))
        d, cp = mtp.draft(gp)                                        # [B, 1], cp: running product of draft probabilities
        dr = [d]
        self.mst.pos_t.copy_(p0 + n)
        self.skip.zero_()
        for _ in range(self.drafts - 1):
            # every slot's chain is unlikely to survive this far: the remaining steps skip their GEMMs (zero outputs)
            self.skip.copy_(torch.maximum(self.skip, ((cp[:, 0] < DRAFT_STOP) | (st.active == 0)).all().int().view(1)))
            ops().skinny_skip(self.skip)
            try:
                gp = mtp(dr[-1], gp, self.mst)
                d, p = mtp.draft(gp)
            finally:
                ops().skinny_skip(None)
            cp = cp * p
            dr.append(d)
            self.mst.pos_t += 1
        nxt = torch.cat([bonus[:, None]] + dr, 1)                    # [B, drafts+1] = [y', d1'..']
        nd = self.drafts + 1
        self.tok[:, :nd].copy_(torch.where(st.active[:, None] > 0, nxt, tok[:, :nd]))
        torch.cuda.current_stream().wait_stream(self._side)
        return out_tok, n, logits, H


class MtpGenerator:
    """Single-slot greedy generation with MTP speculation at a fixed draft length k: the drafter tools' driver
    (tools/eval_drafter.py, tools/lowrank_draft_head.py). Serving uses engine/runtime/scheduler.py, whose outputs with and
    without speculation tests/golden.py checks."""

    def __init__(self, model: FastQwen35, path: str, max_seq_len: int = 32768, k: int = 3, weights: str | None = None, **mtp_kw):
        """mtp_kw: further Mtp options (lowrank=None: the full draft head)."""
        prepare_prefill(model)
        self.model, self.k = model, k
        self.mtp = Mtp(model, path, weights=weights, **mtp_kw)
        self.dev = self.mtp.dev
        self.state = model.new_state(1, max_seq_len)
        self.mst = MtpState(model.cfg, max_seq_len, self.dev, active=self.state.active, rope_delta=self.state.rope_delta)
        self.cycle = MtpCycle(model, self.mtp, self.state, self.mst, k)
        self.stats = dict(steps=0, tokens=0, drafted=0, accepted=0)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, eos_ids=()):
        st, mst, k, cyc = self.state, self.mst, self.k, self.cycle
        st.reset()
        st.pos = 0
        mst.pos_t.zero_()
        self.mtp.set_prompt_vocab(list(input_ids))
        logits, H = prefill(self.model, torch.tensor([input_ids], device=self.dev), st, return_hidden=True)
        y = int(logits.argmax(-1))
        out = [y]
        eos = set(eos_ids)
        # MTP over the prompt: rows (x_{i+1}, h_i), i = 0..T-1, with x_T = y; the last row drafts d1
        toks = torch.tensor(list(input_ids[1:]) + [y], device=self.dev)
        cyc.tok.copy_(torch.tensor([[y] + self.mtp.first_drafts(toks, H, mst, k)], device=self.dev))
        while len(out) < max_new_tokens and y not in eos:
            cyc.graph.replay()
            n = int(cyc.n)
            self.stats["steps"] += 1
            self.stats["drafted"] += k
            self.stats["accepted"] += n - 1
            for t in cyc.out_tok[0, :n].tolist():
                out.append(t)
                if t in eos or len(out) >= max_new_tokens:
                    break
            y = out[-1]
        self.stats["tokens"] += len(out)
        return out

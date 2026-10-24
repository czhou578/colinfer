"""MTP drafter of Qwen3.8-Flash-Next (qwen4_exp) and exact speculation on top of the decode model FastQwen4Exp
(engine/model/fast_qwen4exp.py). This module is the qwen4_exp counterpart of engine/spec/mtp.py (the 27B's drafter),
with the same entry points: Mtp4 (the drafter), Mtp4State (its state), Mtp4Cycle (one speculative cycle as a CUDA graph).

The wiring, confirmed on 2026-10-24 in vLLM's source (vllm-project/vllm, main):
vllm/models/qwen4_exp/nvidia/mtp.py (Qwen4ExpMultiTokenPredictor.forward), nvidia/model.py (Qwen4ExpDecoderLayer.forward,
Qwen4ExpModel.forward), nvidia/hyperconnection.py (GatedResidual.combine_and_mix), nvidia/ops/hc.py (hc_combine_norm),
common/hyperconnection.py (GroupedGemmaRMSNorm) and vllm/v1/spec_decode/llm_base_proposer.py (the positions):

1. The input of a draft row is the 4-stream hidden h4 [T, 4 x 2560] before the final mixer: the target's on the first
   step (Qwen4ExpModel keeps multi_hidden of final_mixer.combine_and_mix in _mtp_hidden_buffer), the drafter's own
   pre-mixer stream on the chained steps ("scheme A": the drafter returns the mixer output for the head and the
   multi stream for the next step).
2. e = fc_embedding(pre_fc_norm_embedding(embed(token))) [T, 2560]. h = pre_fc_norm_hidden(h4) is ONE GemmaRMSNorm over
   the whole 10240-wide vector (not per stream), then fc_hidden runs on each of the 4 streams with the one shared
   [2560, 2560] weight. GemmaRMSNorm: bf16(x / rms(x) * (1 + w)), fp32 math.
3. The decoder layer gets prev_block_output = e and prev_injection = None. combine_and_mix then runs hc_combine_norm
   with no injection: x4_s = bf16(h_s + e) for each of the 4 streams (a plain add with unit weight), then the grouped
   norm of the attention hyper-connection. So the embedding enters every stream as a plain add.
4. The layer: QSA attention with its own indexer and KV, the MoE (512 FP8 experts with 128 x 128 block scales, the
   shared expert with its sigmoid gate) and the two gated residuals, exactly as a target layer. No PLE.
5. The final mixer: GatedResidual(use_combine=False) with hc_per_branch_norm=True. That is a norm per 2560-wide stream
   with a [10240] affine weight, as the target's mixer and hc_kernels.KernelGatedResidual. The checkpoint has no
   inject weight for it. Its output goes to the target's lm_head (there is no mtp lm_head and no mtp.norm).
6. Positions (the EAGLE / MTP proposer): the row that takes (embed(x_{i+1}), h4_i) sits at position i, as in the 27B's
   drafter. A chained step is at the next position. The rotary position of a row is its cache position + rope_delta.
7. The indexer: vLLM selects the blocks at draft step 0 and reuses those indices on the later steps (set_skip_topk).
   This module selects on every row of every step with the target's kernels (qsa_score, qsa_topk). Why: a row's
   selection then depends only on the row (the row invariance of the target), the state holds no index buffer to
   snapshot or restore, and a selection costs about 0.1 ms at 262k tokens. Exactness does not depend on it: the
   accept rule emits only the target's own tokens, whatever the drafts are. Below 2052 tokens every block is
   selected anyway, so both rules give the same rows there.

The drafter's arithmetic is the target's, on the same kernels: the BF16 skinny GEMM (fc_embedding, fc_hidden, the
attention projections, the router, the shared expert, the draft head), csrc/hc.cu, csrc/qsa.cu and csrc/moe.cu. So a
row's bits do not depend on the number of rows in the launch (tests/test_mtp4.py checks T = 1 against T = 8). The
routed experts are FP8 in the checkpoint. The loader dequantizes them (128 x 128 block scales) and quantizes them to
NVFP4 in KernelMoE's stacked layout (one global scale per expert for gate | up, one for down). The result is cached
under ~/.cache/colinfer/drafter/ (about 1.4 GB, keyed by the files that hold the mtp.* tensors), so the next load
reads it instead of quantizing again.

The draft head scores DRAFT_VOCAB frequent tokens (engine/spec/draft_vocab.npy; the tokenizer is the 27B's, byte for
byte) plus PROMPT_SLOTS tokens of the current prompt (set_prompt_vocab): a BF16 copy of those lm_head rows
(69,632 x 2560, 0.36 GB) on the skinny GEMM. The low-rank head of the 27B is not ported yet.

Mtp4Cycle is engine/spec/mtp.py's MtpCycle on this family: verify [y, d1..dk] (FastQwen4Exp.verify, its n-gram rows
from feed_buffer(B, k + 1), which the host fills before each replay), accept (engine/spec/accept.py: greedy, or the
position-keyed sample), commit (FastQwen4Exp.commit, on a parallel graph branch), then the drafter: one catch-up row
per verified position with the target's hidden4, and k - 1 chained steps on the drafter's own stream. The draft
early exit (DRAFT_STOP, ops().skinny_skip) is ported: a skipped step skips the skinny GEMMs and the expert kernels.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from engine.kernels import ops
from engine.model import hc_kernels
from engine.model.fast import MAX_ROWS, capture
from engine.model.fast_qwen4exp import Bf16Linear, FastDecoderLayer4, FastQwen4Exp, FastState4, KernelQSAAttention, hc_rows
from engine.model.moe_kernels import KernelMoE
from engine.model.qwen4exp import IndexedAttention, Qwen4ExpConfig
from engine.model.qwen35 import rope_inv_freq
from engine.runtime.sampler import apply_mask
from engine.spec.accept import draw
from engine.spec.mtp import DRAFT_DIR, DRAFT_STOP, DRAFT_VOCAB, PROMPT_SLOTS
from engine.weights.direct import read_tensors, tensor_names
from engine.weights.loader import weight_map
from engine.weights.quantize import E2M1_VALUES, E4M3_MAX, quantize

FP8_BLOCK = 128      # the block of the checkpoint's FP8 expert scales (weight_scale_inv)
REQUANT_BATCH = 32   # experts dequantized to fp32 at a time (32 x 1280 x 2560 x 4 B = 0.42 GB)
PREFILL_CHUNK = 2048  # rows per chunk of Mtp4.prefill


def drafter_cfg(cfg: Qwen4ExpConfig) -> Qwen4ExpConfig:
    """The config of the drafter's one layer: one full-attention layer, no PLE."""
    return dataclasses.replace(cfg, layer_types=["full_attention"], num_hidden_layers=1, ple_layer_ids=[])


class Mtp4State(FastState4):
    """The drafter's state for `batch` slots: the FastState4 of one attention layer (index 0): k, v fp8 [B, 2, L, 256],
    ik bf16 [B, L, 128], ikb bf16 [B, L / 4, 128], rope_pos int32 [B, L, 3], pos_t. No GDN and no PLE state, so
    snap_tensors() is empty and kv_tensors() lists k, v, ik, ikb, rope_pos. active and rope_delta may share the
    target state's tensors (a replay then never writes the drafter rows of an inactive slot, and the drafter's rows
    rotate as the target's)."""

    def __init__(self, cfg: Qwen4ExpConfig, max_seq_len: int, device, batch: int = 1, active: torch.Tensor | None = None,
                 rope_delta: torch.Tensor | None = None):
        super().__init__(drafter_cfg(cfg), batch, max_seq_len, device)
        if active is not None:
            self.active = active
        if rope_delta is not None:
            self.rope_delta = rope_delta

    def slot(self, b: int) -> "Mtp4State":
        """A single-slot view with its own all-ones mask, for eager drafter work on one slot (prefill, first_drafts)."""
        v = self.view(b, b + 1)
        v.active = torch.ones(1, dtype=torch.int32, device=self.pos_t.device)
        return v


# ------------------------------------------------------------------------------------------------------------ experts
def _dequant_blocks(w8: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
    """FP8 weights e4m3 [b, N, K] with block scales si [b, ceil(N / 128), ceil(K / 128)] -> fp32 [b, N, K]
    (engine.weights.loader.dequant_fp8_block for a batch of experts)."""
    _, N, K = w8.shape
    s = si.float().repeat_interleave(FP8_BLOCK, 1)[:, :N].repeat_interleave(FP8_BLOCK, 2)[:, :, :K]
    return w8.float() * s


def requant_experts(g8, g_si, u8, u_si, d8, d_si, batch: int = REQUANT_BATCH) -> dict[str, torch.Tensor]:
    """The FP8 block-scaled experts of the drafter as NVFP4 in KernelMoE's stacked layout.

    g8, u8 e4m3 [E, I, K] (gate, up), d8 e4m3 [E, K, I] (down); *_si their 128 x 128 block scales (bf16). Returns
    w_gu uint8 [E, 2I, K/2] (the gate rows, then the up rows), sf_gu e4m3 [E, 2I, K/16], w_dn uint8 [E, K, I/2],
    sf_dn e4m3 [E, K, I/16] (row-major block scales) and gs f32 [E, 3]: one global scale per expert for gate | up
    (amax / (448 x 6)) and one for down. The blocks scales come from engine.weights.quantize.quantize."""
    E, I, K = g8.shape
    dev = g8.device
    out = dict(w_gu=torch.empty(E, 2 * I, K // 2, dtype=torch.uint8, device=dev),
               sf_gu=torch.empty(E, 2 * I, K // 16, dtype=torch.float8_e4m3fn, device=dev),
               w_dn=torch.empty(E, K, I // 2, dtype=torch.uint8, device=dev),
               sf_dn=torch.empty(E, K, I // 16, dtype=torch.float8_e4m3fn, device=dev),
               gs=torch.empty(E, 3, dtype=torch.float32, device=dev))
    top = E4M3_MAX * E2M1_VALUES[-1]
    for e0 in range(0, E, batch):
        s = slice(e0, min(e0 + batch, E))
        n = s.stop - s.start
        for w, name, gcol in ((torch.cat([_dequant_blocks(g8[s], g_si[s]), _dequant_blocks(u8[s], u_si[s])], 1), "gu", (0, 1)),
                              (_dequant_blocks(d8[s], d_si[s]), "dn", (2,))):
            gs = (w.abs().amax((1, 2)) / top).clamp_min(1e-30)  # [n]; an all-zero expert gets any scale
            rows, cols = w.shape[1:]
            packed, sf = quantize((w / gs[:, None, None]).view(-1, cols), 1.0)  # = quantize(w_e, gs_e) for each expert
            out["w_" + name][s] = packed.view(n, rows, cols // 2)
            out["sf_" + name][s] = sf.view(n, rows, cols // 16)
            for c in gcol:
                out["gs"][s, c] = gs
            del w, packed, sf
    return out


def drafter_in_scale(model: FastQwen4Exp, E: int) -> torch.Tensor:
    """The input scales of the requantized experts [E, 3] (only the prefill path quantizes the activations; the decode
    path multiplies bf16 activations): the largest input scale of the target's last MoE layer per projection, gate
    and up equal (the grouped gate | up GEMM takes one alpha per expert)."""
    s = model.layers[-1].mlp.in_scale
    gu, dn = float(s[:, :2].max()), float(s[:, 2].max())
    return torch.tensor([gu, gu, dn], dtype=torch.float32, device=s.device).expand(E, 3).contiguous()


# ------------------------------------------------------------------------------------------------------------ drafter
def _nn_param(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t, requires_grad=False)


class _ZcNorm(nn.Module):
    """Zero-centered RMSNorm (GemmaRMSNorm, bf16(x / rms(x) * (1 + w)) in fp32) over the whole last dim, on
    ops().rmsnorm: one block per row, so a row's bits do not depend on the number of rows."""

    def __init__(self, w: torch.Tensor, eps: float):
        super().__init__()
        self.weight, self.eps = w.contiguous(), eps

    def forward(self, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty_like(x2)
        ops().rmsnorm(x2, self.weight, self.eps, out)
        return out.view(x.shape)


class Mtp4(nn.Module):
    """The MTP drafter of a qwen4_exp checkpoint at `path` for the decode model `target`.

    weights: optional safetensors with mtp.* tensors (BF16, not the experts) that replace the checkpoint's. lowrank:
    accepted for the 27B's interface and not used yet (the full draft head runs). cache: keep the NVFP4 experts in
    DRAFT_DIR. tensors / experts: build from tensors instead of a checkpoint (tests): tensors {name without "mtp.":
    bf16 tensor}, experts {w_gu, sf_gu, w_dn, sf_dn, gs} (requant_experts' layout) and optionally in_scale [E, 3].

    forward(tokens [B, T], hidden4 [B, T, 10240], st) -> (g [B, T, 2560] the mixer output, x4 [B, T, 10240] the
    pre-mixer stream) at positions st.pos_t + t; it writes the rows' KV, raw indexer keys, block keys and rotary
    positions and does not move st.pos_t. draft(g) -> (token ids, the drafter's probability of each)."""

    def __init__(self, target: FastQwen4Exp, path: str | None, weights: str | None = None, lowrank: str | None = None, cache: bool = True,
                 tensors: dict | None = None, experts: dict | None = None, log=print):
        super().__init__()
        t0 = time.time()
        self.cfg = target.cfg
        dev = self.dev = target.embed_tokens.weight.device
        self.__dict__["target"] = target  # not a submodule: the drafter does not own the target
        self.weights, self.lowrank = weights, None
        self.cached = False
        self.cache_file = None
        if tensors is None:
            files = {n: f for n, f in weight_map(path).items() if n.startswith("mtp.")}
            if not files:
                raise ValueError(f"{path}: the checkpoint has no mtp.* tensors")
            tensors, experts = self._read(path, files, weights, cache, dev, log)
        if experts.get("in_scale") is None:
            experts = dict(experts, in_scale=drafter_in_scale(target, experts["w_gu"].shape[0]))
        self._build(tensors, experts)
        self._init_head()
        log(f"[mtp4] drafter ready in {time.time() - t0:.1f}s ({'experts from the cache' if self.cached else 'experts requantized'}), "
            f"{self.nbytes() / 1e9:.2f} GB")

    # -------------------------------------------------------------------------------------------- loading
    @staticmethod
    def _cache_path(path: str, files: dict) -> str:
        """The cache of the requantized experts: named by the files that hold the mtp.* experts (size, modification time)."""
        def fid(f):
            st = os.stat(f)
            return [os.path.basename(f), st.st_size, int(st.st_mtime)]
        ex = sorted({f for n, f in files.items() if ".mlp.experts." in n})
        key = {"v": 1, "kind": "qwen4exp-mtp-experts-nvfp4", "checkpoint": os.path.basename(os.path.normpath(path)),
               "shards": [fid(os.path.join(path, f)) for f in ex]}
        h = hashlib.blake2b(json.dumps(key, sort_keys=True).encode(), digest_size=8).hexdigest()
        return os.path.join(DRAFT_DIR, f"quant4-{h}.safetensors")

    def _read(self, path, files, weights, cache, dev, log):
        cfg = self.cfg
        E, I, K = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        names = [n for n in files if ".mlp.experts." not in n]
        t = {}
        if weights:
            for name, w in read_tensors(weights, tensor_names(weights), dev).items():
                if name not in files or ".mlp.experts." in name:
                    raise ValueError(f"{weights}: {name} is not a BF16 tensor of the checkpoint's MTP block")
                t[name[4:]] = w.to(torch.bfloat16)
        by_file = {}
        for n in names:
            if n[4:] not in t:
                by_file.setdefault(files[n], []).append(n)
        for f, ns in by_file.items():
            for name, w in read_tensors(os.path.join(path, f), ns, dev).items():
                t[name[4:]] = w.to(torch.bfloat16)
        self.cache_file = self._cache_path(path, files) if cache else None
        if self.cache_file and os.path.exists(self.cache_file):
            self.cached = True
            return t, read_tensors(self.cache_file, device=dev)
        t1 = time.time()
        f8 = dict(device=dev, dtype=torch.float8_e4m3fn)
        bf = dict(device=dev, dtype=torch.bfloat16)
        nb = lambda n: -(-n // FP8_BLOCK)  # noqa: E731
        g8, u8, d8 = torch.empty(E, I, K, **f8), torch.empty(E, I, K, **f8), torch.empty(E, K, I, **f8)
        gs_, us_, ds_ = torch.empty(E, nb(I), nb(K), **bf), torch.empty(E, nb(I), nb(K), **bf), torch.empty(E, nb(K), nb(I), **bf)
        into = {}
        for e in range(E):
            b = f"mtp.layers.0.mlp.experts.{e}."
            into |= {b + "gate_proj.weight": g8[e], b + "gate_proj.weight_scale_inv": gs_[e], b + "up_proj.weight": u8[e],
                     b + "up_proj.weight_scale_inv": us_[e], b + "down_proj.weight": d8[e], b + "down_proj.weight_scale_inv": ds_[e]}
        missing = sorted(set(into) - set(files))
        if missing:
            raise ValueError(f"{path}: the MTP experts are not FP8 block-scaled as expected (missing {missing[:3]})")
        by_file = {}
        for n in into:
            by_file.setdefault(files[n], []).append(n)
        for f, ns in by_file.items():
            read_tensors(os.path.join(path, f), ns, dev, into=into)
        ex = requant_experts(g8, gs_, u8, us_, d8, ds_)
        del g8, u8, d8, gs_, us_, ds_, into
        torch.cuda.empty_cache()
        log(f"[mtp4] {E} FP8 experts read and requantized to NVFP4 in {time.time() - t1:.1f}s")
        if self.cache_file:
            self._save(ex, log)
        return t, ex

    def _save(self, ex: dict, log):
        from safetensors.torch import save_file
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            host = {}  # through pinned memory: a pageable copy out of the GPU runs at about 0.1 GB/s on this machine
            for k, v in ex.items():
                v = v.detach().contiguous()
                pin = torch.empty(v.numel() * v.element_size(), dtype=torch.uint8, pin_memory=True)
                pin.copy_(v.view(-1).view(torch.uint8))
                host[k] = pin.view(v.dtype).view(v.shape)
            torch.cuda.synchronize()
            tmp = self.cache_file + ".tmp"
            save_file(host, tmp)
            os.replace(tmp, self.cache_file)
        except OSError as e:
            log(f"[mtp4] drafter cache not written ({e})")

    def _build(self, t: dict, ex: dict):
        cfg, dev = self.cfg, self.dev
        eps = cfg.rms_norm_eps
        P = "layers.0."
        self.pre_e = _ZcNorm(t.pop("pre_fc_norm_embedding.weight"), eps)
        self.pre_h = _ZcNorm(t.pop("pre_fc_norm_hidden.weight"), eps)  # one norm over the whole 10240-wide vector
        self.fc_e = Bf16Linear(t.pop("fc_embedding.weight"))
        self.fc_h = Bf16Linear(t.pop("fc_hidden.weight"))               # shared by the 4 streams
        with torch.device("meta"):
            attn = IndexedAttention(drafter_cfg(cfg))
        a = P + "self_attn."
        for mod, name in ((attn.q_proj, "q_proj"), (attn.k_proj, "k_proj"), (attn.v_proj, "v_proj"), (attn.o_proj, "o_proj"),
                          (attn.q_norm, "q_norm"), (attn.k_norm, "k_norm"), (attn.indexer.index_qk_proj, "indexer.index_qk_proj"),
                          (attn.indexer.q_layernorm, "indexer.q_layernorm"), (attn.indexer.k_layernorm, "indexer.k_layernorm")):
            mod.weight = _nn_param(t.pop(a + name + ".weight"))
        KernelQSAAttention.adopt(attn, rope_inv_freq(cfg, dev), cfg.mrope_section[1:])

        def gr(prefix, inj=True):
            return hc_kernels.KernelGatedResidual(t.pop(prefix + "hc_norm.weight"), t.pop(prefix + "input_mix_weight_down.weight"),
                                                  t.pop(prefix + "block_inject_weight.weight") if inj else None,
                                                  t.pop(prefix + "input_mix_weight_up.weight"), eps)
        layer = FastDecoderLayer4.__new__(FastDecoderLayer4)
        nn.Module.__init__(layer)
        layer.layer_idx, layer.block_type, layer.ple = 0, "full_attention", None
        layer.self_attn = attn
        layer.attn_hc = gr(P + "attn_hyper_connection.")
        layer.mlp_hc = gr(P + "mlp_hyper_connection.")
        m = P + "mlp."
        layer.mlp = KernelMoE(ex["w_gu"], ex["sf_gu"], ex["w_dn"], ex["sf_dn"], ex["gs"], ex["in_scale"], t.pop(m + "gate.weight"),
                              t.pop(m + "shared_expert_gate.weight"), t.pop(m + "shared_expert.gate_proj.weight"),
                              t.pop(m + "shared_expert.up_proj.weight"), t.pop(m + "shared_expert.down_proj.weight"))
        self.layer = layer
        self.mixer = gr("hyper_connection_mixer.", inj=False)
        self.__dict__["embed"], self.__dict__["lm_head"] = self.target.embed_tokens, self.target.lm_head  # the target's
        if t:
            raise ValueError(f"unexpected MTP tensors: {sorted(t)[:5]}")
        self.requires_grad_(False)

    def _init_head(self):
        """The draft head: the static frequent tokens, then PROMPT_SLOTS rows that set_prompt_vocab rewrites per request.
        A vocabulary that fits the head (the tests' small model) is scored whole."""
        V = self.cfg.vocab_size
        ids = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab.npy"))[:DRAFT_VOCAB].astype(np.int64)
        ids = ids[ids < V]
        if V <= DRAFT_VOCAB + PROMPT_SLOTS:
            ids, slots = np.arange(V, dtype=np.int64), 0
        else:
            slots = PROMPT_SLOTS
        self.n_static = len(ids)
        self.static = np.zeros(V, dtype=bool)  # token id -> in the static set
        self.static[ids] = True
        allids = np.concatenate([ids, np.full(slots, ids[0], dtype=np.int64)])
        self.vocab_ids = torch.tensor(allids, device=self.dev)
        self.head = Bf16Linear(self.lm_head.w[self.vocab_ids].contiguous(), out_fp32=True)

    def nbytes(self) -> int:
        """The drafter's own GPU bytes (the embedding and lm_head are the target's)."""
        mods = [self.pre_e.weight, self.pre_h.weight, self.fc_e.w, self.fc_h.w, self.head.w]
        a = self.layer.self_attn
        mods += [a.qkvi.w, a.o_proj.w, a.q_norm.weight, a.k_norm.weight, a.indexer.q_layernorm.weight, a.indexer.k_layernorm.weight]
        for g in (self.layer.attn_hc, self.layer.mlp_hc, self.mixer):
            mods += [g.norm_w, g.w_di, g.w_up]
        x = self.layer.mlp
        mods += [x.w_gu, x.sf_gu, x.w_dn, x.sf_dn, x.gs, x.in_scale, x.router, x.sh_gate, x.sh_up, x.sh_down]
        return sum(m.numel() * m.element_size() for m in mods)

    # -------------------------------------------------------------------------------------------- head
    def set_prompt_vocab(self, prompt):
        """Fill the per-request draft-head rows with prompt tokens missing from the static set, in order of first
        occurrence. prompt: token ids (a list or an int64 array)."""
        slots = self.vocab_ids.numel() - self.n_static
        if slots == 0:
            return
        p = np.asarray(prompt, dtype=np.int64)
        u, first = np.unique(p[~self.static[p]], return_index=True)
        extra = u[np.argsort(first)][:slots].tolist()
        extra += [int(self.vocab_ids[0])] * (slots - len(extra))
        ids = torch.tensor(extra, dtype=torch.long, device=self.dev)
        self.vocab_ids[self.n_static:] = ids
        self.head.w[self.n_static:] = self.lm_head.w[ids]

    def draft(self, g: torch.Tensor):
        """Greedy drafts from mixer outputs g [..., 2560] -> (token ids [...], the drafter's probability of each: the
        softmax over the draft vocabulary)."""
        lg = self.head(g)
        return self.vocab_ids[lg.argmax(-1)], torch.exp(lg.amax(-1) - torch.logsumexp(lg, -1))

    # -------------------------------------------------------------------------------------------- forward
    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor, st: Mtp4State):
        """tokens [B, T] (B * T <= 16 for a graph), hidden [B, T, 10240] at positions st.pos_t + t -> (g [B, T, 2560],
        x4 [B, T, 10240])."""
        B, T = tokens.shape
        H, S = self.cfg.hidden_size, self.cfg.hc_count
        M = B * T
        e = self.fc_e(self.pre_e(self.embed(tokens))).view(M, 1, H)
        h = self.fc_h(self.pre_h(hidden.reshape(M, S * H)).view(M * S, H)).view(M, S, H)
        x4 = (h + e).view(M, S * H)  # prev_block_output = e with no injection: a plain add to every stream, rounded to bf16
        rope_rows = self.target._rope_rows(st, T)
        x4 = self.layer(x4, st, B, T, rope_rows, None)
        g = hc_rows(self.mixer, x4)
        return g.view(B, T, H), x4.view(B, T, S * H)

    @torch.inference_mode()
    def prefill(self, tokens: torch.Tensor, hidden: torch.Tensor, st: Mtp4State, chunk: int = PREFILL_CHUNK,
                rope_pos: torch.Tensor | None = None):
        """Drafter rows for a prompt: tokens [T] (x_{i+1}), hidden [T, 10240] (the target's hidden4 of positions i) from
        position st.pos_t of a single-slot view (Mtp4State.slot), writing the drafter's KV, raw indexer keys, block keys
        and rotary positions; st.pos_t moves by T. Returns (g [1, 2560], x4 [1, 10240]) of the last row. A chunk of up
        to 16 rows runs the decode kernels (forward); a longer chunk runs the prefill path of
        engine/model/prefill_qwen4exp.py (cuBLAS projections, the sparse prefill attention, the grouped NVFP4 MoE).
        rope_pos: int32 [T, 3] (images), else each row's cache position + st.rope_delta."""
        from engine.model import prefill_qwen4exp as pq
        H, S = self.cfg.hidden_size, self.cfg.hc_count
        eps = self.cfg.rms_norm_eps
        layer = self.layer
        g = x4 = None
        pos = int(st.pos_t[0])
        for c0 in range(0, tokens.shape[0], chunk):
            tk, hd = tokens[c0:c0 + chunk], hidden[c0:c0 + chunk]
            T = tk.shape[0]
            if T <= MAX_ROWS and rope_pos is None:
                g, x4 = self(tk.view(1, T), hd.view(1, T, -1), st)
                g, x4 = g[0, -1:], x4[0, -1:]
            else:
                e = self.pre_e(self.embed(tk)) @ self.fc_e.w.t()
                h = (self.pre_h(hd.contiguous()).view(T * S, H) @ self.fc_h.w.t()).view(T, S, H)
                x = (h + e.view(T, 1, H)).view(T, S * H)
                st.pos = pos
                y, inj = pq.hc_pre(x, layer.attn_hc)
                pq.hc_post(x, pq.qsa_layer(y, layer.self_attn, st, 0, pos, rope_pos[c0:c0 + T] if rope_pos is not None else None, eps), inj)
                y, inj = pq.hc_pre(x, layer.mlp_hc)
                pq.hc_post(x, pq.moe_layer(y, layer.mlp), inj)
                x4 = x[-1:]
                g = self.mixer(x4.contiguous())
            pos += T
            st.pos_t.fill_(pos)
            st.pos = pos
        return g, x4

    @torch.inference_mode()
    def first_drafts(self, tokens: torch.Tensor, hidden: torch.Tensor, st: Mtp4State, k: int) -> list[int]:
        """Drafter rows for (tokens[i], hidden[i]) from position st.pos_t (eager, one slot; prefill), then k - 1 chained
        steps: the k drafts that follow the last token. st: a single-slot view with an all-ones mask (Mtp4State.slot)."""
        g, x4 = self.prefill(tokens, hidden, st)
        d = [self.draft(g)[0].view(1, 1)]
        xp = x4.view(1, 1, -1)
        for _ in range(k - 1):
            gp, xp = self(d[-1], xp, st)
            d.append(self.draft(gp)[0].view(1, 1))
            st.pos_t += 1
        return torch.cat(d, 1)[0].tolist()


# ------------------------------------------------------------------------------------------------------------ cycle
class Mtp4Cycle:
    """CUDA graph of one speculative cycle of FastQwen4Exp with the drafter Mtp4 for B slots (k drafts each). The
    interface is engine/spec/mtp.py's MtpCycle:

    tok [B, k+1] (the next cycle's input [y, d1..dk] per slot), stop_ids [B, S] (int64, -1 = unused), params
    (SamplerParams: the sampled cycle; None: greedy), drafts (the MTP drafts made for the next cycle, default k), mask
    (int32 [B, k+1, words]: the token bitmask of each verify row) and pool (a graph memory pool) as there. The host
    fills model.feed_buffer(B, k+1) with the n-gram rows of tok before each replay (the family's before_replay).
    early_exit: the draft early exit (DRAFT_STOP).
    After replay: out_tok [B, k+1] (first n[b] valid), n [B] int32, logits [B, k+1, V] fp32 (the verify rows, raw), H
    [B, k+1, 10240] (the target's pre-mixer stream of the verify rows: H[b, n[b] - 1] is the drafter's input for the
    slot's last fed position, the scheduler's h_last)."""

    def __init__(self, model: FastQwen4Exp, mtp: Mtp4, state: FastState4, mst: Mtp4State, k: int = 2, params=None,
                 tok: torch.Tensor | None = None, stop_ids: torch.Tensor | None = None, drafts: int | None = None,
                 mask: torch.Tensor | None = None, pool=None, early_exit: bool = True):
        self.model, self.mtp, self.state, self.mst, self.k, self.params, self.mask = model, mtp, state, mst, k, params, mask
        self.drafts = drafts or k
        self.early_exit = early_exit
        dev = state.pos_t.device
        B = state.pos_t.shape[0]
        assert k + 1 <= MAX_ROWS, f"a verify takes at most {MAX_ROWS} rows per slot"
        self.tok = tok if tok is not None else torch.zeros(B, k + 1, dtype=torch.long, device=dev)
        self.stop_ids = stop_ids if stop_ids is not None else torch.full((B, 1), -1, dtype=torch.long, device=dev)
        assert self.tok.shape == (B, k + 1) and self.stop_ids.shape[0] == B
        self.feed = model.feed_buffer(B, k + 1)
        self.idx = torch.arange(k + 1, device=dev)
        self._side = torch.cuda.Stream()  # the commit's branch
        self.skip = torch.zeros(1, dtype=torch.int32, device=dev)  # the draft early exit flag (skinny_skip)
        state.reset(kv=False)  # the capture's warm-up runs write the first rows; the rest of the KV is never read
        self.graph, (self.out_tok, self.n, self.logits, self.H) = capture(self._body, pool)
        state.reset(kv=False)
        mst.pos_t.zero_()

    def _body(self):
        k, m, mtp, st, mst = self.k, self.model, self.mtp, self.state, self.mst
        tok = self.tok
        B = tok.shape[0]
        logits, H = m.verify(tok, st)                                # [B, k+1, V], [B, k+1, 10240]; rows from feed_buffer
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
        # the drafting below never reads the target's GDN / PLE state or its positions: commit on a parallel branch
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            m.commit(st, n)
        nl = n.long()
        out_tok = torch.cat([d, bonus[:, None]], 1)                  # accepted drafts are d[:n-1]; the bonus is out_tok[n-1]
        out_tok = torch.where(self.idx[None] == nl[:, None] - 1, bonus[:, None], out_tok)
        # catch-up: rows at positions p0 .. p0+k with tokens x_{i+1} (= out_tok[i] for i < n) and the target's H[i]; the
        # rows from p0 + n on are rewritten by the chained steps and the next cycle
        mst.pos_t.copy_(p0)
        g, x4 = mtp(out_tok, H, mst)                                 # [B, k+1, 2560], [B, k+1, 10240]
        last = (nl - 1).clamp_min(0)
        gp = g.gather(1, last[:, None, None].expand(B, 1, g.shape[-1]))
        xp = x4.gather(1, last[:, None, None].expand(B, 1, x4.shape[-1]))
        dd, cp = mtp.draft(gp)                                       # [B, 1]; cp: the running product of draft probabilities
        dr = [dd]
        mst.pos_t.copy_(p0 + n)
        self.skip.zero_()
        for _ in range(self.drafts - 1):
            if self.early_exit:  # every slot's chain is unlikely to survive this far: the remaining steps skip their GEMMs
                self.skip.copy_(torch.maximum(self.skip, ((cp[:, 0] < DRAFT_STOP) | (st.active == 0)).all().int().view(1)))
                ops().skinny_skip(self.skip)
            try:
                gp, xp = mtp(dr[-1], xp, mst)
                dd, p = mtp.draft(gp)
            finally:
                ops().skinny_skip(None)
            cp = cp * p
            dr.append(dd)
            mst.pos_t += 1
        nxt = torch.cat([bonus[:, None]] + dr, 1)                    # [B, drafts+1] = [y', d1'..]
        nd = self.drafts + 1
        self.tok[:, :nd].copy_(torch.where(st.active[:, None] > 0, nxt, tok[:, :nd]))
        torch.cuda.current_stream().wait_stream(self._side)
        return out_tok, n, logits, H

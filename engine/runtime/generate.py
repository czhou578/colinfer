"""Single-slot generation for the Phase 1 reference model: prefill, then one token per forward.
No graphs and no fusion. It returns the tokens, and optionally the fp32 logits of the first N steps."""
from __future__ import annotations

import torch

from engine.model.qwen35 import Qwen35ForCausalLM


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, gen: torch.Generator | None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


@torch.inference_mode()
def generate(model: Qwen35ForCausalLM, input_ids: torch.Tensor, max_new_tokens: int, *, eos_ids=(), temperature: float = 0.0,
             top_k: int = 0, top_p: float = 1.0, seed: int | None = None, keep_logits: int = 0, max_seq_len: int | None = None):
    """input_ids [1, T] long on the model device. Greedy when temperature == 0."""
    assert input_ids.shape[0] == 1, "single slot"
    T = input_ids.shape[1]
    state = model.new_state(1, max_seq_len or (T + max_new_tokens))
    gen = None
    if seed is not None:
        gen = torch.Generator(device=input_ids.device).manual_seed(seed)
    logits = model(input_ids, state, last_only=True)[:, -1]
    out, kept = [], []
    eos = set(int(e) for e in eos_ids)
    for step in range(max_new_tokens):
        if step < keep_logits:
            kept.append(logits[0].clone())
        nxt = sample(logits, temperature, top_k, top_p, gen)
        tok = int(nxt)
        out.append(tok)
        if tok in eos:
            break
        logits = model(nxt.view(1, 1), state, last_only=True)[:, -1]
    return out, (torch.stack(kept) if kept else None)

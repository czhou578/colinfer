"""Single-slot greedy generation for the Phase 1 reference model: prefill, then one token per forward.
No graphs and no fusion."""
from __future__ import annotations

import torch

from engine.model.qwen35 import Qwen35ForCausalLM


@torch.inference_mode()
def generate(model: Qwen35ForCausalLM, input_ids: torch.Tensor, max_new_tokens: int, *, eos_ids=(),
             max_seq_len: int | None = None) -> list[int]:
    """input_ids [1, T] long on the model device. The tokens up to and including the first stop token."""
    assert input_ids.shape[0] == 1, "single slot"
    state = model.new_state(1, max_seq_len or (input_ids.shape[1] + max_new_tokens))
    logits = model(input_ids, state, last_only=True)[:, -1]
    out, eos = [], set(int(e) for e in eos_ids)
    for _ in range(max_new_tokens):
        nxt = logits.argmax(dim=-1)
        out.append(int(nxt))
        if out[-1] in eos:
            break
        logits = model(nxt.view(1, 1), state, last_only=True)[:, -1]
    return out

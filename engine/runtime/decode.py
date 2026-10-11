"""One plain decode step as a CUDA graph: embed -> 64 layers -> lm_head -> sampler, T = 1 per slot.

The scheduler without speculation (--spec none) runs one per slot (engine/runtime/scheduler.py). With speculation,
engine/spec/mtp.py MtpCycle captures the whole speculative cycle instead. Both capture with engine/model/fast.py
capture().
"""
from __future__ import annotations

import torch

from engine.model.fast import FastQwen35, FastState, capture
from engine.model.prefill import prepare_prefill
from engine.runtime.sampler import SamplerParams, sample


class DecodeGraph:
    """Capture runs on a fresh state and resets it afterwards (warm-up executes the step for real). Per step the host
    writes the input tokens into a static buffer, replays, and reads back the sampled ids (logits in self.logits)."""

    def __init__(self, model: FastQwen35, state: FastState, params: SamplerParams | None = None):
        """params: optional SamplerParams (B slots) shared with other graphs; default: a private one."""
        prepare_prefill(model)  # re-points weights (stacking); must happen before the graph records addresses
        self.model, self.state = model, state
        B = state.pos_t.shape[0]
        self.params = params if params is not None else SamplerParams(B, model.cfg.vocab_size, state.pos_t.device)
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=state.pos_t.device)

        def body():
            logits = model(self.tok, state)
            return logits, sample(logits, self.params, state.pos_t)  # pos_t now holds the predicted token's position
        state.reset()
        self.graph, (self.logits, self.next) = capture(body)
        state.reset()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B] long on the device. Returns sampled ids [B] (device; greedy for slots with temperature 0)."""
        self.tok.copy_(tokens.view(-1, 1))
        self.graph.replay()
        return self.next

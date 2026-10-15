"""Image input: the vision tower of the checkpoint (a 27-block ViT in BF16, 0.46B parameters, transformers'
Qwen3_5VisionModel) and the multimodal RoPE positions of a prompt that holds images.

A prompt holds each image as a run of n placeholder tokens: one per 2 x 2 group of 16-pixel patches of the resized
image. Each run gets its own ids past the vocabulary, derived from a hash of the image's bytes and its grid
(Image.ids). Thus the prefix caches, in memory and on disk, tell two images apart, and a conversation that sends the
same image again reuses it. Prefill embeds these tokens as the image pad token and replaces their rows with the
output of the vision tower (engine/model/prefill.py embed()). The engine runs the vision tower only for images whose
tokens it prefills, and the image processor's work (resize, normalize, patch: up to ~0.5 s) happens only then
(Image.patches), so a restored prefix costs nothing.

Positions (multimodal RoPE, as get_rope_index of transformers' Qwen3.5): a text token at prompt index p rotates by
p + delta, where delta sums max(gh, gw) - gh * gw over the images before p (gh x gw: the grid of the image's tokens).
The tokens of an image that starts at index s rotate by (t, h, w) = (s', s' + row, s' + col), with s' = s + delta(s).
Prefill gives the attention prologue each row's (t, h, w) (rope_positions); decode gives it the slot's delta
(rope_delta), since a reply holds no images.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Callable

import numpy as np
import torch
from safetensors import safe_open

from engine.weights.loader import weight_map

ID_BITS = 48  # image token ids: vocab + (key + j) mod 2^48, for token j of an image whose pixels hash to key


@dataclasses.dataclass
class Image:
    """One image of a prompt: its grid and key from the image processor (engine/server/images.py), its place in the
    prompt, its patches (given, or made by `make` when the vision tower needs them) and, once computed, the tower's
    output."""
    pixels: torch.Tensor | None   # [patches, 3 * 2 * 16 * 16] float32 (CPU); None until patches() makes them
    grid: tuple[int, int, int]    # (t, h, w) in patches
    key: int                      # a 48-bit hash of the image's bytes and the grid
    merge: int = 2                # patches per token side (spatial_merge_size)
    start: int = -1               # the prompt index of its first token
    embeds: torch.Tensor | None = None  # [n, hidden] bf16 (device), the vision tower's output
    make: Callable[[], torch.Tensor] | None = None  # the processor's resize, normalize and patch of the image's bytes

    def patches(self) -> torch.Tensor:
        """The pixels, made now if the image came without them."""
        if self.pixels is None:
            if self.make is None:
                raise ValueError("an image without pixels")
            self.pixels = self.make()
        return self.pixels

    @property
    def hw(self) -> tuple[int, int]:
        """The grid of its tokens."""
        return self.grid[1] // self.merge, self.grid[2] // self.merge

    @property
    def n(self) -> int:
        h, w = self.hw
        return h * w

    def __post_init__(self):
        if self.grid[0] != 1:
            raise ValueError("an image has one temporal patch (video input is not supported)")

    def ids(self, vocab: int) -> list[int]:
        return [vocab + (self.key + j) % (1 << ID_BITS) for j in range(self.n)]


def rope_positions(images: list[Image], lo: int, hi: int) -> np.ndarray:
    """The (t, h, w) rotary positions of the prompt indices [lo, hi): int32 [hi - lo, 3]."""
    p = np.arange(lo, hi, dtype=np.int64)
    out = np.repeat(p[:, None], 3, axis=1)  # text before the first image
    acc = 0  # the delta of the text before the current image
    for im in sorted(images, key=lambda x: x.start):
        s, n = im.start, im.n
        gh, gw = im.hw
        inside = (p >= s) & (p < s + n)
        j = p[inside] - s
        base = s + acc
        out[inside, 0] = base
        out[inside, 1] = base + j // gw
        out[inside, 2] = base + j % gw
        acc += max(gh, gw) - n
        after = p >= s + n
        out[after] = (p[after] + acc)[:, None]
    return out.astype(np.int32)


def rope_delta(images: list[Image]) -> int:
    """The delta of every position after the last image (the reply's)."""
    return sum(max(im.hw) - im.n for im in images)


class VisionTower:
    """The checkpoint's vision tower on the GPU (BF16, 0.92 GB), as transformers runs it. With load=False, the weights
    are loaded by the first encode() (on the engine thread), so a server that never sees an image keeps the memory;
    the patch shape and the merge size come from the config at once."""

    def __init__(self, path: str, device="cuda", load: bool = True):
        self.path, self.device, self.model = path, device, None
        vc = json.load(open(os.path.join(path, "config.json")))["vision_config"]
        self.merge = vc["spatial_merge_size"]
        self.patch_dim = vc["in_channels"] * vc["temporal_patch_size"] * vc["patch_size"] ** 2
        if load:
            self.load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding
        path, device = self.path, self.device
        c = json.load(open(os.path.join(path, "config.json")))
        vc = Qwen3_5VisionConfig(**c["vision_config"])
        vc._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3_5VisionModel(vc)
        sd = {}
        for name, f in weight_map(path).items():
            if name.startswith("model.visual."):
                with safe_open(os.path.join(path, f), framework="pt", device=str(device)) as sf:
                    sd[name[len("model.visual."):]] = sf.get_tensor(name).to(torch.bfloat16)
        model.load_state_dict(sd, strict=True, assign=True)
        with torch.device(device):  # its frequencies are a buffer, not a weight: computed again, in fp32 as transformers keeps them
            model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(vc)
        left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        if left:
            raise RuntimeError(f"vision tower: tensors left on meta: {left[:5]}")
        assert (self.merge, self.patch_dim) == (vc.spatial_merge_size, vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2)
        self.model = model.eval()

    @torch.inference_mode()
    def encode(self, im: Image) -> torch.Tensor:
        """The image's token embeddings [n, hidden] bf16, in the order of its tokens (row-major over its grid)."""
        if self.model is None:
            self.load()
        px = im.patches()
        if px.shape != (im.grid[0] * im.grid[1] * im.grid[2], self.patch_dim):
            raise ValueError(f"image patches {tuple(px.shape)} do not match the grid {im.grid}")
        out = self.model(hidden_states=px.to(self.device), grid_thw=torch.tensor([im.grid], device=self.device)).pooler_output
        if im.make is not None:
            im.pixels = None  # up to 100 MB of host memory, made again if ever needed
        return out.to(torch.bfloat16)


def vision_available(path: str) -> bool:
    """True when the checkpoint has a vision tower (its config and weights)."""
    c = json.load(open(os.path.join(path, "config.json")))
    return "vision_config" in c and any(n.startswith("model.visual.") for n in weight_map(path))

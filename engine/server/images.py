"""Images in requests (engine/model/vision.py runs them).

* Messages: an OpenAI image part ({"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}) becomes the
  part the chat template renders as one image pad token ({"type": "image"}); engine/server/anthropic.py turns the
  Anthropic image blocks into such parts. Only data URLs: the server fetches nothing.
* Pixels: the checkpoint's image processor (transformers' Qwen2VLImageProcessorPil with the settings of
  preprocessor_config.json): RGB, a bicubic resize to a multiple of 32 pixels per side with at least 65,536 and at most
  max_pixels pixels, 16-pixel patches in 2 x 2 groups. An image costs one prompt token per 32 x 32 pixels, so the
  default max_pixels (2048 x 2048) caps it at 4,096 tokens.
* Prompt: expand() replaces the k-th image pad token of the rendered prompt with the k-th image's run of tokens.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os

import torch

from engine.model.vision import Image

MAX_PIXELS = 2048 * 2048
# The decode of an image is capped at this multiple of max_pixels. A client may send a photo larger than the resize
# target, so the cap is not max_pixels itself. At the default max_pixels, 8 x gives 32M pixels: about 130 MB of RGBA.
# The server runs one decode at a time (engine/server/api.py). PIL's own bomb limit (179M pixels) is too high.
DECODE_FACTOR = 8
# The image formats of the OpenAI and Anthropic APIs. No other PIL plugin runs on client bytes: an ICO decodes its
# embedded PNG inside open(), before any size check, and an EPS runs the client's PostScript in Ghostscript.
FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
NOTE = "[image omitted: this server reads text only]"


def data_url_bytes(url) -> bytes:
    """The bytes of a base64 data URL. ValueError for anything else (the client's error)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("an image must be a base64 data URL (data:image/png;base64,...): the server fetches nothing")
    head, sep, data = url.partition(",")
    if not sep or not head.endswith(";base64"):
        raise ValueError("an image data URL must be base64-encoded")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("an image data URL holds invalid base64") from None


def _image_url(part: dict):
    u = part.get("image_url")
    return u.get("url") if isinstance(u, dict) else u


def extract(messages: list[dict], enabled: bool) -> tuple[list[dict], list[bytes]]:
    """(the messages with each image part replaced by the template's image part, or by a text note when enabled is
    False; the bytes of each image, in prompt order)."""
    out, blobs = [], []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list) or not any(isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            out.append(m)
            continue
        parts = []
        for p in c:
            if p.get("type") != "image_url":
                parts.append(p)
            elif enabled:
                blobs.append(data_url_bytes(_image_url(p)))
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": NOTE})
        out.append(dict(m, content=parts))
    return out, blobs


class ImageProcessor:
    """The checkpoint's image processor, from its preprocessor_config.json. max_pixels caps the resized image."""

    def __init__(self, path: str, max_pixels: int = MAX_PIXELS):
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil, smart_resize
        c = json.load(open(os.path.join(path, "preprocessor_config.json")))
        size = c["size"]
        self.min_pixels, self.max_pixels = size["shortest_edge"], min(size["longest_edge"], max_pixels)
        self.merge, self.patch = c["merge_size"], c["patch_size"]
        self.resize = smart_resize  # the processor's choice of the resized size (height, width)
        self.proc = Qwen2VLImageProcessorPil(size={"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels},
                                             patch_size=self.patch, temporal_patch_size=c["temporal_patch_size"], merge_size=self.merge,
                                             image_mean=c["image_mean"], image_std=c["image_std"])

    def _open(self, data: bytes):
        """The decoded image. ValueError if it is not a PNG, JPEG, WEBP or GIF, or if it has more than
        DECODE_FACTOR x max_pixels pixels. Pillow reads the size from the header, so the check comes before the decode."""
        import PIL.Image
        limit = DECODE_FACTOR * self.max_pixels
        try:
            img = PIL.Image.open(io.BytesIO(data), formats=FORMATS)  # reads the header only
            w, h = img.size
            if w * h <= limit:
                img.load()
        except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError, OSError, ValueError) as e:
            raise ValueError(f"cannot read the image: {e}") from None
        if w * h > limit:
            raise ValueError(f"image of {w} x {h} pixels: it must have at most {limit:,} pixels")
        return img

    def load(self, data: bytes) -> Image:
        """Checks one image (decoded now, not kept) and gives it its grid, from its size, and its key, from its bytes.
        Its patches are made when the vision tower needs them (Image.patches: the resize, normalization and patching
        take up to ~0.5 s), which a prefix cache hit never does. ValueError if it is not an image that Pillow reads, or if
        it has too many pixels (see _open)."""
        img = self._open(data)
        w, h = img.size
        if min(w, h) < 1 or max(w, h) / min(w, h) > 200:
            raise ValueError(f"image of {w} x {h} pixels: the aspect ratio must be below 200")
        rh, rw = self.resize(h, w, factor=self.patch * self.merge, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        grid = (1, rh // self.patch, rw // self.patch)
        hsh = hashlib.blake2b(digest_size=6)
        hsh.update(data)
        hsh.update(repr(grid).encode())
        key = int.from_bytes(hsh.digest(), "little")
        return Image(None, grid, key, self.merge, make=lambda: self.patches(data, grid))

    def patches(self, data: bytes, grid: tuple) -> torch.Tensor:
        """The processor's patches of an image that load() accepted: [patches, 3 * 2 * 16 * 16] float32."""
        out = self.proc(images=[self._open(data).convert("RGB")], return_tensors="pt")
        if tuple(int(x) for x in out["image_grid_thw"][0]) != grid:
            raise ValueError(f"image grid {tuple(out['image_grid_thw'][0].tolist())} differs from the one derived from its size, {grid}")
        return out["pixel_values"].float().contiguous()


def expand(prompt: list[int], images: list[Image], pad_id: int, vocab: int) -> list[int]:
    """The prompt with the k-th image pad token replaced by the tokens of images[k] (whose start it sets)."""
    at = [i for i, t in enumerate(prompt) if t == pad_id]
    if len(at) != len(images):
        raise ValueError(f"the prompt has {len(at)} image places for {len(images)} images")
    out, prev = [], 0
    for i, im in zip(at, images):
        out += prompt[prev:i]
        im.start = len(out)
        out += im.ids(vocab)
        prev = i + 1
    return out + prompt[prev:]

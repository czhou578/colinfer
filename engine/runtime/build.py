"""Builds the engine as the server runs it: the CUDA kernels, the quantized model and its INT6 / INT5 decode copies, the
MTP drafter, the startup self-test and the scheduler.

The server (engine/server/worker.py) builds it here, and so do the end-to-end checks and the benchmarks (tests/golden.py,
tests/scheduler_check.py, tests/passkey.py, bench/). The defaults are the server's, so a check of the server
configuration runs exactly that configuration.
"""
from __future__ import annotations

import json
import os
import time
from typing import Literal

from engine.kernels import ops
from engine.model.fast import attach_decode_copies, decode_copies_paths, load_fast_model, to_fast
from engine.model.vision import VisionTower, vision_available
from engine.runtime.scheduler import Scheduler
from engine.selftest import run_selftest
from engine.spec.mtp import DRAFT_DIR, Mtp
from engine.spec.suffix import MIN_MATCH
from engine.weights.direct import release_buffers
from engine.weights.loader import MODEL, resolve


def load_model(path_or_repo: str = MODEL, decode_weights: str = "int", log=print):
    """(checkpoint directory, the model on the kernel decode path). decode_weights: "int" streams the INT6 / INT5 decode
    copies of the attention / GDN projections (tools/int6_requant.py) when their files exist; "checkpoint" streams the
    checkpoint's FP8 weights."""
    if decode_weights not in ("int", "checkpoint"):
        raise ValueError(f"decode_weights: int or checkpoint, not {decode_weights!r}")
    path = resolve(path_or_repo)
    model = to_fast(load_fast_model(path))
    if decode_weights == "int":
        files = decode_copies_paths(path)
        if all(os.path.exists(f) for f in files):
            log(f"[engine] decode streams the INT6 / INT5 projection copies: {attach_decode_copies(model, files)} linears")
        else:
            log(f"[engine] no INT6 / INT5 decode copies at {files} (tools/int6_requant.py): decoding the checkpoint's FP8 projections")
    return path, model


def load_drafter(model, path: str, weights: str = "auto", log=print):
    """The MTP drafter. weights: "auto" = the fine-tuned head (tools/train_drafter.py) when its file exists, "none" = the
    checkpoint's, or a path."""
    if weights == "auto":
        weights = os.path.join(DRAFT_DIR, "mtp_ft.safetensors")
        weights = weights if os.path.exists(weights) else None
    elif weights == "none":
        weights = None
    mtp = Mtp(model, path, weights=weights)
    log(f"[engine] MTP drafter: {weights or 'checkpoint weights'}; low-rank draft head: {'on' if mtp.lr_B is not None else 'off'}")
    return mtp


def boundary_token(path: str) -> int:
    """The chat message boundary (<|im_start|>), where the scheduler ends prefill chunks and takes checkpoints."""
    from transformers import AutoTokenizer  # a slow import, only for callers that have no tokenizer yet
    return AutoTokenizer.from_pretrained(path).convert_tokens_to_ids("<|im_start|>")


def _file_id(f: str) -> list:
    """A weights file as the disk cache's fingerprint names it: a changed file gives new KV bytes."""
    st = os.stat(f)
    return [os.path.basename(f), st.st_size, int(st.st_mtime)]


def build_engine(model: str = MODEL, *, slots: int = 3, max_seq_len: int = 262144, checkpoints: int = 32, spec: str = "mtp", k: int = 7,
                 drafter_weights: str = "auto", suffix_drafts: int = MIN_MATCH, decode_weights: str = "int",
                 boundary: Literal["auto"] | int | None = "auto", selftest: bool = True, vision: bool = True, vision_preload: bool = False,
                 metrics=None, log=print):
    """The scheduler with its model, as `python -m engine.server` builds it from the same flags (the defaults are the
    server's). spec: "mtp" or "none". boundary: the message-boundary token id, "auto" (the checkpoint's) or None.
    vision: image input (engine/model/vision.py) when the checkpoint has a vision tower; its 0.92 GB of weights load on
    the first image, or at startup with vision_preload.
    Returns (scheduler, the startup seconds of each phase). scheduler.fingerprint names what the cached state depends
    on (the checkpoint, the decode weights, the drafter), for the disk cache (engine/runtime/sessions.py)."""
    if spec not in ("mtp", "none"):
        raise ValueError(f"spec: mtp or none, not {spec!r}")
    t0 = time.perf_counter()
    ops()  # build / load the CUDA extension
    t1 = time.perf_counter()
    path, m = load_model(model, decode_weights, log)
    mtp = load_drafter(m, path, drafter_weights, log) if spec == "mtp" else None
    release_buffers()  # the direct reader's pinned staging buffers (engine/weights/direct.py)
    if boundary == "auto":
        boundary = boundary_token(path)
    tower = VisionTower(path, load=vision_preload) if vision and vision_available(path) else None
    if tower is not None:
        log("[engine] image input on: vision tower " + ("loaded" if tower.loaded else "loads on the first image"))
    t2 = time.perf_counter()
    if selftest:
        run_selftest(verbose=True)  # refuses to start if any matmul path is numerically wrong
    sched = Scheduler(m, n_slots=slots, max_seq_len=max_seq_len, n_checkpoints=checkpoints, mtp=mtp, k=k, metrics=metrics,
                      boundary_token=boundary, suffix_min=suffix_drafts)
    if tower is not None:
        sched.vision, sched.image_pad = tower, json.load(open(os.path.join(path, "config.json")))["image_token_id"]
    copies = decode_copies_paths(path)
    int_copies = decode_weights == "int" and all(os.path.exists(f) for f in copies)
    sched.fingerprint = dict(model=os.path.basename(path), decode=[_file_id(f) for f in copies] if int_copies else "checkpoint",
                             drafter=None if mtp is None else (_file_id(mtp.weights) if mtp.weights else "checkpoint"))
    t3 = time.perf_counter()
    return sched, dict(kernels_s=round(t1 - t0, 1), weights_s=round(t2 - t1, 1), graphs_selftest_s=round(t3 - t2, 1))

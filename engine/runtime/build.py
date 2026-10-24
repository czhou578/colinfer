"""Builds the engine as the server runs it: the CUDA kernels, the quantized model and the artifacts of its prepared
directory (engine/runtime/prepared.py: the INT6 / INT5 decode copies, the fine-tuned drafter, the low-rank draft head),
the MTP drafter, the startup self-test and the scheduler. Before it builds anything, it compares the startup memory of
the engine with MemAvailable (check_memory). An over-commit of the unified memory powers the machine off, so the build
stops with NotEnoughMemory when the engine does not fit.

The server (engine/server/worker.py) builds it here, and so do the end-to-end checks and the benchmarks (tests/golden.py,
tests/scheduler_check.py, tests/passkey.py, bench/). The defaults are the server's. tests/scheduler_check.py keeps them,
so it runs the server's configuration, YaRN included. tests/golden.py runs 32768-token slots: the server's unscaled model
(--max-seq-len 262144).

The model family of the checkpoint (engine/model/family.py, by the model_type of its config.json) gives the defaults
of the slots, the slot length, the checkpoints and the speculation, the memory figures of check_memory, the loader and
the prefill callables. The 27B (qwen3_5) loads here (load_dense_model).
"""
from __future__ import annotations

import json
import os
import time
from typing import Literal

from engine.kernels import ops
from engine.model.family import Family, check_served, family_of
from engine.model.fast import attach_decode_copies, load_fast_model, to_fast
from engine.model.qwen35 import apply_yarn, yarn_attention_factor
from engine.model.vision import VisionTower, vision_available
from engine.runtime.prepared import Prepared
from engine.runtime.prepared import read as read_prepared
from engine.runtime.scheduler import Scheduler
from engine.selftest import run_selftest
from engine.spec.mtp import Mtp
from engine.spec.suffix import MIN_MATCH
from engine.weights.direct import release_buffers
from engine.weights.loader import MODEL, resolve


def native_context(path: str) -> int:
    """The checkpoint's native context: max_position_embeddings of its config.json (of its text_config, when it has one)."""
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    return c.get("text_config", c)["max_position_embeddings"]


def check_seq_len(path: str, fam: Family, max_seq_len: int | None) -> None:
    """Raises ValueError when max_seq_len exceeds the native context of a family without YaRN."""
    if max_seq_len is not None and not fam.supports.yarn and max_seq_len > native_context(path):
        raise ValueError(f"max_seq_len {max_seq_len} exceeds the native context {native_context(path)} of {os.path.basename(path)}, and "
                         f"the engine has no YaRN for the {fam.name} family: lower --max-seq-len")


def load_model(path_or_repo: str = MODEL, decode_weights: str = "int", log=print, prep: Prepared | None | str = "auto",
               max_seq_len: int | None = None):
    """(checkpoint directory, the model on the kernel decode path), through the loader of the checkpoint's family
    (engine/model/family.py). decode_weights: "int" streams the INT6 / INT5 decode copies of the attention / GDN
    projections from the prepared directory (tools/prepare.py) when it has them; "checkpoint" streams the checkpoint's
    weights. prep: the manifest of the prepared directory ("auto": read it). max_seq_len: the slot length the model will
    run at; past the checkpoint's native context, YaRN scales its rope by the ratio (engine/model/qwen35.py apply_yarn),
    as Qwen documents for its models. A family without YaRN raises ValueError there."""
    if decode_weights not in ("int", "checkpoint"):
        raise ValueError(f"decode_weights: int or checkpoint, not {decode_weights!r}")
    path = resolve(path_or_repo)
    check_served(path)
    fam = family_of(path)  # a family the engine cannot serve yet raises NotImplementedError
    check_seq_len(path, fam, max_seq_len)
    if prep == "auto":
        prep = read_prepared(path)
    return fam.load(path, decode_weights, log, prep, max_seq_len)


def load_dense_model(path: str, decode_weights: str, log, prep: Prepared | None, max_seq_len: int | None):
    """The loader of the qwen3_5 family (the 27B): the arguments of load_model, with the checkpoint directory resolved
    and the prepared directory read."""
    model = load_fast_model(path)
    native = model.cfg.max_position_embeddings
    if max_seq_len is not None and max_seq_len > native:
        apply_yarn(model, max_seq_len / native)
        log(f"[engine] YaRN x{max_seq_len / native:g}: {max_seq_len} tokens per slot over the checkpoint's {native} native positions "
            f"(rope frequencies scaled, q / k rotated dims x{yarn_attention_factor(model.cfg):.4f})")
    model = to_fast(model)
    if decode_weights == "int":
        if prep is not None and prep.decode:
            log(f"[engine] decode streams the INT6 / INT5 projection copies of {prep.dir}: {attach_decode_copies(model, prep.decode)} linears")
        else:
            log("[engine] no prepared INT6 / INT5 decode copies (tools/prepare.py): decoding the checkpoint's FP8 projections")
    return path, model


def load_drafter(model, path: str, weights: str = "auto", log=print, prep: Prepared | None = None):
    """The MTP drafter. weights: "auto" = the fine-tuned head of the prepared directory when it has one, "none" = the
    checkpoint's, or a path. The low-rank draft head is the prepared directory's, when it has one."""
    if weights == "auto":
        weights = prep.drafter if prep is not None else None
    elif weights == "none":
        weights = None
    mtp = Mtp(model, path, weights=weights, lowrank=prep.lowrank if prep is not None else None)
    log(f"[engine] MTP drafter: {weights or 'checkpoint weights'}; low-rank draft head: {'on' if mtp.lr_B is not None else 'off'}; "
        f"quantized {'from the cache' if mtp.cached else 'now (cached for the next start)' if mtp.cache_file else 'now'}")
    return mtp


def boundary_token(path: str) -> int:
    """The chat message boundary (<|im_start|>), where the scheduler ends prefill chunks and takes checkpoints."""
    from transformers import AutoTokenizer  # a slow import, only for callers that have no tokenizer yet
    return AutoTokenizer.from_pretrained(path).convert_tokens_to_ids("<|im_start|>")


def _file_id(f: str) -> list:
    """A weights file as the disk cache's fingerprint names it: a changed file gives new KV bytes."""
    st = os.stat(f)
    return [os.path.basename(f), st.st_size, int(st.st_mtime)]


# The startup memory of the engine beyond the slot state. Measured on 2026-10-23 at the defaults: MemAvailable fell by
# 93.9e9 bytes, and the slot state is 55.2e9 of them. The rest is the weights (20.2e9), the INT copies (5.2e9), the
# checkpoint ring (4.9e9), the drafter, the graph pool, the CUDA context and the pinned buffers of the conversation cache.
ENGINE_FIXED_BYTES = 40 * 10**9
# The memory that the server takes after startup, on top of ENGINE_FIXED_BYTES. The HTTP layer (engine/server/api.py)
# bounds each part: at most max_queue + 1 = 9 requests between the body read and the queue, each with a body of up to
# 32 MB, its parsed copy and its text (about 0.3e9 together, 2.7e9 in all); one tokenize at the text limit (1.0e9) or
# one image decode (0.13e9) at a time; the vision tower on the first image (0.92e9). The sum is about 4.7e9. The rest is
# a margin for the pinned host buffers. Without a reserve, about 0.4e9 stays free after the first image.
GROWTH_RESERVE_BYTES = 6 * 10**9


class NotEnoughMemory(RuntimeError):
    """The engine does not fit in the free unified memory. An over-commit powers the machine off, with no out-of-memory
    error, so the engine does not start."""


def meminfo_bytes(key: str) -> int:
    """A field of /proc/meminfo in bytes, for example MemAvailable. The GPU allocations come from the same unified
    memory. A missing field is a KeyError."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(key)


def process_rss_bytes() -> int:
    """VmRSS of this process, from /proc/self/status, in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def startup_bytes(path: str, slots: int | None = None, max_seq_len: int | None = None, spec: str | None = None) -> int:
    """The unified memory that the engine takes at startup: the state of the slots (the KV cache and the GDN state,
    and the drafter's with spec mtp), built on the meta device, plus the fixed bytes of the family (the 27B:
    ENGINE_FIXED_BYTES). Other flags than the defaults take less. None: the default of the family. The state shapes are
    the Scheduler's: it builds them with the family's new_state (model.new_state) and MtpState."""
    fam = family_of(path)
    d = fam.defaults
    slots, max_seq_len, spec = d.slots if slots is None else slots, d.max_seq_len if max_seq_len is None else max_seq_len, spec or d.spec
    return fam.fixed_bytes + fam.state_bytes(path, slots, max_seq_len, spec)


def check_memory(path: str, slots: int | None = None, max_seq_len: int | None = None, spec: str | None = None) -> None:
    """Raises NotEnoughMemory when the startup memory and GROWTH_RESERVE_BYTES exceed MemAvailable. It reads only the
    checkpoint's config.json and builds no tensor, so it runs before the CUDA context and the weights. The memory that
    this process holds already counts as available: ENGINE_FIXED_BYTES is the fall of MemAvailable from a fresh process,
    so a second check, after the CUDA context and the imports, must not count them twice. None: the default of the
    family."""
    d = family_of(path).defaults
    slots, max_seq_len = d.slots if slots is None else slots, d.max_seq_len if max_seq_len is None else max_seq_len
    need = startup_bytes(path, slots, max_seq_len, spec) + GROWTH_RESERVE_BYTES
    avail = meminfo_bytes("MemAvailable") + process_rss_bytes()
    if need > avail:
        raise NotEnoughMemory(f"the engine needs about {need / 1e9:.0f} GB of memory ({slots} slots x {max_seq_len} tokens, with "
                              f"{GROWTH_RESERVE_BYTES / 1e9:.0f} GB for the growth after startup), and {avail / 1e9:.0f} GB is available: "
                              "stop other GPU work, or lower the slots or the tokens per slot (--slots, --max-seq-len)")


def build_engine(model: str = MODEL, *, slots: int | None = None, max_seq_len: int | None = None, checkpoints: int | None = None,
                 spec: str | None = None, k: int = 7, drafter_weights: str = "auto", suffix_drafts: int = MIN_MATCH, decode_weights: str = "int",
                 boundary: Literal["auto"] | int | None = "auto", selftest: bool = True, vision: bool | None = None, vision_preload: bool = False,
                 metrics=None, log=print):
    """The scheduler with its model, as `python -m engine.server` builds it from the same flags (the defaults are the
    server's). slots, max_seq_len, checkpoints, spec, vision: None takes the default of the checkpoint's family
    (engine/model/family.py; the 27B: 3 slots of 524288 tokens, 32 checkpoints, spec mtp, vision on). spec: "mtp" or
    "none". boundary: the message-boundary token id, "auto" (the checkpoint's) or None. vision: image input
    (engine/model/vision.py) when the family supports it and the checkpoint has a vision tower; its 0.92 GB of weights
    load on the first image, or at startup with vision_preload. Before the CUDA context, check_memory raises
    NotEnoughMemory when the engine does not fit in MemAvailable.
    Returns (scheduler, the startup seconds of each phase). scheduler.fingerprint names what the cached state depends
    on (the checkpoint, the decode weights, the drafter), for the disk cache (engine/runtime/sessions.py)."""
    if spec not in ("mtp", "none", None):
        raise ValueError(f"spec: mtp or none, not {spec!r}")
    path = resolve(model)
    check_served(path)  # a family the engine cannot serve yet is refused before anything else
    fam = family_of(path)
    d = fam.defaults
    slots = d.slots if slots is None else slots
    max_seq_len = d.max_seq_len if max_seq_len is None else max_seq_len
    checkpoints = d.checkpoints if checkpoints is None else checkpoints
    spec = spec or d.spec
    vision = d.vision if vision is None else vision
    if spec == "mtp" and not fam.supports.mtp:
        raise ValueError(f"spec mtp: the engine has no MTP drafter for the {fam.name} family yet (spec none)")
    check_seq_len(path, fam, max_seq_len)
    check_memory(path, slots, max_seq_len, spec)  # before the CUDA context and the weights
    t0 = time.perf_counter()
    ops()  # build / load the CUDA extension
    t1 = time.perf_counter()
    prep = read_prepared(path)  # a manifest that names a missing file stops the start here
    log(f"[engine] prepared directory: {prep.dir} (made {prep.made})" if prep is not None else
        "[engine] no prepared directory (tools/prepare.py): the checkpoint alone")
    path, m = fam.load(path, decode_weights, log, prep, max_seq_len)
    mtp = load_drafter(m, path, drafter_weights, log, prep) if spec == "mtp" else None
    release_buffers()  # the direct reader's pinned staging buffers (engine/weights/direct.py)
    if boundary == "auto":
        boundary = boundary_token(path)
    tower = VisionTower(path, load=vision_preload) if vision and fam.supports.vision and vision_available(path) else None
    if tower is not None:
        log("[engine] image input on: vision tower " + ("loaded" if tower.loaded else "loads on the first image"))
    t2 = time.perf_counter()
    if selftest:
        run_selftest(verbose=True)  # refuses to start if any matmul path is numerically wrong
    sched = Scheduler(m, n_slots=slots, max_seq_len=max_seq_len, n_checkpoints=checkpoints, mtp=mtp, k=k, metrics=metrics,
                      boundary_token=boundary, suffix_min=suffix_drafts, prefill_fn=fam.prefill, prepare_fn=fam.prepare_prefill,
                      before_replay=fam.before_replay(m), feed_buffer=fam.feed_buffer(m))
    sched.family = fam.name
    if tower is not None:
        sched.vision, sched.image_pad = tower, json.load(open(os.path.join(path, "config.json")))["image_token_id"]
    int_copies = fam.supports.int_copies and decode_weights == "int" and prep is not None and prep.decode
    sched.fingerprint = dict(model=os.path.basename(path), decode=[_file_id(f) for f in prep.decode] if int_copies else "checkpoint",
                             drafter=None if mtp is None else (_file_id(mtp.weights) if mtp.weights else "checkpoint"))
    t3 = time.perf_counter()
    return sched, dict(kernels_s=round(t1 - t0, 1), weights_s=round(t2 - t1, 1), graphs_selftest_s=round(t3 - t2, 1))

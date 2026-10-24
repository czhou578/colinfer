"""The model families of the engine, by the model_type of a checkpoint's config.json.

- qwen3_5: the dense Qwen3.5 / 3.6 / 3.8 family (Qwen3.8-27B). The engine serves it: engine/model/qwen35.py, fast.py,
  prefill.py. A checkpoint with another model_type, or with none, also gets this family (the behavior of the engine
  before the dispatch).
- qwen4_exp: Qwen3.8-Flash-Next, the MoE family: the decode model on the kernels (engine/model/fast_qwen4exp.py), its
  prefill (engine/model/prefill_qwen4exp.py) and the n-gram rows of its PLE layer (engine/weights/ple.py PleFeeder, with
  a host row cache of --ngram-cache-gb). The scheduler feeds the rows of each decode step through before_replay into
  the model's feed buffer, and gives each prefill chunk the two tokens before it (prefill_prev2). No drafter, no
  images, no YaRN yet (docs/history/qwen4exp_plan.md). A build without these modules refuses the checkpoint with a
  clear message, instead of a KeyError from the dense config reader.

family_of(path) gives the Family of a checkpoint: the callables and the constants that differ between the families.
The build (engine/runtime/build.py) and the server (engine/server/api.py) read the defaults, the memory figures and the
loader from it. The scheduler gets the prefill callables and the decode hooks from it.

check_served(path) runs before the memory check of the build, so the refusal costs nothing. Tests: tests/test_family.py.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
from typing import Callable

PLAN = "docs/history/qwen4exp_plan.md"
IN_PROGRESS = {"qwen4_exp": "Qwen3.8-Flash-Next"}
# The families whose serving integration is not finished: check_served refuses them even when their modules exist (name ->
# the reason). qwen4_exp left it when the end-to-end check (tests/scheduler_check_qwen4exp.py) and the server checks passed.
NOT_SERVED_YET: dict[str, str] = {}

# The startup memory of a qwen4_exp engine beyond the slot state and the n-gram row cache, at the defaults of the
# family (16 checkpoints, no drafter, no decode copies). Measured on 2026-10-24: from the start of the server process
# to its ready line (the warm-up included), MemAvailable fell by 96.73e9 bytes; the slot state is 8.69e9 of them and the
# row cache of 1 GB 1.10e9. The rest, 86.94e9, is the weights (78.5e9 on the GPU after the load), the checkpoint ring
# (1.85e9), the graph pool, the prefill workspace, the CUDA context and the pinned host buffers. The plan's estimate
# (docs/history/qwen4exp_plan.md section 4) was 85.5e9. A build without the warm-up (tests/scheduler_check_qwen4exp.py)
# measured 84.89e9 at its end, and its first long prompts took the rest.
QWEN4EXP_FIXED_BYTES = 87 * 10**9
QWEN4EXP_PREFILL_CHUNK = 4096  # tokens per prefill chunk (bench/prefill_chunk_bench.py: the MoE ceiling is 7.5k tok/s at 4096)
NGRAM_CACHE_GB = 1.0           # the default of --ngram-cache-gb: 6.25M rows of 160 bytes
NGRAM_ROW = 160                # bytes of one n-gram row (engine/weights/ple.py ROW_BYTES)
NGRAM_SLOT_BYTES = NGRAM_ROW + 16  # the host bytes of one row of the cache: the row, its id and its LRU stamp (csrc/ple_io.cpp)


def ngram_cache_rows(gb: float) -> int:
    """The rows of an n-gram row cache of gb GB (1e9 bytes) of row data; 0 turns the cache off."""
    return int(gb * 1e9) // NGRAM_ROW


@dataclasses.dataclass(frozen=True)
class Defaults:
    """The server's values for the flags that depend on the family (engine/server/api.py, docs/server.md)."""
    slots: int
    max_seq_len: int
    checkpoints: int
    spec: str      # "mtp" or "none"
    vision: bool


@dataclasses.dataclass(frozen=True)
class Supports:
    """What the engine can run for the family. The build refuses or skips a feature that the family does not support."""
    mtp: bool         # the MTP drafter (spec mtp)
    vision: bool      # image input
    yarn: bool        # a slot length past the native context (YaRN)
    int_copies: bool  # the INT6 / INT5 decode copies of the prepared directory


def _none(model):
    return None


def _no_host_bytes(ngram_cache_gb: float) -> int:
    return 0


@dataclasses.dataclass(frozen=True)
class Family:
    """The parts of the engine that differ between the model families.

    - load(path, decode_weights, log, prep, max_seq_len, ngram_cache_gb) -> (path, model): the model on the kernel
      decode path. ngram_cache_gb: the n-gram row cache of qwen4_exp (the dense family has no n-gram rows).
    - new_state(model, slots, max_seq_len): the state of the slots. It has the hooks kv_tensors() and snap_tensors()
      (engine/model/qwen35.py ModelState).
    - state_bytes(path, slots, max_seq_len, spec): the bytes of that state, and of the drafter's state with spec mtp.
      It builds them on the meta device from config.json alone. fixed_bytes: the rest of the startup memory at the
      defaults.
    - prefill(model, input_ids, state, ...) and prepare_prefill(model): the signatures of engine/model/prefill.py.
      prefill_chunk: the scheduler's tokens per prefill chunk. prefill_prev2: the prefill also takes prev2, the two
      tokens of the slot's history before the chunk (fewer at the start of a slot).
    - host_bytes(ngram_cache_gb): the host memory of the family's caches at startup (qwen4_exp: the n-gram rows).
    - before_replay(model) -> None or a callable (slot, tokens). The scheduler calls the callable right before each
      decode replay, with the token ids of the rows of the slot (qwen4_exp reads the n-gram rows of these tokens).
    - feed_buffer(model) -> None or a callable (B, T) -> tensor: the fixed device buffer that a decode graph passes to
      the model as its third argument (qwen4_exp: the n-gram rows u8 [B, T, 16, 160])."""
    name: str
    load: Callable
    new_state: Callable
    state_bytes: Callable
    fixed_bytes: int
    prefill: Callable
    prepare_prefill: Callable
    defaults: Defaults
    supports: Supports
    before_replay: Callable = _none
    feed_buffer: Callable = _none
    prefill_chunk: int = 2048
    prefill_prev2: bool = False
    host_bytes: Callable = _no_host_bytes


def model_type(path: str) -> str:
    """The model_type of the checkpoint directory `path` ("" when its config.json has none)."""
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f).get("model_type", "")


def _refusal(path: str, t: str) -> NotImplementedError:
    why = NOT_SERVED_YET.get(t, "its modules do not exist yet")
    return NotImplementedError(f"{os.path.basename(path)}: model_type {t!r} ({IN_PROGRESS[t]}) is not served yet: {why} ({PLAN}). "
                               "tools/qwen4exp_reference.py runs the reference on the checkpoint; tests/golden_qwen4exp.py runs the decode path.")


# ------------------------------------------------------------------------------------------------ qwen3_5 (the 27B)
def _qwen35_load(path, decode_weights, log, prep, max_seq_len, ngram_cache_gb=None):
    from engine.runtime.build import load_dense_model
    return load_dense_model(path, decode_weights, log, prep, max_seq_len)


def _qwen35_state_bytes(path: str, slots: int, max_seq_len: int, spec: str) -> int:
    """The KV cache and the GDN state of the slots, and of the drafter with spec mtp, on the meta device. The shapes are
    those of the scheduler: it builds them with model.new_state and MtpState."""
    from engine.model.fast import FastState
    from engine.model.qwen35 import Qwen35Config
    from engine.spec.mtp import MtpState
    cfg = Qwen35Config.from_checkpoint(path)
    states = [FastState(cfg, slots, max_seq_len, "meta")] + ([MtpState(cfg, max_seq_len, "meta", batch=slots)] if spec == "mtp" else [])
    return sum(t.nbytes for s in states for t in s.kv_tensors() + s.snap_tensors())


def _new_state(model, slots: int, max_seq_len: int):
    return model.new_state(slots, max_seq_len)


def _qwen35() -> Family:
    from engine.model.prefill import CHUNK, prefill, prepare_prefill
    from engine.runtime.build import ENGINE_FIXED_BYTES
    return Family(name="qwen3_5", load=_qwen35_load, new_state=_new_state, state_bytes=_qwen35_state_bytes, fixed_bytes=ENGINE_FIXED_BYTES,
                  prefill=prefill, prepare_prefill=prepare_prefill, prefill_chunk=CHUNK,
                  # 524288 tokens per slot: twice the checkpoint's native 262144, YaRN x2 (engine/runtime/build.py load_dense_model)
                  defaults=Defaults(slots=3, max_seq_len=524288, checkpoints=32, spec="mtp", vision=True),
                  supports=Supports(mtp=True, vision=True, yarn=True, int_copies=True))


# ------------------------------------------------------------------------------------------------ qwen4_exp (Flash-Next)
def _module(name: str):
    """The module engine.model.<name>, or None when it does not exist yet. An import error inside the module raises."""
    full = f"engine.model.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        if e.name != full:
            raise
        return None


def prev2_of(history, eos: int) -> tuple[int, int]:
    """The two tokens before the next one of a slot whose history is `history`: its last two, with eos in front of a
    history shorter than two (the n-gram context of a sequence start, engine/weights/ple.py)."""
    last = [int(t) for t in history[-2:]]
    return tuple([eos] * (2 - len(last)) + last)


def attach_feeder(model, ngram_cache_gb: float = NGRAM_CACHE_GB, log=print):
    """model.feeder = a PleFeeder (engine/weights/ple.py) over the n-gram table of the model's PLE layer, with a row
    cache of ngram_cache_gb GB. The prefill (engine/model/prefill_qwen4exp.py) and before_replay read the rows through it."""
    from engine.weights.ple import PleFeeder
    ple = model.ple_layer()
    rows = ngram_cache_rows(ngram_cache_gb)
    model.feeder = PleFeeder(ple.table, cache_rows=rows, params=ple.params, image_token_id=model.cfg.image_token_id)
    log(f"[engine] n-gram rows: {os.path.basename(ple.table.path)}, a host row cache of {rows} rows ({rows * NGRAM_SLOT_BYTES / 1e9:.2f} GB)"
        if rows else f"[engine] n-gram rows: {os.path.basename(ple.table.path)}, no row cache")
    return model


def ngram_before_replay(model):
    """The before_replay hook of qwen4_exp: the n-gram rows of the replay's T tokens (one plain step: T = 1; a verify
    of k drafts: T = k + 1), after the slot's last two tokens, into the model's fixed feed buffer (1, T). The copy
    runs on the current stream, before the replay that reads the buffer; the feeder's pinned buffer stays valid for
    two more calls, and the scheduler waits for each replay."""
    feeder, eos = model.feeder, model.cfg.eos_token_id

    def before_replay(slot, tokens):
        rows = feeder.rows_for(tokens, prev2_of(slot.tokens, eos))
        model.feed_buffer(1, len(tokens)).view(rows.shape).copy_(rows, non_blocking=True)
    return before_replay


def _qwen4exp(path: str) -> Family:
    """The family of Qwen3.8-Flash-Next: the decode model of engine/model/fast_qwen4exp.py (load_fast_qwen4exp,
    FastState4), the prefill of engine/model/prefill_qwen4exp.py in chunks of QWEN4EXP_PREFILL_CHUNK, and the n-gram
    feeder (attach_feeder, ngram_before_replay). A missing module or name raises the refusal of check_served."""
    fast, pf = _module("fast_qwen4exp"), _module("prefill_qwen4exp")
    try:
        load_fast, state_cls, pf_prefill = fast.load_fast_qwen4exp, fast.FastState4, pf.prefill
    except AttributeError:  # a module that does not exist yet (None), or a name that it does not have yet
        raise _refusal(path, "qwen4_exp") from None

    def load(path, decode_weights, log, prep, max_seq_len, ngram_cache_gb=NGRAM_CACHE_GB):
        path, model = load_fast(path, decode_weights, log, prep, max_seq_len)
        return path, attach_feeder(model, NGRAM_CACHE_GB if ngram_cache_gb is None else ngram_cache_gb, log)

    def prefill(model, input_ids, state, chunk=QWEN4EXP_PREFILL_CHUNK, prev2=None, **kw):
        """prefill_qwen4exp.prefill; prev2: the slot's history before the chunk (its last two tokens count)."""
        return pf_prefill(model, input_ids, state, chunk=chunk, prev2=None if prev2 is None else prev2_of(prev2, model.cfg.eos_token_id), **kw)

    def state_bytes(path: str, slots: int, max_seq_len: int, spec: str) -> int:
        from engine.model.qwen4exp import Qwen4ExpConfig
        st = state_cls(Qwen4ExpConfig.from_checkpoint(path), slots, max_seq_len, "meta")
        return sum(t.nbytes for t in st.kv_tensors() + st.snap_tensors())

    return Family(name="qwen4_exp", load=load, new_state=_new_state, state_bytes=state_bytes, fixed_bytes=QWEN4EXP_FIXED_BYTES,
                  prefill=prefill, prepare_prefill=getattr(pf, "prepare_prefill", _none),
                  defaults=Defaults(slots=2, max_seq_len=262144, checkpoints=16, spec="none", vision=False),
                  supports=Supports(mtp=False, vision=False, yarn=False, int_copies=False),
                  before_replay=ngram_before_replay, feed_buffer=lambda model: model.feed_buffer,
                  prefill_chunk=QWEN4EXP_PREFILL_CHUNK, prefill_prev2=True,
                  host_bytes=lambda gb: ngram_cache_rows(NGRAM_CACHE_GB if gb is None else gb) * NGRAM_SLOT_BYTES)


def family_of(path: str) -> Family:
    """The Family of the checkpoint directory `path`, by its model_type. Raises the NotImplementedError of check_served
    for a family whose modules do not exist yet."""
    if model_type(path) == "qwen4_exp":
        return _qwen4exp(path)
    return _qwen35()


def check_served(path: str) -> str:
    """The model_type of the checkpoint at `path`. Raises NotImplementedError for a family in NOT_SERVED_YET, and for a family
    of IN_PROGRESS whose modules do not exist."""
    t = model_type(path)
    if t in NOT_SERVED_YET:
        raise _refusal(path, t)
    if t in IN_PROGRESS:
        family_of(path)
    return t

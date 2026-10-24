"""The model families of the engine, by the model_type of a checkpoint's config.json.

- qwen3_5: the dense Qwen3.5 / 3.6 / 3.8 family (Qwen3.8-27B). The engine serves it: engine/model/qwen35.py, fast.py,
  prefill.py. A checkpoint with another model_type, or with none, also gets this family (the behavior of the engine
  before the dispatch).
- qwen4_exp: Qwen3.8-Flash-Next, the MoE family. The reference model (engine/model/qwen4exp.py), the loader
  (engine/weights/qwen4exp_loader.py), the n-gram table reader (engine/weights/ple.py) and the kernels (csrc/moe.cu,
  moe_prefill.cu, qsa.cu, hc.cu) exist. The decode model on the kernels (engine/model/fast_qwen4exp.py), the prefill
  path (engine/model/prefill_qwen4exp.py) and the server integration are the milestones M1 to M5 of
  docs/history/qwen4exp_plan.md. Until those modules exist, the engine refuses the checkpoint with a clear message,
  instead of a KeyError from the dense config reader.

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
# The families whose serving integration is not finished: check_served refuses them even when their modules exist. Round 2 of
# M1-M4 (docs/history/qwen4exp_plan.md) removes qwen4_exp from this set when the scheduler feeds the n-gram rows and the
# end-to-end check passes.
NOT_SERVED_YET = {"qwen4_exp": "the decode model, the prefill and the kernels exist; the scheduler wiring is in progress"}

# The startup memory of a qwen4_exp engine beyond the slot state (docs/history/qwen4exp_plan.md section 4), at the
# defaults of the family (16 checkpoints, no drafter, no decode copies): the routed experts (67.95e9), the BF16 rest
# (9.88e9), the CUDA context with the allocator, the pinned buffers and the graph pool (5.8e9), and the checkpoint
# ring of 16 snapshots of 115.6e6 bytes (1.85e9). The first start of the decode model must measure it, as for the 27B.
QWEN4EXP_FIXED_BYTES = int(67.95e9 + 9.88e9 + 5.8e9 + 16 * 115.6e6)


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


@dataclasses.dataclass(frozen=True)
class Family:
    """The parts of the engine that differ between the model families.

    - load(path, decode_weights, log, prep, max_seq_len) -> (path, model): the model on the kernel decode path.
    - new_state(model, slots, max_seq_len): the state of the slots. It has the hooks kv_tensors() and snap_tensors()
      (engine/model/qwen35.py ModelState).
    - state_bytes(path, slots, max_seq_len, spec): the bytes of that state, and of the drafter's state with spec mtp.
      It builds them on the meta device from config.json alone. fixed_bytes: the rest of the startup memory at the
      defaults.
    - prefill(model, input_ids, state, ...) and prepare_prefill(model): the signatures of engine/model/prefill.py.
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


def model_type(path: str) -> str:
    """The model_type of the checkpoint directory `path` ("" when its config.json has none)."""
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f).get("model_type", "")


def _refusal(path: str, t: str) -> NotImplementedError:
    why = NOT_SERVED_YET.get(t, "its modules do not exist yet")
    return NotImplementedError(f"{os.path.basename(path)}: model_type {t!r} ({IN_PROGRESS[t]}) is not served yet: {why} ({PLAN}). "
                               "tools/qwen4exp_reference.py runs the reference on the checkpoint; tests/golden_qwen4exp.py runs the decode path.")


# ------------------------------------------------------------------------------------------------ qwen3_5 (the 27B)
def _qwen35_load(path, decode_weights, log, prep, max_seq_len):
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
    from engine.model.prefill import prefill, prepare_prefill
    from engine.runtime.build import ENGINE_FIXED_BYTES
    return Family(name="qwen3_5", load=_qwen35_load, new_state=_new_state, state_bytes=_qwen35_state_bytes, fixed_bytes=ENGINE_FIXED_BYTES,
                  prefill=prefill, prepare_prefill=prepare_prefill,
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


def _qwen4exp(path: str) -> Family:
    """The family of Qwen3.8-Flash-Next: the decode model of engine/model/fast_qwen4exp.py (load_fast_qwen4exp,
    FastState4) and the prefill of engine/model/prefill_qwen4exp.py. A missing module or name raises the refusal of
    check_served."""
    fast, pf = _module("fast_qwen4exp"), _module("prefill_qwen4exp")
    try:
        load, state_cls, prefill = fast.load_fast_qwen4exp, fast.FastState4, pf.prefill
    except AttributeError:  # a module that does not exist yet (None), or a name that it does not have yet
        raise _refusal(path, "qwen4_exp") from None

    def state_bytes(path: str, slots: int, max_seq_len: int, spec: str) -> int:
        from engine.model.qwen4exp import Qwen4ExpConfig
        st = state_cls(Qwen4ExpConfig.from_checkpoint(path), slots, max_seq_len, "meta")
        return sum(t.nbytes for t in st.kv_tensors() + st.snap_tensors())

    return Family(name="qwen4_exp", load=load, new_state=_new_state, state_bytes=state_bytes, fixed_bytes=QWEN4EXP_FIXED_BYTES,
                  prefill=prefill, prepare_prefill=getattr(pf, "prepare_prefill", _none),
                  defaults=Defaults(slots=2, max_seq_len=262144, checkpoints=16, spec="none", vision=False),
                  supports=Supports(mtp=False, vision=False, yarn=False, int_copies=False),
                  # round 2: before_replay becomes a closure over the n-gram feeder (engine/weights/ple.py PleFeeder)
                  before_replay=lambda model: getattr(model, "before_replay", None),
                  feed_buffer=lambda model: getattr(model, "feed_buffer", None))


def family_of(path: str) -> Family:
    """The Family of the checkpoint directory `path`, by its model_type. Raises the NotImplementedError of check_served
    for a family whose modules do not exist yet."""
    if model_type(path) == "qwen4_exp":
        return _qwen4exp(path)
    return _qwen35()


def check_served(path: str) -> str:
    """The model_type of the checkpoint at `path`. Raises NotImplementedError for a family in NOT_SERVED_YET, and for a family
    whose support is in progress and whose modules do not exist yet."""
    t = model_type(path)
    if t in NOT_SERVED_YET:
        raise _refusal(path, t)
    if t in IN_PROGRESS:
        family_of(path)
    return t

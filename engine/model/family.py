"""The model families of the engine, by the model_type of a checkpoint's config.json.

- qwen3_5: the dense Qwen3.5 / 3.6 / 3.8 family (Qwen3.8-27B). The engine serves it: engine/model/qwen35.py, fast.py,
  prefill.py.
- qwen4_exp: Qwen3.8-Flash-Next, the MoE family. The reference model (engine/model/qwen4exp.py), the loader
  (engine/weights/qwen4exp_loader.py), the n-gram table reader (engine/weights/ple.py) and the kernels (csrc/moe.cu,
  moe_prefill.cu, qsa.cu, hc.cu) exist. The decode model on the kernels, the prefill path and the server integration are the
  milestones M1 to M5 of docs/history/qwen4exp_plan.md. Until then the engine refuses the checkpoint with a clear message,
  instead of a KeyError from the dense config reader.

check_served(path) runs before the memory check of the build, so the refusal costs nothing.
"""
from __future__ import annotations

import json
import os

PLAN = "docs/history/qwen4exp_plan.md"
IN_PROGRESS = {"qwen4_exp": "Qwen3.8-Flash-Next"}


def model_type(path: str) -> str:
    """The model_type of the checkpoint directory `path` ("" when its config.json has none)."""
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f).get("model_type", "")


def check_served(path: str) -> str:
    """The model_type of the checkpoint at `path`. Raises NotImplementedError for a family whose support is in progress."""
    t = model_type(path)
    if t in IN_PROGRESS:
        raise NotImplementedError(f"{os.path.basename(path)}: model_type {t!r} ({IN_PROGRESS[t]}) is not served yet. The reference model, the "
                                  f"loader and the kernels exist; the decode model on the kernels is the next milestone ({PLAN}). "
                                  "tools/qwen4exp_reference.py runs the reference on the checkpoint.")
    return t

"""The family check of the build (engine/model/family.py): the dense family passes, the in-progress MoE family is refused
with a message that names the plan. Run: uv run pytest tests/test_family.py -q"""
import json

import pytest

from engine.model.family import check_served, model_type
from engine.runtime import build


def write_config(tmp_path, mt):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": mt}))
    return str(tmp_path)


def test_dense_family_passes(tmp_path):
    assert check_served(write_config(tmp_path, "qwen3_5")) == "qwen3_5"


def test_unknown_type_is_not_refused_here(tmp_path):
    assert model_type(write_config(tmp_path, "")) == ""
    assert check_served(str(tmp_path)) == ""


def test_moe_family_is_refused_with_the_plan(tmp_path):
    path = write_config(tmp_path, "qwen4_exp")
    with pytest.raises(NotImplementedError, match="qwen4_exp.*qwen4exp_plan.md"):
        check_served(path)
    with pytest.raises(NotImplementedError, match="not served yet"):
        build.build_engine(path, selftest=False)
    with pytest.raises(NotImplementedError, match="not served yet"):
        build.load_model(path)

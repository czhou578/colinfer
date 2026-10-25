"""The prepared directory (engine/runtime/prepared.py) and the tool that fills it (tools/prepare.py), on files in a
temporary directory: the manifest round trip and its errors, and each way an artifact gets there (kept, taken from
where its tool writes it, made by that tool, skipped). The tools run as a stand-in that writes the file it is asked for."""
import json
import os

import pytest

from engine.runtime import prepared as pr
from tools import prepare as tool


def touch(p, size=8):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"x" * size)
    return p


# ---------------------------------------------------------------------------------------------------------- the manifest
def test_manifest_round_trip_and_errors(tmp_path):
    ckpt = str(tmp_path / "snap")
    d = str(tmp_path / "prepared")
    assert pr.read(ckpt, d) is None  # no manifest: the checkpoint alone
    os.makedirs(d)
    touch(f"{d}/{pr.DECODE_FILES[0]}")
    touch(f"{d}/{pr.LOWRANK_FILE}")
    mp = pr.write(d, "snap", [pr.DECODE_FILES[0]], None, pr.LOWRANK_FILE)
    p = pr.read(ckpt, d)
    assert p.dir == d and p.checkpoint == "snap" and p.decode == (f"{d}/{pr.DECODE_FILES[0]}",) and p.drafter is None
    assert p.lowrank == f"{d}/{pr.LOWRANK_FILE}" and p.made
    with pytest.raises(ValueError, match="not for 'other'"):
        pr.read(str(tmp_path / "other"), d)
    os.remove(f"{d}/{pr.LOWRANK_FILE}")
    with pytest.raises(ValueError, match="missing; run tools/prepare.py"):
        pr.read(ckpt, d)
    with open(mp, "w") as f:
        json.dump({"version": 99, "checkpoint": "snap"}, f)
    with pytest.raises(ValueError, match="version 1"):
        pr.read(ckpt, d)
    with open(mp, "w") as f:
        json.dump({"version": 1, "checkpoint": "snap", "decode": "no"}, f)
    with pytest.raises(ValueError, match="list"):
        pr.read(ckpt, d)
    assert pr.prepared_dir("/x/y/snap/") == os.path.join(pr.PREPARED_DIR, "snap")


# -------------------------------------------------------------------------------------------------------------- the tool
class Tools:
    """The stand-in for the two tools: it writes the file after --out and remembers each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        touch(argv[argv.index("--out") + 1], size=len(self.calls))

    def names(self):
        return [os.path.basename(a[1]) for a in self.calls]


@pytest.fixture
def env(tmp_path):
    ckpt = str(tmp_path / "hf" / "snap")
    os.makedirs(ckpt)
    legacy = {"decode": [str(tmp_path / "requant" / n) for n in pr.DECODE_FILES], "drafter": str(tmp_path / "drafter" / pr.DRAFTER_FILE),
              "lowrank": str(tmp_path / "drafter" / pr.LOWRANK_FILE)}
    out = str(tmp_path / "prepared")
    logs = []

    def go(tools, **kw):
        args = dict(legacy_decode=legacy["decode"], legacy_drafter=legacy["drafter"], legacy_lowrank=legacy["lowrank"], run=tools, log=logs.append)
        args.update(kw)
        return tool.prepare(ckpt, out, **args)
    return ckpt, out, legacy, go, logs


def test_takes_the_files_the_tools_wrote(env):
    ckpt, out, legacy, go, logs = env
    for p in legacy["decode"] + [legacy["drafter"], legacy["lowrank"]]:
        touch(p)
    tools = Tools()
    p = go(tools)
    assert not tools.calls  # nothing to make
    assert p.decode == tuple(f"{out}/{n}" for n in pr.DECODE_FILES)
    assert p.drafter == f"{out}/{pr.DRAFTER_FILE}" and p.lowrank == f"{out}/{pr.LOWRANK_FILE}"
    assert all(os.path.samefile(a, b) for a, b in zip(legacy["decode"], p.decode))  # hard links: the same size and mtime
    assert os.path.samefile(legacy["drafter"], p.drafter) and os.path.samefile(legacy["lowrank"], p.lowrank)
    assert sum("linked from" in m for m in logs) == 4
    p2 = go(tools)  # a second run keeps everything
    assert not tools.calls and p2 == pr.read(ckpt, out) and sum("kept" in m for m in logs) == 4


def test_makes_what_is_missing_with_the_tools(env):
    ckpt, out, legacy, go, logs = env
    touch(legacy["decode"][0])  # the INT6 file exists, the INT5 one and the rest do not
    tools = Tools()
    p = go(tools)
    assert tools.names() == ["int6_requant.py", "lowrank_draft_head.py"]
    int5 = tools.calls[0]
    assert int5[int5.index("--bits") + 1] == "5" and int5[int5.index("--filter") + 1] == "linear_attn"
    assert int5[int5.index("--nvfp4") + 1] == ckpt and int5[int5.index("--out") + 1] == f"{out}/{pr.DECODE_FILES[1]}"
    lr = tools.calls[1]
    assert lr[lr.index("--weights") + 1] == "none" and lr[lr.index("--out") + 1] == f"{out}/{pr.LOWRANK_FILE}"
    assert len(p.decode) == 2 and p.drafter is None and p.lowrank is not None


def test_the_manifest_is_written_before_the_low_rank_tool_runs(env):
    ckpt, out, legacy, go, logs = env
    for p in legacy["decode"] + [legacy["drafter"]]:
        touch(p)
    seen = {}

    class Peek(Tools):
        def __call__(self, argv):
            seen["during"] = pr.read(ckpt, out)  # what the tool's own load_model() sees
            super().__call__(argv)
    tools = Peek()
    p = go(tools)
    assert tools.names() == ["lowrank_draft_head.py"] and tools.calls[0][tools.calls[0].index("--weights") + 1] == f"{out}/{pr.DRAFTER_FILE}"
    assert len(seen["during"].decode) == 2 and seen["during"].drafter == p.drafter and seen["during"].lowrank is None
    assert p.lowrank == f"{out}/{pr.LOWRANK_FILE}"


def test_skips(env):
    ckpt, out, legacy, go, logs = env
    tools = Tools()
    p = go(tools, skip_int=True, skip_lowrank=True)
    assert not tools.calls and p.decode == () and p.drafter is None and p.lowrank is None
    assert sum("skipped" in m for m in logs) == 3
    assert pr.read(ckpt, out) == p


def test_a_new_drafter_replaces_the_old_and_remakes_the_low_rank_head(env):
    ckpt, out, legacy, go, logs = env
    for p in legacy["decode"] + [legacy["drafter"], legacy["lowrank"]]:
        touch(p)
    tools = Tools()
    go(tools)
    old_lowrank = os.stat(f"{out}/{pr.LOWRANK_FILE}")
    new = touch(str(legacy["drafter"]) + ".new", size=99)
    p = go(tools, drafter=new)
    assert tools.names() == ["lowrank_draft_head.py"]  # made again, on the new drafter
    assert tools.calls[0][tools.calls[0].index("--weights") + 1] == p.drafter and os.path.samefile(new, p.drafter)
    assert os.stat(p.lowrank).st_size != old_lowrank.st_size and not os.path.samefile(legacy["lowrank"], p.lowrank)
    assert any("a new drafter" in m for m in logs)
    p2 = go(tools, drafter=new)  # the same drafter again: everything kept
    assert tools.names() == ["lowrank_draft_head.py"] and p2.lowrank == p.lowrank


def test_copies_when_a_link_is_refused(env, monkeypatch):
    ckpt, out, legacy, go, logs = env
    touch(legacy["lowrank"])

    def no_link(src, dst):
        raise OSError("cross-device")
    monkeypatch.setattr(tool.os, "link", no_link)
    p = go(Tools(), skip_int=True)
    assert os.path.exists(p.lowrank) and not os.path.samefile(legacy["lowrank"], p.lowrank) and tool.same(legacy["lowrank"], p.lowrank)
    assert any("copied from" in m for m in logs)


# ---------------------------------------------------------------------------------------------------------- qwen4_exp
def test_manifest_family_round_trip(tmp_path):
    """The family key: written for qwen4_exp, absent (empty) in the 27B's manifests, which read as before."""
    ckpt, d = str(tmp_path / "snap"), str(tmp_path / "prepared")
    os.makedirs(d)
    files = pr.decode4_files()
    assert files == [pr.decode4_file(c, pr.DECODE4_FORMATS[c]) for c in pr.DECODE4_CLASSES if pr.DECODE4_FORMATS[c] != "bf16"]
    assert pr.decode4_files({c: "int6" for c in pr.DECODE4_CLASSES})[0] == "decode4_gdn_in_int6.safetensors"
    assert pr.decode4_files({**pr.DECODE4_FORMATS, "lm_head": "bf16"}) == [f for f in files if "lm_head" not in f]
    for f in files:
        touch(f"{d}/{f}")
    pr.write(d, "snap", files, None, None, family="qwen4_exp")
    p = pr.read(ckpt, d)
    assert p.family == "qwen4_exp" and p.decode == tuple(f"{d}/{f}" for f in files) and p.drafter is None and p.lowrank is None
    pr.write(d, "snap", files[:1], None, None)  # the 27B's form: no family key
    with open(f"{d}/{pr.MANIFEST}") as f:
        assert "family" not in json.load(f)
    assert pr.read(ckpt, d).family == ""


class Requant:
    """The stand-in for tools/requant_qwen4exp.py: it writes the files of the classes after --classes into --out."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        out = argv[argv.index("--out") + 1]
        formats = dict(x.split("=") for x in argv[argv.index("--formats") + 1].split(","))
        for c in argv[argv.index("--classes") + 1].split(","):
            touch(f"{out}/{pr.decode4_file(c, formats[c])}")


@pytest.fixture
def env4(tmp_path):
    ckpt = str(tmp_path / "hf" / "snap4")
    os.makedirs(ckpt)
    with open(f"{ckpt}/config.json", "w") as f:
        json.dump({"model_type": "qwen4_exp", "text_config": {}}, f)
    return ckpt, str(tmp_path / "prepared4")


def test_qwen4exp_dispatch_makes_the_copies(env4):
    """A qwen4_exp checkpoint runs tools/requant_qwen4exp.py once for the missing classes and skips the 27B's steps."""
    ckpt, out = env4
    assert tool.model_type(ckpt) == "qwen4_exp"
    os.makedirs(out)
    have = [c for c in pr.DECODE4_CLASSES if pr.DECODE4_FORMATS[c] != "bf16"]
    touch(f"{out}/{pr.decode4_file(have[0], pr.DECODE4_FORMATS[have[0]])}")  # made before: kept
    tools, logs = Requant(), []
    p = tool.prepare(ckpt, out, run=tools, log=logs.append, legacy_decode=["/nonexistent"], legacy_drafter="/nonexistent")
    assert len(tools.calls) == 1 and os.path.basename(tools.calls[0][1]) == "requant_qwen4exp.py"
    argv = tools.calls[0]
    assert argv[argv.index("--model") + 1] == ckpt and argv[argv.index("--out") + 1] == out
    assert argv[argv.index("--classes") + 1].split(",") == have[1:]
    assert p.family == "qwen4_exp" and p.decode == tuple(f"{out}/{f}" for f in pr.decode4_files())
    assert p.drafter is None and p.lowrank is None and any("not for this family" in m for m in logs)
    assert not any("int6_requant" in " ".join(a) for a in tools.calls)
    tools2 = Requant()
    assert tool.prepare(ckpt, out, run=tools2, log=logs.append) == pr.read(ckpt, out) and not tools2.calls  # all kept


def test_qwen4exp_dispatch_formats_and_skip(env4):
    ckpt, out = env4
    tools, logs = Requant(), []
    fm = {**pr.DECODE4_FORMATS, "gdn_in": "int5", "lm_head": "nvfp4", "hc_up": "bf16"}
    p = tool.prepare4(ckpt, out, formats=fm, run=tools, log=logs.append)
    spec = tools.calls[0][tools.calls[0].index("--formats") + 1]
    assert "lm_head=nvfp4" in spec and "gdn_in=int5" in spec and "hc_up" not in spec
    assert [os.path.basename(f) for f in p.decode] == pr.decode4_files(fm) and any("hc_up: bf16" in m for m in logs)
    p2 = tool.prepare4(ckpt, str(out) + "_skip", skip_int=True, run=Requant(), log=logs.append)
    assert p2.decode == () and p2.family == "qwen4_exp"


def test_qwen4exp_main_dispatch(env4, monkeypatch):
    """main() on a qwen4_exp checkpoint: --formats reaches the tool, and the 27B's legacy paths are not consulted."""
    ckpt, out = env4
    calls = []
    def fake(path, o, skip_int, formats):
        calls.append((path, o, skip_int, formats))
        return pr.Prepared(o, "snap4", (), None, None)
    monkeypatch.setattr(tool, "prepare4", fake)
    monkeypatch.setattr("engine.weights.loader.resolve", lambda m: ckpt)
    assert tool.main(["--model", ckpt, "--out", out, "--formats", "lm_head=nvfp4", "--skip-int"]) == 0
    assert calls[0][:3] == (ckpt, out, True) and calls[0][3]["lm_head"] == "nvfp4" and calls[0][3]["attn"] == pr.DECODE4_FORMATS["attn"]

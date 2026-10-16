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

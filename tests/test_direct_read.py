"""The direct safetensors reader (engine/weights/direct.py) against safetensors itself: every dtype the checkpoints
use, odd shapes, tensors across chunk boundaries, a subset of names, empty tensors, and a file too small for a chunk."""
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import engine.weights.direct as D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU (pinned buffers)")


def tensors(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {"a.weight": r(3, 1001).bfloat16(), "b.weight_scale": r(1).float(), "c.qweight": (r(37, 5000) * 50).to(torch.uint8),
            "d.weight": r(513, 129).to(torch.float8_e4m3fn), "e.ids": torch.randint(0, 2 ** 40, (777,), generator=g), "f.flag": r(5) > 0,
            "g.big": r(300, 4096).bfloat16(), "h.empty": torch.empty(0, 7), "i.weight": r(129).half(),
            "j.weight_scale_2": torch.tensor(1.5), "k.input_scale": r(()).bfloat16()}  # scalars, as the checkpoint's global scales


def same(path, got, want_names):
    with safe_open(path, framework="pt", device="cuda") as sf:
        assert sorted(got) == sorted(want_names)
        for k in want_names:
            t = sf.get_tensor(k)
            assert got[k].dtype == t.dtype and got[k].shape == t.shape and got[k].is_cuda, k
            assert torch.equal(got[k].reshape(-1).view(torch.uint8), t.reshape(-1).view(torch.uint8)), k


@pytest.mark.parametrize("chunk,buffers", [(D.CHUNK, D.BUFFERS), (16384, 2), (4096, 1)])
def test_reads_match_safetensors(tmp_path, monkeypatch, chunk, buffers):
    monkeypatch.setattr(D, "CHUNK", chunk)
    monkeypatch.setattr(D, "BUFFERS", buffers)
    monkeypatch.setattr(D, "GAP", 20000)
    path = str(tmp_path / "t.safetensors")
    ts = tensors(0)
    save_file(ts, path, metadata={"x": "y"})
    assert sorted(D.tensor_names(path)) == sorted(ts)
    same(path, D.read_tensors(path), list(ts))
    sub = ["c.qweight", "a.weight", "h.empty", "i.weight"]  # out of file order, with gaps larger than GAP between them
    same(path, D.read_tensors(path, sub), sub)
    with pytest.raises(KeyError):
        D.read_tensors(path, ["nope"])


def test_small_file_and_no_direct_io(tmp_path, monkeypatch):
    path = str(tmp_path / "s.safetensors")
    ts = {"w": torch.arange(10, dtype=torch.int32), "z": torch.zeros(0)}
    save_file(ts, path)
    same(path, D.read_tensors(path), list(ts))
    monkeypatch.setattr(D, "_open", lambda p: (os.open(p, os.O_RDONLY), False))  # the fallback: buffered reads
    same(path, D.read_tensors(path), list(ts))
    big = str(tmp_path / "b.safetensors")
    save_file(tensors(1), big)
    same(big, D.read_tensors(big), list(tensors(1)))

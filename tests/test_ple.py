"""engine/weights/ple.py: NgramTable over a synthetic n-gram file (shards out of index order, rows across page
boundaries, the last row, repeated ids, the buffered fallback) and the hash parameters of a second PLE layer against
transformers' own functions. The model test (tests/test_qwen4exp_small.py) checks the ids against transformers.
Run: ./py -m pytest tests/test_ple.py -q
"""
import os

import numpy as np
import pytest
import torch

from engine.model.qwen4exp import Qwen4ExpConfig
from engine.weights.ple import PAGE, NgramTable, ngram_params, ngram_rows
from tests.qwen4exp_ckpt import ngram_shards, tiny_table_fp8, write_safetensors

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
ROWS, DIM, SHARDS, ORDER = 4000, 160, 4, [2, 0, 3, 1]


@pytest.fixture(scope="module")
def table_file(tmp_path_factory):
    table, scale = tiny_table_fp8(ROWS, DIM, seed=7)
    path = str(tmp_path_factory.mktemp("ple") / "model-fp8-mtp-ple.safetensors")
    items = [("mtp.decoy.weight", torch.randn(33, 7).to(torch.bfloat16)), (PREFIX + ".weight_scale", scale)]
    items += ngram_shards(table, SHARDS, ORDER, PREFIX)
    write_safetensors(path, items, {"format": "pt"}, pad=24)  # 24 bytes before each tensor: offsets off the row grid
    return path, table, scale


def test_header_and_offsets(table_file):
    path, table, scale = table_file
    t = NgramTable(path, PREFIX, threads=4)
    assert (t.n_shards, t.rows_per_shard, t.dim, t.n_rows) == (SHARDS, ROWS // SHARDS, DIM, ROWS)
    assert t.scale == float(scale.float()) and t.scale_tensor.dtype == torch.bfloat16
    assert list(np.argsort(t.shard_offsets)) == ORDER  # the file order, taken from the header
    with pytest.raises(IndexError):
        t.offsets(np.array([ROWS]))
    t.close()


@pytest.mark.parametrize("direct", [True, False])
def test_rows_equal_the_table(table_file, monkeypatch, direct):
    path, table, scale = table_file
    if not direct:
        monkeypatch.setattr("engine.weights.ple._open", lambda p: (os.open(p, os.O_RDONLY), False))
    t = NgramTable(path, PREFIX, threads=3)
    assert t.direct == direct
    raw = table.view(torch.uint8).numpy()
    offs = t.offsets(np.arange(ROWS))
    straddle = np.flatnonzero(offs // PAGE != (offs + DIM - 1) // PAGE)
    file_last = int(np.argmax(offs))  # the row that ends the file
    assert straddle.size and offs[file_last] + DIM == os.path.getsize(path)
    g = np.random.default_rng(0)
    ids = np.concatenate([[0, ROWS - 1, file_last, straddle[0], straddle[-1], 999, 1000, 1001], g.integers(0, ROWS, 300),
                          [5, 5, 5, file_last]])  # repeated ids, shard boundaries
    got = t.rows(ids)
    assert got.shape == (ids.size, DIM) and np.array_equal(got, raw[ids])
    assert np.array_equal(t.rows(np.arange(ROWS)), raw)  # every row: one run of pages per shard
    assert t.rows(np.array([], np.int64)).shape == (0, DIM)
    out = torch.empty(ids.size, DIM, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
    assert t.rows_into(ids, out) is out and np.array_equal(out.numpy(), raw[ids])
    v = t.dequant(torch.from_numpy(got[:10]), torch.bfloat16)
    assert torch.equal(v, (table[ids[:10]].float() * scale.float()).to(torch.bfloat16))
    t.close()


def test_rows_with_many_threads_and_one_row_per_page(tmp_path):
    """Rows of one page each, read on more threads than runs."""
    table, scale = tiny_table_fp8(64, 4096, seed=1)
    path = str(tmp_path / "t.safetensors")
    write_safetensors(path, [(PREFIX + ".weight_scale", scale)] + ngram_shards(table, 2, [1, 0], PREFIX))
    t = NgramTable(path, PREFIX, threads=64)
    ids = np.array([63, 0, 31, 32, 7])
    assert np.array_equal(t.rows(ids), table.view(torch.uint8).numpy()[ids])
    t.close()


def test_refuses_a_file_without_shards(tmp_path):
    path = str(tmp_path / "t.safetensors")
    write_safetensors(path, [(PREFIX + ".shard_1.weight", torch.zeros(4, 8).to(torch.float8_e4m3fn))])
    with pytest.raises(ValueError, match="shards"):
        NgramTable(path, PREFIX)


def test_second_ple_layer_params_match_transformers():
    """ngram_params of PLE index 1 (heads 16 .. 31 of the prime sequence, other multipliers) against transformers'
    functions, as its Qwen4ExpTextNGramEmbedding computes them."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers, _find_nth_prime_after
    c = Qwen4ExpConfig()
    p = ngram_params(c, 1)
    assert list(p.sizes) == [_find_nth_prime_after(c.ngram_vocab_size_base - 1, 16 + h + 1) for h in range(16)]
    assert list(p.multipliers) == _build_layer_multipliers(c.vocab_size, 3, 1, c.seed).tolist()
    assert p.offsets[0] == 0 and p.padded_rows % 128 == 0


def test_ngram_rows_numpy_shapes_and_refusals():
    p = ngram_params(Qwen4ExpConfig(), 0)
    eos = p.eos
    r1 = ngram_rows(np.array([5, eos, 7]), np.array([eos, eos]), p)
    assert r1.shape == (3, 16) and r1.dtype == np.int64
    r2 = ngram_rows(np.array([[5, eos, 7]]), np.array([[eos, eos]]), p)
    assert np.array_equal(r1, r2[0])
    assert np.array_equal(ngram_rows(np.array([7]), np.array([5, eos]), p), r1[2:])  # the context after an eos is (eos, eos) for the trigram
    assert np.array_equal(ngram_rows(np.array([7]), np.array([eos, eos]), p)[:, 8:], r1[2:, 8:])
    assert (r1 >= 0).all() and (r1 < p.padded_rows).all()
    with pytest.raises(ValueError):
        ngram_rows(np.array([p.vocab_size]), np.array([eos, eos]), p)  # an image placeholder id must become image_token_id first


def test_hotset_lru_curve_equals_a_simulated_lru():
    """tools/ple_hotset.py's stack-distance hit rates equal an LRU cache simulated access by access."""
    from collections import OrderedDict

    from tools.ple_hotset import lru_hit_rates
    acc = np.random.default_rng(3).zipf(1.3, 5000) % 700
    sizes = [1, 5, 50, 300, 1000]
    for s, h in zip(sizes, lru_hit_rates(acc, sizes)):
        cache, hits = OrderedDict(), 0
        for r in acc.tolist():
            if r in cache:
                hits += 1
                cache.move_to_end(r)
            else:
                cache[r] = None
                if len(cache) > s:
                    cache.popitem(last=False)
        assert h == hits / acc.size, s

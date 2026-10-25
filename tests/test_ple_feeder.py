"""engine/weights/ple.py PleFeeder and the row cache of csrc/ple_io.cpp, over a synthetic n-gram file with small hash
parameters: the rows equal NgramTable.rows of the ids, with and without the cache, through evictions; the cache hits
after a warm-up; the prefetch path; the eos rule and the image ids. The feeder has a worker thread: run this file
three times under -X dev after a change (.claude/rules/sanitizers.md).
Run: uv run pytest tests/test_ple_feeder.py -q"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engine.weights.ple import NgramTable, PleFeeder, gather_threads, ngram_params, ngram_rows
from tests.qwen4exp_ckpt import ngram_shards, tiny_table_fp8, write_safetensors

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
VOCAB, EOS, IMAGE = 1000, 998, 997


@pytest.fixture(scope="module")
def params():
    cfg = SimpleNamespace(ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=300, make_ngram_vocab_size_divisible_by=128, vocab_size=VOCAB,
                          seed=1234, eos_token_id=EOS)
    return ngram_params(cfg, 0)


@pytest.fixture(scope="module")
def table_file(tmp_path_factory, params):
    rows = params.padded_rows
    table, scale = tiny_table_fp8(rows, 160, seed=3)
    path = str(tmp_path_factory.mktemp("ple") / "model-fp8-mtp-ple.safetensors")
    write_safetensors(path, [(PREFIX + ".weight_scale", scale)] + ngram_shards(table, 4, [1, 3, 0, 2], PREFIX), pad=40)
    return path, table.view(torch.uint8).numpy()


def _tokens(n, seed, eos_every=0, images=0):
    g = np.random.default_rng(seed)
    t = g.integers(0, 900, n)
    if eos_every:
        t[eos_every::eos_every] = EOS
    if images:
        t[g.choice(n, images, replace=False)] = VOCAB + g.integers(0, 50, images)  # engine image placeholders
    return t


def _want(raw, params, tokens, prev2):
    t = np.where(tokens >= VOCAB, IMAGE, tokens)
    c = np.where(np.asarray(prev2) >= VOCAB, IMAGE, prev2)
    ids = ngram_rows(t, c, params)
    return raw[ids]  # [T, 16, 160]


@pytest.mark.parametrize("cache_rows", [0, 50, 100000])
@pytest.mark.parametrize("direct", [True, False])
def test_rows_equal_the_table(table_file, params, monkeypatch, cache_rows, direct):
    path, raw = table_file
    if not direct:
        monkeypatch.setattr("engine.weights.ple._open", lambda p: (os.open(p, os.O_RDONLY), False))
    t = NgramTable(path, PREFIX)
    f = PleFeeder(t, cache_rows, params=params, image_token_id=IMAGE)
    for i, (n, ctx) in enumerate([(1, (EOS, EOS)), (300, (5, 7)), (2048, (EOS, 3)), (17, (1200, 4))]):  # 1200: an image id in the context
        tok = _tokens(n, i, eos_every=13, images=3 if n > 10 else 0)
        got = f.rows_for(tok, ctx)
        assert got.shape == (n, 16, 160) and got.dtype == torch.uint8
        assert (not torch.cuda.is_available()) or got.is_pinned()
        assert np.array_equal(got.numpy(), _want(raw, params, tok, ctx)), i
    f.close()
    t.close()


def test_cache_hits_after_warmup(table_file, params):
    path, raw = table_file
    t = NgramTable(path, PREFIX)
    f = PleFeeder(t, 100000, params=params, image_token_id=IMAGE)
    tok = _tokens(1000, 5)
    a = f.rows_for(tok, (EOS, EOS)).clone()
    hits, misses, held, cap = f.stats(reset=True)
    n_unique = np.unique(f.ids(tok, (EOS, EOS))).size
    assert hits == 0 and misses == n_unique and held == n_unique and cap >= 100000
    read = f.gather_rows
    b = f.rows_for(tok, (EOS, EOS))
    assert torch.equal(a, b)
    assert f.stats() == (n_unique, 0, n_unique, cap) and f.gather_rows == read  # all hits, no read
    f.close()
    t.close()


def test_small_cache_evicts_and_stays_exact(table_file, params):
    path, raw = table_file
    t = NgramTable(path, PREFIX)
    f = PleFeeder(t, 16, params=params, image_token_id=IMAGE)  # two sets of 8 ways
    for i in range(5):
        tok = _tokens(200, 10 + i)
        assert np.array_equal(f.rows_for(tok, (EOS, EOS)).numpy(), _want(raw, params, tok, (EOS, EOS)))
    assert f.stats()[2] == 16
    f.close()
    t.close()


def test_prefetch(table_file, params):
    path, raw = table_file
    t = NgramTable(path, PREFIX)
    f = PleFeeder(t, 100000, params=params, image_token_id=IMAGE, buffers=4)
    a, b, c = _tokens(500, 21), _tokens(700, 22), _tokens(300, 23)
    f.prefetch(a, (EOS, EOS))
    got = f.rows_for(a, (EOS, EOS))  # waits for the prefetch, returns its rows
    assert np.array_equal(got.numpy(), _want(raw, params, a, (EOS, EOS)))
    assert f._pending is None
    f.prefetch(b, (a[-2], a[-1]))
    got_c = f.rows_for(c, (EOS, EOS))  # other tokens: computed now, the prefetch of b stays pending
    assert np.array_equal(got_c.numpy(), _want(raw, params, c, (EOS, EOS)))
    assert f._pending is not None
    got_b = f.rows_for(b, (a[-2], a[-1]))
    assert np.array_equal(got_b.numpy(), _want(raw, params, b, (a[-2], a[-1])))
    assert np.array_equal(got_c.numpy(), _want(raw, params, c, (EOS, EOS)))  # still valid: another buffer of the ring
    for i in range(6):  # prefetch the next chunk while the current one is in use, as the scheduler does
        nxt = _tokens(256, 30 + i)
        f.prefetch(nxt, (EOS, EOS))
        assert np.array_equal(f.rows_for(nxt, (EOS, EOS)).numpy(), _want(raw, params, nxt, (EOS, EOS)))
    f.close()
    t.close()


def test_eos_rule_and_image_ids(params, table_file):
    path, _ = table_file
    t = NgramTable(path, PREFIX)
    f = PleFeeder(t, 0, params=params, image_token_id=IMAGE)
    ids = f.ids(np.array([5, EOS, 7, 8]), (3, 4))
    # the token after eos: its bigram context is eos, its trigram context (eos, eos)
    assert np.array_equal(ids[2], ngram_rows(np.array([7]), np.array([EOS, EOS]), params)[0])
    assert np.array_equal(ids[2, :8], ngram_rows(np.array([7]), np.array([5, EOS]), params)[0, :8])
    # the token two after eos: its trigram context is (eos, 7)
    assert np.array_equal(ids[3], ngram_rows(np.array([8]), np.array([EOS, 7]), params)[0])
    # image placeholders (>= vocab) hash as the image token, in the tokens and in the context
    assert np.array_equal(f.ids(np.array([VOCAB + 5, 9]), (VOCAB + 1, 2)), ngram_rows(np.array([IMAGE, 9]), np.array([IMAGE, 2]), params))
    f.close()
    t.close()


def test_gather_threads():
    assert gather_threads(1) == 1 and gather_threads(64) == 2 and gather_threads(32768) == 64
    assert all(gather_threads(n) <= gather_threads(n + 1) for n in range(0, 10000, 7))


def test_cache_binding_checks():
    from engine.kernels import ops
    c = ops().ple_cache_new(100)
    with pytest.raises(RuntimeError):
        ops().ple_cache_lookup(c, torch.zeros(3, dtype=torch.int32), torch.zeros(3, 160, dtype=torch.uint8))
    with pytest.raises(RuntimeError):
        ops().ple_cache_insert(c, torch.zeros(3, dtype=torch.int64), torch.zeros(3, 128, dtype=torch.uint8))
    ids = torch.tensor([-1, 4, 4], dtype=torch.int64)
    ops().ple_cache_insert(c, ids, torch.full((3, 160), 9, dtype=torch.uint8))
    out = torch.zeros(3, 160, dtype=torch.uint8)
    assert ops().ple_cache_lookup(c, ids, out).tolist() == [1, 0, 0]  # a negative id is never cached
    assert ops().ple_cache_stats(c)[2] == 1


def test_out_buffer_and_staging(table_file, params):
    path, raw = table_file
    t = NgramTable(path, PREFIX)
    f = PleFeeder(t, 1000, params=params, image_token_id=IMAGE)
    tok = _tokens(16, 40)
    out = torch.empty(16, 16, 160, dtype=torch.uint8)
    assert f.rows_for(tok, (EOS, EOS), out=out) is out
    assert np.array_equal(out.numpy(), _want(raw, params, tok, (EOS, EOS)))
    with pytest.raises(ValueError):
        f.rows_for(tok, (EOS, EOS), out=torch.empty(15, 16, 160, dtype=torch.uint8))
    b = f.staging(0, 300)
    assert b.shape == (300, 16, 160) and (not torch.cuda.is_available() or b.is_pinned())
    nxt = _tokens(300, 41)
    f.prefetch(nxt, (EOS, EOS), out=b)
    got = f.rows_for(nxt, (EOS, EOS), out=f.staging(1, 300))  # the prefetch's tensor: b
    assert got.data_ptr() == b.data_ptr() and np.array_equal(got.numpy(), _want(raw, params, nxt, (EOS, EOS)))
    f.release(0)
    assert f.staging(0, 200).data_ptr() == b.data_ptr()  # the same buffer once its copy is done
    f.close()
    t.close()

"""The disk cache of conversations (engine/runtime/sessions.py) under the scheduler, on the stub model of
tests/test_scheduler.py: its next token depends on the whole history, through the GDN state and the KV cache that the
store writes and reads back. Thus an output equals the reference continuation only if a restore gave the slot exactly
the state of its prefix."""
import os
import random
import threading
import time

import pytest
import torch

import engine.runtime.scheduler as S
from engine.model.fast import FastState
from engine.runtime.sessions import SessionStore, _shapes
from tests.test_scheduler import BND, CFG, Checked, ExtModel, L, StubModel, chat, ext_prefill, ext_reference, prompt, reference, stub_prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
BLOCK = 64


@pytest.fixture
def make(monkeypatch, tmp_path):
    """Schedulers with a store in tmp_path. A writer that copies while a graph is captured would invalidate the capture
    (the server attaches the store after its captures), so each new scheduler, and the end of the test, waits for the
    writers."""
    monkeypatch.setattr(S, "prefill", stub_prefill)
    stores = []

    def make(budget=10**9, **kw):
        for st in stores:
            st.flush()
        kw = dict(n_slots=3, max_seq_len=L, n_checkpoints=8, prefill_chunk=64, ckpt_interval=10**6, boundary_token=BND) | kw
        sched = Checked(StubModel(), **kw)
        with torch.inference_mode():  # as the server's engine thread builds it
            sched.store = SessionStore(sched.state, None, {"model": "stub"}, budget, str(tmp_path), block=BLOCK, min_tokens=BLOCK,
                                       log=lambda *a: None)
        sched.store.min_gain = 16
        stores.append(sched.store)
        return sched
    yield make
    for st in stores:
        st.flush()


def run(sched, p, n=20, **kw):
    r = sched.run([S.Request(p, max_new_tokens=n, **kw)])[0]
    assert r.output == reference(p, n)
    return r


def test_an_evicted_conversation_restores_from_disk(make):
    sched, rng = make(), random.Random(0)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)
    for _ in range(3):  # three other conversations take all the slots
        run(sched, prompt(rng, 100))
    assert not any(s.tokens[:len(p1)] == p1 for s in sched.slots)
    p2 = p1 + r1.output + prompt(rng, 30)
    r2 = run(sched, p2)
    assert r2.reused == len(p1) + len(r1.output) - 1
    assert sched.metrics.disk_tokens.get() == r2.reused and sched.store.stats["restores"] == 1


def test_a_restart_restores_from_disk(make):
    sched, rng = make(), random.Random(1)
    p1 = chat(rng, prompt(rng, 300), prompt(rng, 100))  # a first message of 256+ tokens gets a checkpoint at its end
    r1 = run(sched, p1, cache_salt="a")
    sched.store.flush()
    sched2 = make()  # a new engine: nothing in memory, the same directory
    p2 = p1 + r1.output + [BND] + prompt(rng, 40) + [BND] + prompt(rng, 2)
    assert run(sched2, p2, cache_salt="a").reused == len(p1) + len(r1.output) - 1
    assert run(sched2, p2, cache_salt="b").reused == 0  # another salt shares nothing
    p3 = chat(rng, p1[1:301], prompt(rng, 50))  # the same system prompt in a new conversation: its boundary checkpoint
    assert run(make(), p3, cache_salt="a").reused == 301


def test_shared_prefixes_share_their_blocks(make):
    sched, rng = make(n_slots=1), random.Random(2)
    system = prompt(rng, 4 * BLOCK)
    for _ in range(3):
        run(sched, chat(rng, system, prompt(rng, 2 * BLOCK)))
    sched.store.flush()
    st = sched.store.status()
    # each conversation: the 4 blocks of the shared system prompt (stored once), and 2 more of its own
    assert st["blocks"] == 4 + 3 * 2 and st["checkpoints"] == 4  # the system prompt's checkpoint, and each reply end


def test_the_budget_evicts_the_least_recently_used_files(make, tmp_path):
    rng = random.Random(3)
    sched = make(budget=60_000)
    convs = [prompt(rng, 300) for _ in range(6)]
    for p in convs:
        run(sched, p)
    sched.store.flush()
    st = sched.store.status()
    on_disk = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(tmp_path) for f in fs if f.endswith((".kv", ".ck")))
    assert st["bytes"] <= 60_000 and on_disk == st["bytes"]
    for p in convs:  # evicted or not, every next turn is exact
        run(make(budget=60_000), p + prompt(rng, 20))


def test_a_bad_file_falls_back_to_prefill(make, tmp_path):
    sched, rng = make(n_slots=1), random.Random(4)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)
    sched.store.flush()
    blocks = [os.path.join(d, f) for d, _, fs in os.walk(tmp_path) for f in fs if f.endswith(".kv")]
    with open(blocks[0], "r+b") as f:
        f.truncate(100)
    run(sched, prompt(rng, 50))  # takes the only slot
    r2 = run(sched, p1 + r1.output + prompt(rng, 10))
    assert r2.reused == 0 and sched.store.stats["errors"] == 1


def test_a_pending_save_holds_back_a_rewrite_of_its_rows(make, monkeypatch):
    write = SessionStore._write

    def slow_write(self, *a):
        time.sleep(0.2)
        return write(self, *a)
    monkeypatch.setattr(SessionStore, "_write", slow_write)
    sched, rng = make(n_slots=1), random.Random(5)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)
    run(sched, prompt(rng, 200))  # overwrites the slot while the writer is still busy with p1
    r2 = run(sched, p1 + r1.output + prompt(rng, 10))
    assert r2.reused == len(p1) + len(r1.output) - 1


def test_the_next_turn_does_not_wait_for_the_write_of_the_last_reply(make, monkeypatch):
    """The client sends the reply back changed: turn 2 restores the prompt end of turn 1 and rewrites the slot from
    there, over rows of the reply-end checkpoint that the writer has not read yet. Those rows are in its partial block,
    copied when the save was queued, so turn 2 starts at once, and the checkpoint on disk still restores exactly."""
    gate, write = threading.Event(), SessionStore._write

    def gated_write(self, *a):
        gate.wait(5)
        return write(self, *a)
    monkeypatch.setattr(SessionStore, "_write", gated_write)
    sched, rng = make(n_slots=1), random.Random(10)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)
    t0 = time.perf_counter()
    r2 = run(sched, p1 + r1.output[:-5] + prompt(rng, 10))
    waited = time.perf_counter() - t0
    gate.set()
    assert r2.reused == len(p1) and waited < 2.5
    p3 = p1 + r1.output + prompt(rng, 10)
    assert run(make(), p3).reused == len(p1) + len(r1.output) - 1  # a new engine: from disk


def test_a_ring_buffer_is_not_reused_while_its_save_is_pending(make, monkeypatch):
    """The writer reads a checkpoint's GDN state from the scheduler's ring. With a ring of two, the snapshots of the
    next requests (in other slots: no wait for the save) would take that buffer; they must take the other one (or
    none), so the file holds the saved state."""
    gate, write = threading.Event(), SessionStore._write

    def gated_write(self, *a):
        gate.wait(5)
        return write(self, *a)
    monkeypatch.setattr(SessionStore, "_write", gated_write)
    sched, rng = make(n_checkpoints=2), random.Random(12)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)  # its reply-end save waits for the gate, pinning its ring buffer
    for _ in range(2):
        run(sched, prompt(rng, 300))  # two snapshots each: they may reuse only the other buffer (the second finds none)
    with sched.store.lock:
        pend = {j.buf: j.n for j in sched.store.pending}
    assert pend and all(len(c.tokens) == pend[c.buf] for c in sched.ckpts if c.buf in pend)  # a pinned buffer holds its saved state
    gate.set()
    sched.store.flush()
    r2 = run(make(n_checkpoints=2), p1 + r1.output + prompt(rng, 10))  # a new engine: from disk, exact
    assert r2.reused == len(p1) + len(r1.output) - 1


def test_a_slot_that_still_holds_the_rows_reads_only_the_gdn_state(make, monkeypatch):
    sched, rng = make(n_checkpoints=1), random.Random(6)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)
    sched.store.flush()  # saved: its ring buffer may be reused
    run(sched, prompt(rng, 100))  # another slot; the ring of one checkpoint now holds only this one
    assert not any(c.slot == 0 for c in sched.ckpts) and sched.slots[0].tokens[:len(p1)] == p1
    reads = []
    read = SessionStore._read
    monkeypatch.setattr(SessionStore, "_read", lambda self, path, buf: reads.append(path) or read(self, path, buf))
    r2 = run(sched, p1 + r1.output + prompt(rng, 10))
    assert r2.reused == len(p1) + len(r1.output) - 1 and len(reads) == 1 and reads[0].endswith(".ck")


def test_requests_without_prefix_reuse_are_not_saved(make):
    sched, rng = make(), random.Random(7)
    run(sched, prompt(rng, 300), disk=False)
    sched.store.flush()
    assert sched.store.status()["files"] == 0


def test_a_disk_restore_for_a_side_request_copies_the_prefix_into_another_slot(make):
    """The system-prompt checkpoint of the main conversation left the ring but is on disk, and the main conversation's
    slot still holds its KV: a side request reads only the GDN state, in another slot with a copy of the KV prefix."""
    sched, rng = make(n_checkpoints=4), random.Random(9)
    system = prompt(rng, 300)
    main = chat(rng, system, prompt(rng, 1200))
    r1 = run(sched, main)
    sched.store.flush()  # saved: the ring buffers may be reused
    run(sched, prompt(rng, 200))  # another conversation: its snapshots push the main one's system-prompt checkpoint out of the ring
    assert not any(len(c.tokens) == 301 for c in sched.ckpts)
    r2 = run(sched, chat(rng, system, prompt(rng, 40)), 10)
    assert r2.slot != r1.slot and r2.reused == 301 and sched.store.stats["restores"] == 1
    p3 = main + r1.output + prompt(rng, 30)
    r3 = run(sched, p3)
    assert r3.slot == r1.slot and r3.reused == len(main) + len(r1.output) - 1  # the main conversation, still in memory


# ------------------------------------------------------------------------------------------------ a state with more tensors
def test_the_fingerprint_of_the_27b_state_keeps_its_shapes():
    """The shapes of the fingerprint come from the state's hooks, and equal the dict that the store wrote before the
    hooks (the name of the directory of the 27B's files depends on it). A state with extra tensors adds two keys."""
    st, mst = FastState(CFG, 2, 64, "meta"), FastState(CFG, 2, 64, "meta")
    old = {"kv": [list(st.k[i].shape[1:]) for i in sorted(st.k)], "kv_dtype": str(st.k[min(st.k)].dtype),
           "conv": [list(st.conv[i].shape[1:]) for i in sorted(st.rec)], "rec": [list(st.rec[i].shape[1:]) for i in sorted(st.rec)],
           "drafter_kv": list(mst.k[1].shape[1:])}
    mst.k = {0: mst.k[1]}
    assert _shapes(st, mst) == old
    ext = ExtModel().new_state(2, 64)
    assert set(_shapes(ext, None)) - set(old) == {"kv_extra", "snap_extra"}


@pytest.fixture
def make_ext(tmp_path):
    """make() of this file on the ExtModel of tests/test_scheduler.py, with its prefill as the scheduler's prefill_fn."""
    stores = []

    def make(budget=10**9, **kw):
        for st in stores:
            st.flush()
        kw = dict(n_slots=3, max_seq_len=L, n_checkpoints=8, prefill_chunk=64, ckpt_interval=10**6, boundary_token=BND) | kw
        sched = Checked(ExtModel(), prefill_fn=ext_prefill, **kw)
        with torch.inference_mode():
            sched.store = SessionStore(sched.state, None, {"model": "ext"}, budget, str(tmp_path), block=BLOCK, min_tokens=BLOCK,
                                       log=lambda *a: None)
        sched.store.min_gain = 16
        stores.append(sched.store)
        return sched
    yield make
    for st in stores:
        st.flush()


def run_ext(sched, p, n=20, **kw):
    r = sched.run([S.Request(p, max_new_tokens=n, **kw)])[0]
    assert r.output == ext_reference(p, n)
    return r


def test_a_state_with_extra_tensors_restores_from_disk(make_ext):
    """The rows of the extra per-position tensors (one with a row per 4 positions) go into the blocks and the partial
    block, and the extra snapshot tensor into the checkpoint: after an eviction, after a restart, and into another slot
    whose KV is still in memory, each next turn is exact."""
    sched, rng = make_ext(), random.Random(30)
    p1 = prompt(rng, 299)  # a partial block that ends inside a group of 4 positions
    r1 = run_ext(sched, p1)
    for _ in range(3):  # three other conversations take all the slots
        run_ext(sched, prompt(rng, 100))
    p2 = p1 + r1.output + prompt(rng, 30)
    r2 = run_ext(sched, p2)
    assert r2.reused == len(p1) + len(r1.output) - 1 and sched.store.stats["restores"] == 1
    sched.store.flush()
    p3 = p2 + r2.output + prompt(rng, 7)
    assert run_ext(make_ext(), p3).reused == len(p2) + len(r2.output) - 1  # a restart: a new engine, the same directory


# ------------------------------------------------------------------------------------------------ qwen4_exp on a small model
@pytest.fixture(scope="module")
def small4(tmp_path_factory):
    """The small random qwen4_exp model of tests/test_fast_qwen4exp.py on the kernels, with the family's hooks: its n-gram
    table written to a file and read through a PleFeeder (model.feeder), the family's prefill (chunks with prev2), the
    before_replay that fills the feed buffer."""
    import copy
    import shutil

    from engine.model import family as F
    from engine.model.fast_qwen4exp import to_fast4
    from engine.weights.ple import NgramTable, PleFeeder
    from tests.qwen4exp_ckpt import ngram_shards, write_safetensors
    from tests.test_fast_qwen4exp import build_reference
    d = tmp_path_factory.mktemp("small4")
    shutil.copy(os.path.join(os.path.dirname(__file__), "data", "qwen4exp_config.json"), d / "config.json")
    fam = F.family_of(str(d))
    fast = to_fast4(copy.deepcopy(build_reference()))
    ple = fast.ple_layer()
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    path = str(d / "model-fp8-mtp-ple.safetensors")
    table = torch.from_numpy(ple.table.data).view(torch.float8_e4m3fn)
    write_safetensors(path, [(prefix + ".weight_scale", torch.tensor([ple.table.scale], dtype=torch.bfloat16))] +
                      ngram_shards(table, 4, [2, 0, 3, 1], prefix))
    nt = NgramTable(path, prefix)
    fast.feeder = PleFeeder(nt, 4096, params=ple.params, image_token_id=fast.cfg.image_token_id)
    yield fast, fam
    fast.feeder.close()
    nt.close()


def make4(small4, store_dir=None):
    """A scheduler of 2 slots of 512 tokens on the small qwen4_exp model with the family's callables, as build_engine
    makes it, and a store of 64-token blocks in store_dir (None: no store)."""
    fast, fam = small4
    sched = Checked(fast, n_slots=2, max_seq_len=512, n_checkpoints=4, prefill_chunk=64, ckpt_interval=10**6, prefill_fn=fam.prefill,
                    prepare_fn=fam.prepare_prefill, before_replay=fam.before_replay(fast), feed_buffer=fam.feed_buffer(fast),
                    prefill_prev2=fam.prefill_prev2)
    if store_dir is not None:
        with torch.inference_mode():
            sched.store = SessionStore(sched.state, None, {"model": "small4", "drafter": "none"}, 10**9, str(store_dir), block=BLOCK,
                                       min_tokens=BLOCK, log=lambda *a: None)
        sched.store.min_gain = 16
    return sched


def test_a_qwen4exp_state_restores_from_disk_byte_for_byte(small4, tmp_path):
    """On the small qwen4_exp model: a conversation of 150 tokens (three prefill chunks, the n-gram rows through the
    feeder) and its reply go to disk; after a restart (a zeroed state, a new store on the same directory), the next turn
    restores the reply end from disk. The slot then holds the bytes it held: the rows of every kv_tensors() tensor (k, v,
    raw and block indexer keys at one row per 4 positions, rotary positions) and every snap_tensors() tensor (GDN, PLE
    conv). The next turn's greedy output equals the one after a restore from memory."""
    rng = random.Random(40)
    sched = make4(small4, tmp_path)
    p1 = [rng.randrange(4, 1000) for _ in range(150)]
    r1 = sched.run([S.Request(p1, max_new_tokens=12)])[0]
    sched.store.flush()
    b, n = r1.slot, len(p1) + len(r1.output) - 1  # the reply end: the last output token was never fed
    st = sched.state
    assert n % 4 and n > 2 * BLOCK  # a partial block, and a partial group of 4 positions in it
    kv = [t[b, :, :-(-n // r)].clone() for t, r in zip(st.kv_tensors(), st.kv_ratios())]
    snap = [t[b].clone() for t in st.snap_tensors()]
    assert all(x.view(torch.uint8).any() for x in kv + snap)
    sched.store = None
    del sched
    sched = make4(small4, tmp_path)  # a restart: nothing in memory, the same directory
    with torch.inference_mode():
        sched.state.reset()
    p2 = p1 + r1.output + [rng.randrange(4, 1000) for _ in range(20)]
    req = sched.submit(S.Request(p2, max_new_tokens=12))
    with torch.inference_mode():
        sched._admit()
    s = sched.slots[req.slot]
    assert req.reused == n and sched.store.stats["restores"] == 1 and sched.metrics.disk_tokens.get() == n
    st = sched.state
    for t, r, w in zip(st.kv_tensors(), st.kv_ratios(), kv):
        assert torch.equal(t[s.idx, :, :-(-n // r)].view(torch.uint8), w.view(torch.uint8)), t.shape
    for t, w in zip(st.snap_tensors(), snap):
        assert torch.equal(t[s.idx], w), t.shape
    while sched.busy():
        sched.step()
    sched.store.flush()
    sched.store = None
    del sched
    ref = make4(small4)  # the same two turns, turn 2 restored from memory
    assert ref.run([S.Request(p1, max_new_tokens=12)])[0].output == r1.output
    rm = ref.run([S.Request(p2, max_new_tokens=12)])[0]
    assert rm.reused == n and rm.output == req.output

"""The disk cache of conversations (engine/runtime/sessions.py) under the scheduler, on the stub model of
tests/test_scheduler.py: its next token depends on the whole history, through the GDN state and the KV cache that the
store writes and reads back. Thus an output equals the reference continuation only if a restore gave the slot exactly
the state of its prefix."""
import os
import random
import time

import pytest
import torch

import engine.runtime.scheduler as S
from engine.runtime.sessions import SessionStore
from tests.test_scheduler import BND, L, StubModel, chat, prompt, reference, stub_prefill

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
        sched = S.Scheduler(StubModel(), **kw)
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


def test_a_slot_that_still_holds_the_rows_reads_only_the_gdn_state(make, monkeypatch):
    sched, rng = make(n_checkpoints=1), random.Random(6)
    p1 = prompt(rng, 300)
    r1 = run(sched, p1)
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
    run(sched, prompt(rng, 200))  # another conversation: its snapshots push the main one's system-prompt checkpoint out of the ring
    sched.store.flush()
    assert not any(len(c.tokens) == 301 for c in sched.ckpts)
    r2 = run(sched, chat(rng, system, prompt(rng, 40)), 10)
    assert r2.slot != r1.slot and r2.reused == 301 and sched.store.stats["restores"] == 1
    p3 = main + r1.output + prompt(rng, 30)
    r3 = run(sched, p3)
    assert r3.slot == r1.slot and r3.reused == len(main) + len(r1.output) - 1  # the main conversation, still in memory

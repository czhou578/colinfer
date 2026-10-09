"""The bookkeeping of engine/runtime/scheduler.py on a stub model (GPU, no checkpoint): queueing (one request at a
time), chunked prefill, prefix checkpoints (snapshot, restore, eviction, salts), a conversation's cache across a side
request, finish reasons, aborts and reset.

The stub's next token depends on the whole history of its slot, through both kinds of state that the scheduler moves
around: a hash of the history in the GDN recurrent state, and the tokens themselves in the attention KV cache (their
sum). Thus an output equals the reference continuation only if the scheduler gave the request exactly the state of its
own prompt, whatever it restored.
"""
import random

import pytest
import torch

import engine.runtime.scheduler as S
from engine.model.fast import FastState
from engine.model.qwen35 import Qwen35Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

V, P, L = 64, 1009, 2048  # vocabulary, hash modulus, slot length
BND = V - 1               # the message-boundary token
CFG = Qwen35Config(hidden_size=8, intermediate_size=8, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
                   num_attention_heads=1, num_key_value_heads=1, head_dim=8, linear_num_key_heads=1, linear_num_value_heads=1,
                   linear_key_head_dim=2, linear_value_head_dim=2, linear_conv_kernel_dim=2, vocab_size=V)


def reference(prompt, n, eos=(), min_tokens=0):
    """The stub's greedy continuation of a prompt, computed on the host."""
    h, s, out = 0, 0, []
    for t in prompt:
        h, s = (h * 31 + t) % P, s + t
    while len(out) < n:
        out.append((h + s) % V)
        if out[-1] in eos and len(out) >= min_tokens:
            break
        h, s = (h * 31 + out[-1]) % P, s + out[-1]
    return out


class StubModel:
    """Next token = (h + the sum of the tokens in the KV cache) mod V, h = the history's hash in the recurrent state."""
    cfg = CFG
    layers = []
    lm_head = None
    prefill_ready = False

    def new_state(self, batch, max_seq_len):
        return FastState(CFG, batch, max_seq_len, "cuda")

    def __call__(self, tok, state):
        """One decode step for the active slots (DecodeGraph captures it): feed tok at pos_t, advance pos_t."""
        act, y = state.active > 0, tok[:, 0]
        h = state.rec[0][:, 0, 0, 0]
        h_new = torch.where(act, torch.remainder(h * 31 + y.float(), P), h)
        state.rec[0][:, 0, 0, 0] = h_new
        kb = state.k[1].view(torch.uint8)[:, 0, :, 0]  # [B, L]: one token per byte
        rows, pos = torch.arange(kb.shape[0], device="cuda"), state.pos_t.long().clamp_max(kb.shape[1] - 1)
        kb[rows, pos] = torch.where(act, y.to(torch.uint8), kb[rows, pos])
        state.pos_t += state.active
        s = (kb.int() * (torch.arange(kb.shape[1], device="cuda") < state.pos_t[:, None])).sum(-1)
        nxt = torch.remainder(h_new.long() + s, V)
        return torch.zeros(kb.shape[0], V, device="cuda").scatter_(1, nxt[:, None], 10.0)


def stub_prefill(model, ids, state, return_hidden=False):
    """engine/model/prefill.py's contract on a single-slot view: feed ids from state.pos, return the next logits."""
    toks, p0 = ids[0].tolist(), state.pos
    h = float(state.rec[0][0, 0, 0, 0])
    for t in toks:
        h = (h * 31 + t) % P
    state.rec[0][0, 0, 0, 0] = h
    kb = state.k[1].view(torch.uint8)[0, 0, :, 0]
    kb[p0:p0 + len(toks)] = torch.tensor(toks, dtype=torch.uint8, device="cuda")
    state.pos += len(toks)
    state.pos_t += len(toks)
    logits = torch.zeros(1, V, device="cuda")
    logits[0, (int(h) + int(kb[:state.pos].int().sum())) % V] = 10.0
    return logits


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(S, "prefill", stub_prefill)

    def make(**kw):
        kw = dict(n_slots=3, max_seq_len=L, n_checkpoints=8, prefill_chunk=64, ckpt_interval=128, boundary_token=BND) | kw
        return S.Scheduler(StubModel(), **kw)
    return make


def prompt(rng, n):
    return [rng.randrange(V - 1) for _ in range(n)]  # no boundary tokens


def chat(rng, system, user):
    """[BND] system [BND] user [BND] x: a chat whose first message ends past 256 tokens is split there (a checkpoint)."""
    return [BND] + system + [BND] + user + [BND] + prompt(rng, 2)


def test_queued_requests_match_the_reference(make):
    sched, rng = make(), random.Random(0)
    reqs = [S.Request(prompt(rng, n), max_new_tokens=m) for n, m in ((5, 20), (70, 7), (200, 33), (130, 12), (9, 40), (300, 3), (640, 25))]
    sched.run(reqs)  # 7 requests on 3 slots, one at a time: queueing, chunked prefill, slot reuse
    for r in reqs:
        assert r.output == reference(r.prompt, r.max_new_tokens) and r.finish_reason == "length"


def test_stop_tokens_and_min_tokens(make):
    sched, rng = make(), random.Random(1)
    p = prompt(rng, 50)
    ref = reference(p, 60)
    eos = (ref[10],)
    first = ref.index(eos[0])
    r = sched.run([S.Request(p, max_new_tokens=60, eos_ids=eos)])[0]
    assert r.output == ref[:first + 1] and r.finish_reason == "stop"
    r = sched.run([S.Request(p, max_new_tokens=60, eos_ids=eos, min_tokens=first + 2)])[0]
    assert r.output == reference(p, 60, eos, first + 2)


def test_next_turn_restores_the_end_of_turn_checkpoint(make):
    sched, rng = make(), random.Random(2)
    p1 = prompt(rng, 150)
    r1 = sched.run([S.Request(p1, max_new_tokens=20)])[0]
    p2 = p1 + r1.output + prompt(rng, 30)
    r2 = sched.run([S.Request(p2, max_new_tokens=20)])[0]
    assert r2.reused == len(p1) + len(r1.output) - 1  # the last output token was never fed
    assert r2.output == reference(p2, 20)
    r3 = sched.run([S.Request(p2, max_new_tokens=20, cache_salt="other")])[0]  # checkpoints are shared within a salt only
    assert r3.reused == 0 and r3.output == r2.output


def test_a_side_request_waits_and_keeps_the_main_conversation_cached(make):
    sched, rng = make(), random.Random(3)
    main = sched.submit(S.Request(prompt(rng, 150), max_new_tokens=20))
    while not main.output:
        sched.step()
    side = sched.submit(S.Request(prompt(rng, 60), max_new_tokens=10))  # e.g. a title request while main decodes
    sched.step()
    assert side.slot == -1  # it waits for main
    while sched.busy():
        sched.step()
    assert side.slot != main.slot and side.output == reference(side.prompt, 10)
    p2 = main.prompt + main.output + prompt(rng, 30)  # main's next turn: its slot still holds its history
    r2 = sched.run([S.Request(p2, max_new_tokens=20)])[0]
    assert r2.reused == len(main.prompt) + len(main.output) - 1 and r2.output == reference(p2, 20)


def test_reply_sent_back_in_another_form_restores_the_reply_start(make):
    sched, rng = make(ckpt_interval=10**6), random.Random(7)
    system, user = prompt(rng, 300), prompt(rng, S.REPLY_SPLIT_MIN)
    p1 = chat(rng, system, user)
    reply_start = len(p1) - 3  # the [BND] that opens the generation prompt
    r1 = sched.run([S.Request(p1, max_new_tokens=20)])[0]
    assert any(len(c.tokens) == reply_start for c in sched.ckpts)
    # the client sends the reply back changed (e.g. without its reasoning): turn 2 differs from p1 after the [BND]
    p2 = p1[:reply_start + 1] + prompt(rng, 12) + [BND] + prompt(rng, 40) + [BND] + prompt(rng, 2)
    r2 = sched.run([S.Request(p2, max_new_tokens=20)])[0]
    assert r2.reused == reply_start and r2.output == reference(p2, 20)
    assert r1.output == reference(p1, 20)


def test_reply_sent_back_unchanged_skips_the_reply_start_snapshot(make):
    sched, rng = make(ckpt_interval=10**6), random.Random(8)
    p1 = chat(rng, prompt(rng, 300), prompt(rng, 40))  # a short last message: no reply-start snapshot
    r1 = sched.run([S.Request(p1, max_new_tokens=20)])[0]
    p2 = p1 + r1.output + [BND] + prompt(rng, S.REPLY_SPLIT_MIN) + [BND] + prompt(rng, 2)
    r2 = sched.run([S.Request(p2, max_new_tokens=20)])[0]
    assert r2.reused == len(p1) + len(r1.output) - 1  # the end of turn 1's generation
    assert not any(len(c.tokens) == len(p2) - 3 for c in sched.ckpts)  # this client needs no reply-start snapshot
    assert r2.output == reference(p2, 20)


def test_outputs_stay_exact_as_checkpoints_are_evicted(make):
    sched, rng = make(n_checkpoints=4), random.Random(4)
    ps = [prompt(rng, 600) for _ in range(5)]  # snapshots every 128 tokens: the ring of 4 evicts all the time
    for p in ps:
        assert sched.run([S.Request(p, max_new_tokens=10)])[0].output == reference(p, 10)
    for p in (ps[-1] + prompt(rng, 50), ps[0] + prompt(rng, 50)):
        r = sched.run([S.Request(p, max_new_tokens=10)])[0]
        assert r.output == reference(p, 10)
    assert len(sched.ckpts) <= 4


def test_aborts_queued_and_running(make):
    sched, rng = make(), random.Random(5)
    reqs = [S.Request(prompt(rng, 40), max_new_tokens=300) for _ in range(4)]
    for r in reqs:
        sched.submit(r)
    assert sched.abort(reqs[3].rid)  # still queued
    for _ in range(6):
        sched.step()
    assert reqs[0].output and sched.abort(reqs[0].rid)  # decoding
    while sched.busy():
        sched.step()
    assert [r.finish_reason for r in reqs] == ["abort", "length", "length", "abort"] and reqs[3].output == []
    assert reqs[0].output == reference(reqs[0].prompt, 300)[:len(reqs[0].output)]
    assert reqs[1].output == reference(reqs[1].prompt, 300) and reqs[2].output == reference(reqs[2].prompt, 300)
    assert sched.metrics.requests.get(reason="abort") == 2 and not sched.abort(reqs[0].rid)


def test_reset_forgets_everything(make):
    sched, rng = make(), random.Random(6)
    p = prompt(rng, 300)
    sched.run([S.Request(p, max_new_tokens=5)])
    assert sched.ckpts and sched.metrics.requests.get(reason="length") == 1
    sched.reset()
    assert not sched.ckpts and all(not s.tokens for s in sched.slots) and sched.metrics.requests.get(reason="length") == 0
    r = sched.run([S.Request(p + [1, 2], max_new_tokens=5)])[0]
    assert r.reused == 0 and r.output == reference(p + [1, 2], 5)
    sched.submit(S.Request(p, max_new_tokens=5))
    with pytest.raises(RuntimeError):
        sched.reset()


def test_submit_checks_token_ids(make):
    sched = make()
    for bad in (S.Request([]), S.Request([5, V]), S.Request([-1]), S.Request([5], eos_ids=(V,)), S.Request([1] * L)):
        with pytest.raises(ValueError):
            sched.submit(bad)
    assert not sched.queue


def test_a_time_limit_ends_a_running_request_and_drops_a_queued_one(make):
    sched, rng = make(), random.Random(9)
    a = S.Request(prompt(rng, 40), max_new_tokens=2000, max_seconds=0.05)
    b = S.Request(prompt(rng, 40), max_new_tokens=5, max_seconds=0.01)  # waits behind a, so its time runs out in the queue
    sched.run([a, b])
    assert a.finish_reason == "timeout" and 0 < len(a.output) < 2000
    assert a.output == reference(a.prompt, 2000)[:len(a.output)]
    assert b.finish_reason == "timeout" and b.output == [] and b.slot == -1
    assert sched.metrics.requests.get(reason="timeout") == 2

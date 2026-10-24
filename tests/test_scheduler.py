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


def stub_prefill(model, ids, state, return_hidden=False, logits=True):
    """engine/model/prefill.py's contract on a single-slot view: feed ids from state.pos, return the next logits (None
    with logits=False: a chunk before the end of the prompt)."""
    toks, p0 = ids[0].tolist(), state.pos
    h = float(state.rec[0][0, 0, 0, 0])
    for t in toks:
        h = (h * 31 + t) % P
    state.rec[0][0, 0, 0, 0] = h
    kb = state.k[1].view(torch.uint8)[0, 0, :, 0]
    kb[p0:p0 + len(toks)] = torch.tensor(toks, dtype=torch.uint8, device="cuda")
    state.pos += len(toks)
    state.pos_t += len(toks)
    if not logits:
        return None
    logits = torch.zeros(1, V, device="cuda")
    logits[0, (int(h) + int(kb[:state.pos].int().sum())) % V] = 10.0
    return logits


class Checked(S.Scheduler):
    """The scheduler, whose every step is followed by a check of its invariant: each checkpoint is a prefix of its slot's
    history (the scheduler relies on it rather than compare them). A subclass, not a wrapper of step: a reference cycle
    would leave old schedulers to the garbage collector, which may free their CUDA graphs during a new one's capture."""

    def step(self):
        super().step()
        for c in self.ckpts:
            assert tuple(self.slots[c.slot].tokens[:len(c.tokens)]) == c.tokens, "a checkpoint is not a prefix of its slot's history"


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(S, "prefill", stub_prefill)

    def make(**kw):
        kw = dict(n_slots=3, max_seq_len=L, n_checkpoints=8, prefill_chunk=64, ckpt_interval=128, boundary_token=BND) | kw
        return Checked(StubModel(), **kw)
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


def test_requests_count_their_decode_steps(make):
    """Plain decode: one cycle per output token after the first (the prefill makes the first), no drafts; a request that
    ends in the queue ran no cycle."""
    sched, rng = make(), random.Random(5)
    r = sched.run([S.Request(prompt(rng, 30), max_new_tokens=12)])[0]
    assert (r.cycles, r.drafted, r.accepted) == (11, 0, 0)
    one = sched.run([S.Request(prompt(rng, 30), max_new_tokens=1)])[0]
    assert (len(one.output), one.cycles) == (1, 0)
    queued = S.Request(prompt(rng, 30), max_new_tokens=5, max_seconds=1e-9)
    sched.submit(S.Request(prompt(rng, 300), max_new_tokens=3))
    sched.submit(queued)
    while sched.busy():
        sched.step()
    assert (queued.finish_reason, queued.cycles, queued.t_admit, queued.t_first) == ("timeout", 0, 0.0, 0.0)


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
    assert r2.slot == r1.slot  # the overwritten rest of turn 1 (its reply) is short: no copy into another slot


def test_a_side_request_with_the_same_system_prompt_keeps_the_conversation_cached(make):
    sched, rng = make(ckpt_interval=10**6), random.Random(10)
    system = prompt(rng, 300)
    main = chat(rng, system, prompt(rng, 1200))
    r1 = sched.run([S.Request(main, max_new_tokens=20)])[0]
    side = chat(rng, system, prompt(rng, 40))  # e.g. a title request with the main conversation's system prompt
    r2 = sched.run([S.Request(side, max_new_tokens=10)])[0]
    assert r2.slot != r1.slot and r2.reused == 301 and r2.output == reference(side, 10)  # its prefix, copied into another slot
    p3 = main + r1.output + prompt(rng, 30)
    r3 = sched.run([S.Request(p3, max_new_tokens=20)])[0]
    assert r3.slot == r1.slot and r3.reused == len(main) + len(r1.output) - 1 and r3.output == reference(p3, 20)


def test_a_copy_into_another_slot_drops_the_checkpoints_there(make):
    """The side request's slot held another conversation, with checkpoints shorter than the copied prefix: they go,
    since they are no prefix of the slot's new history (a disk save of one would pair its tokens with the new KV)."""
    sched, rng = make(n_slots=2, ckpt_interval=10**6), random.Random(11)
    system = prompt(rng, 300)
    main = chat(rng, system, prompt(rng, 1200))
    r1 = sched.run([S.Request(main, max_new_tokens=20)])[0]
    other = prompt(rng, 200)
    r0 = sched.run([S.Request(other, max_new_tokens=10)])[0]
    assert r0.slot != r1.slot and any(c.slot == r0.slot and len(c.tokens) < 301 for c in sched.ckpts)
    side = chat(rng, system, prompt(rng, 40))
    r2 = sched.run([S.Request(side, max_new_tokens=10)])[0]
    assert r2.slot == r0.slot and r2.reused == 301 and r2.output == reference(side, 10)
    assert not any(c.slot == r0.slot and len(c.tokens) < 301 for c in sched.ckpts)


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


@pytest.mark.parametrize("slots,ring,tail", [(3, 6, 64), (2, 3, 64), (3, 8, S.COPY_MIN_TAIL), (1, 4, S.COPY_MIN_TAIL)])
def test_random_conversations_stay_exact(make, monkeypatch, slots, ring, tail):
    """Turns that send the reply back unchanged or changed, side requests with a shared system prompt, branches from
    an earlier point, new conversations: whatever the scheduler restores, copies or evicts, each output is exact. A low
    COPY_MIN_TAIL sends many requests to another slot with a copy of their prefix."""
    monkeypatch.setattr(S, "COPY_MIN_TAIL", tail)
    sched, rng = make(n_slots=slots, n_checkpoints=ring, ckpt_interval=64), random.Random(slots * 10 + ring)
    systems, done = [prompt(rng, 300), prompt(rng, 40)], []
    for _ in range(40):
        kind = rng.choice(["new", "turn", "turn", "changed", "side", "branch"]) if done else "new"
        p, out = rng.choice(done) if done else ([], [])
        user = prompt(rng, rng.choice((5, 60, 200)))
        p = {"new": lambda: chat(rng, rng.choice(systems), user), "turn": lambda: p + out + [BND] + user + [BND] + prompt(rng, 2),
             "changed": lambda: p[:-3] + prompt(rng, 5) + [BND] + user, "side": lambda: chat(rng, systems[0], user),
             "branch": lambda: p[:rng.randrange(1, len(p))] + user}[kind]()
        p = p if len(p) < L - 64 else p[-(L // 2):]  # a conversation at the slot length starts over
        r = sched.run([S.Request(p, max_new_tokens=rng.choice((3, 20)))])[0]
        assert r.output == reference(p, r.max_new_tokens), kind
        done.append((p, r.output))


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


# ------------------------------------------------------------------------------------------------ a state with more tensors
P2, R = 997, 4  # the modulus of the second hash; positions per row of the block tensor


class ExtState(FastState):
    """FastState with the extra tensors of a state like qwen4_exp's FastState4, through the same hooks: a per-position
    tensor `ex` (a copy of the tokens), a tensor `exb` with one row per R positions (the token at the first position of
    each group) and a snapshot tensor `h2` (a second hash of the history). The scheduler and the disk cache know them
    only through kv_tensors() and snap_tensors()."""

    def __init__(self, cfg, batch, max_seq_len, device):
        super().__init__(cfg, batch, max_seq_len, device)
        self.ex = torch.zeros(batch, 1, max_seq_len, 1, dtype=torch.uint8, device=device)
        self.exb = torch.zeros(batch, 1, max_seq_len // R, 1, dtype=torch.uint8, device=device)
        self.h2 = torch.zeros(batch, 1, dtype=torch.float32, device=device)

    def reset(self, kv: bool = True):
        super().reset(kv)
        self.h2.zero_()
        if kv:
            self.ex.zero_()
            self.exb.zero_()

    def view(self, lo, hi):
        v = super().view(lo, hi)
        v.ex, v.exb, v.h2 = self.ex[lo:hi], self.exb[lo:hi], self.h2[lo:hi]
        return v

    def kv_tensors(self):
        return super().kv_tensors() + [self.ex, self.exb]

    def snap_tensors(self):
        return super().snap_tensors() + [self.h2]


def ext_reference(prompt, n):
    """The ExtModel's greedy continuation: (h + 2 h2 + 4 s + 5 sb) mod V. s: the sum of the tokens (in k and in ex);
    sb: the sum of the tokens at the positions 0, R, 2R, ... (in exb)."""
    h, h2, s, sb, out = 0, 0, 0, 0, []

    def feed(t, p):
        nonlocal h, h2, s, sb
        h, h2, s, sb = (h * 31 + t) % P, (h2 * 7 + t) % P2, s + t, sb + (t if p % R == 0 else 0)
    for p, t in enumerate(prompt):
        feed(t, p)
    while len(out) < n:
        out.append((h + 2 * h2 + 4 * s + 5 * sb) % V)
        feed(out[-1], len(prompt) + len(out) - 1)
    return out


class ExtModel(StubModel):
    """StubModel on an ExtState: the next token also reads the extra tensors, so an output is exact only if the
    scheduler moved them with the rest of the state."""

    def new_state(self, batch, max_seq_len):
        return ExtState(CFG, batch, max_seq_len, "cuda")

    def __call__(self, tok, state):
        act, y = state.active > 0, tok[:, 0]
        h, h2 = state.rec[0][:, 0, 0, 0], state.h2[:, 0]
        h_new = torch.where(act, torch.remainder(h * 31 + y.float(), P), h)
        h2_new = torch.where(act, torch.remainder(h2 * 7 + y.float(), P2), h2)
        state.rec[0][:, 0, 0, 0], state.h2[:, 0] = h_new, h2_new
        kb, ex, exb = state.k[1].view(torch.uint8)[:, 0, :, 0], state.ex[:, 0, :, 0], state.exb[:, 0, :, 0]
        rows, pos = torch.arange(kb.shape[0], device="cuda"), state.pos_t.long().clamp_max(kb.shape[1] - 1)
        y8 = y.to(torch.uint8)
        kb[rows, pos] = torch.where(act, y8, kb[rows, pos])
        ex[rows, pos] = torch.where(act, y8, ex[rows, pos])
        exb[rows, pos // R] = torch.where(act & (pos % R == 0), y8, exb[rows, pos // R])
        state.pos_t += state.active
        n = state.pos_t[:, None]
        s = (kb.int() * (torch.arange(kb.shape[1], device="cuda") < n)).sum(-1)
        sx = (ex.int() * (torch.arange(ex.shape[1], device="cuda") < n)).sum(-1)
        sb = (exb.int() * (torch.arange(exb.shape[1], device="cuda") * R < n)).sum(-1)
        nxt = torch.remainder(h_new.long() + 2 * h2_new.long() + s + 3 * sx + 5 * sb, V)
        return torch.zeros(kb.shape[0], V, device="cuda").scatter_(1, nxt[:, None], 10.0)


def ext_prefill(model, ids, state, return_hidden=False, logits=True):
    """stub_prefill for the ExtModel. The scheduler gets it as its prefill_fn (the family's prefill)."""
    toks, p0 = ids[0].tolist(), state.pos
    h, h2 = float(state.rec[0][0, 0, 0, 0]), float(state.h2[0, 0])
    for t in toks:
        h, h2 = (h * 31 + t) % P, (h2 * 7 + t) % P2
    state.rec[0][0, 0, 0, 0], state.h2[0, 0] = h, h2
    t8 = torch.tensor(toks, dtype=torch.uint8, device="cuda")
    kb, ex, exb = state.k[1].view(torch.uint8)[0, 0, :, 0], state.ex[0, 0, :, 0], state.exb[0, 0, :, 0]
    kb[p0:p0 + len(toks)] = t8
    ex[p0:p0 + len(toks)] = t8
    for i, t in enumerate(toks):
        if (p0 + i) % R == 0:
            exb[(p0 + i) // R] = t
    state.pos += len(toks)
    state.pos_t += len(toks)
    if not logits:
        return None
    n = state.pos
    out = torch.zeros(1, V, device="cuda")
    out[0, (int(h) + 2 * int(h2) + 4 * int(kb[:n].int().sum()) + 5 * int(exb[:-(-n // R)].int().sum())) % V] = 10.0
    return out


@pytest.fixture
def make_ext():
    """Schedulers of the ExtModel, with its prefill as prefill_fn (S.prefill stays the dense one) and a before_replay
    hook that records its calls."""
    def make(**kw):
        calls = []
        kw = dict(n_slots=3, max_seq_len=L, n_checkpoints=8, prefill_chunk=64, ckpt_interval=128, boundary_token=BND) | kw
        sched = Checked(ExtModel(), prefill_fn=ext_prefill, before_replay=lambda s, toks: calls.append((s.idx, list(toks), len(s.tokens))), **kw)
        sched.replays = calls
        return sched
    return make


def test_the_state_hooks_of_the_27b_keep_the_old_order():
    """kv_tensors() and snap_tensors() of FastState list the tensors in the order that the ring and the session files
    used before the hooks: the k / v pairs by layer, then conv by layer, then rec by layer."""
    cfg = Qwen35Config(hidden_size=8, intermediate_size=8, num_hidden_layers=4, layer_types=["linear_attention", "full_attention"] * 2,
                       num_attention_heads=1, num_key_value_heads=1, head_dim=8, linear_num_key_heads=1, linear_num_value_heads=1,
                       linear_key_head_dim=2, linear_value_head_dim=2, linear_conv_kernel_dim=2, vocab_size=V)
    st = FastState(cfg, 2, 64, "meta")
    assert [id(t) for t in st.kv_tensors()] == [id(t) for t in (st.k[1], st.v[1], st.k[3], st.v[3])]
    assert [id(t) for t in st.snap_tensors()] == [id(t) for t in (st.conv[0], st.conv[2], st.rec[0], st.rec[2])]


def test_a_state_with_extra_tensors_keeps_them_across_restores_and_copies(make_ext, monkeypatch):
    """The checks of the dense stub on the ExtModel: a turn restores its checkpoint, a side request runs in another slot
    with a copy of the prefix (the block rows too), and the rows of each plain decode step reach before_replay."""
    monkeypatch.setattr(S, "COPY_MIN_TAIL", 64)
    sched, rng = make_ext(ckpt_interval=10**6), random.Random(20)
    system = prompt(rng, 301)  # the copied prefix (302 tokens) ends inside a group of R positions
    main = chat(rng, system, prompt(rng, 300))
    r1 = sched.run([S.Request(main, max_new_tokens=20)])[0]
    assert r1.output == ext_reference(main, 20)
    side = chat(rng, system, prompt(rng, 40))
    r2 = sched.run([S.Request(side, max_new_tokens=10)])[0]
    assert r2.slot != r1.slot and r2.reused == 302 and r2.output == ext_reference(side, 10)
    p3 = main + r1.output + prompt(rng, 30)
    r3 = sched.run([S.Request(p3, max_new_tokens=20)])[0]
    assert r3.slot == r1.slot and r3.reused == len(main) + len(r1.output) - 1 and r3.output == ext_reference(p3, 20)
    # each plain step of r3: the one row [y], y the last output token; the history does not hold y yet
    steps = [c for c in sched.replays[-19:]]
    assert [c[1] for c in steps] == [[t] for t in r3.output[:-1]] and all(c[0] == r3.slot for c in steps)
    assert [c[2] for c in steps] == list(range(len(p3), len(p3) + 19))


@pytest.mark.parametrize("slots,ring,tail", [(3, 6, 64), (2, 3, S.COPY_MIN_TAIL)])
def test_random_conversations_on_a_state_with_extra_tensors_stay_exact(make_ext, monkeypatch, slots, ring, tail):
    """test_random_conversations_stay_exact on the ExtModel."""
    monkeypatch.setattr(S, "COPY_MIN_TAIL", tail)
    sched, rng = make_ext(n_slots=slots, n_checkpoints=ring, ckpt_interval=64), random.Random(slots * 10 + ring + 1)
    systems, done = [prompt(rng, 299), prompt(rng, 41)], []
    for _ in range(30):
        kind = rng.choice(["new", "turn", "changed", "side", "branch"]) if done else "new"
        p, out = rng.choice(done) if done else ([], [])
        user = prompt(rng, rng.choice((5, 61, 203)))
        p = {"new": lambda: chat(rng, rng.choice(systems), user), "turn": lambda: p + out + [BND] + user + [BND] + prompt(rng, 2),
             "changed": lambda: p[:-3] + prompt(rng, 5) + [BND] + user, "side": lambda: chat(rng, systems[0], user),
             "branch": lambda: p[:rng.randrange(1, len(p))] + user}[kind]()
        p = p if len(p) < L - 64 else p[-(L // 2):]
        r = sched.run([S.Request(p, max_new_tokens=rng.choice((3, 20)))])[0]
        assert r.output == ext_reference(p, r.max_new_tokens), kind
        done.append((p, r.output))


def test_each_prefill_chunk_gets_the_two_tokens_before_it(make_ext):
    """With prefill_prev2 (qwen4_exp), the scheduler passes prev2 = the last two tokens of the slot's history before each
    chunk: none at a slot's start, then the end of the previous chunk, or of the restored checkpoint."""
    calls = []

    def prefill(model, ids, state, prev2=None, **kw):
        calls.append((state.pos, list(prev2)))
        return ext_prefill(model, ids, state, **kw)
    sched, rng = make_ext(prefill_chunk=64, ckpt_interval=10**6), random.Random(21)
    sched.prefill_fn, sched.prefill_prev2 = prefill, True
    p1 = prompt(rng, 140)  # chunks of 64 and 76 tokens (the scheduler finishes a prompt of 5 / 4 chunks or less in one)
    r1 = sched.run([S.Request(p1, max_new_tokens=5)])[0]
    assert calls == [(0, []), (64, p1[62:64])] and r1.output == ext_reference(p1, 5)
    calls.clear()
    p2 = p1 + r1.output + prompt(rng, 10)
    r2 = sched.run([S.Request(p2, max_new_tokens=5)])[0]
    n = len(p1) + len(r1.output) - 1
    assert r2.reused == n and calls == [(n, p2[n - 2:n])] and r2.output == ext_reference(p2, 5)


# ------------------------------------------------------------------------------------------------ the qwen4_exp state
class State4Model:
    """A stand-in for FastQwen4Exp on the real FastState4 of Qwen3.8-Flash-Next (its config's shapes, a short slot): a
    decode step only moves the positions. The test fills the state itself."""
    prefill_ready = False

    def __init__(self, L):
        from engine.model.qwen4exp import Qwen4ExpConfig
        self.cfg, self.L = Qwen4ExpConfig(), L

    def new_state(self, batch, max_seq_len):
        from engine.model.fast_qwen4exp import FastState4
        return FastState4(self.cfg, batch, max_seq_len, "cuda")

    def __call__(self, tok, state, feed=None):
        state.pos_t += state.active
        return torch.zeros(tok.shape[0], 8, device="cuda")


@torch.inference_mode()
def test_the_ring_snapshots_and_restores_the_qwen4exp_state():
    """The checkpoint ring on FastState4 at the real shapes: one buffer per snap_tensors() tensor (36 GDN conv windows
    and recurrent states, then the PLE conv state; 115.6 MB per snapshot). A restore into the other slot gives it the
    snapshot and the rows of every kv_tensors() tensor (k, v, raw and block indexer keys, rotary positions)."""
    Lm, n = 64, 22  # 22 positions: 6 rows of block keys, the last one partial
    sched = S.Scheduler(State4Model(Lm), n_slots=2, max_seq_len=Lm, n_checkpoints=3, prepare_fn=lambda m: None)
    st = sched.state
    snap = st.snap_tensors()
    assert len(sched.ring) == len(snap) == 36 + 36 + 1
    assert [(r.shape[1:], r.dtype) for r in sched.ring] == [(t.shape[1:], t.dtype) for t in snap]
    assert sum(r[0].nbytes for r in sched.ring) == 36 * (10240 * 3 * 2 + 48 * 128 * 128 * 4) + 10240 * 9 * 2
    g = torch.Generator(device="cuda").manual_seed(4)

    def fill(t):
        if t.dtype == torch.float8_e4m3fn:
            t.view(torch.uint8).copy_(torch.randint(0, 120, t.shape, device="cuda", dtype=torch.uint8, generator=g))
        elif t.dtype == torch.int32:
            t.copy_(torch.randint(0, 1000, t.shape, device="cuda", dtype=torch.int32, generator=g))
        else:
            t.copy_(torch.randn(t.shape, device="cuda", generator=g))
    for t in st.kv_tensors() + snap:
        fill(t)
    s0, s1 = sched.slots
    s0.tokens = list(range(n))
    want_snap = [t[0].clone() for t in snap]
    want_kv = [t[0, :, :-(-n // r)].clone() for t, r in zip(st.kv_tensors(), st.kv_ratios())]
    sched._snapshot(s0)
    c = sched.ckpts[0]
    for t in snap:  # the live state moves on
        fill(t)
    sched._restore(s1, c, 0)
    assert all(torch.equal(t[1], w) for t, w in zip(snap, want_snap))
    assert all(torch.equal(t[1, :, :w.shape[1]].view(torch.uint8), w.view(torch.uint8)) for t, w in zip(st.kv_tensors(), want_kv))
    assert [w.shape[1] for w in want_kv] == [n] * 36 + [6] * 12 + [n]  # k / v pairs and raw keys, block keys (12 layers), rope_pos
    assert s1.tokens == s0.tokens and int(st.pos_t[1]) == n

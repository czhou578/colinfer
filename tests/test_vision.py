"""Image input (engine/model/vision.py, engine/server/images.py):
- the multimodal RoPE positions against transformers' get_rope_index
- the attention prologue at (t, h, w) positions against the reference rotary, and the text path unchanged bit for bit
- the request side: image parts, data URLs, the Anthropic blocks, the image processor, the prompt expansion
- the scheduler on the stub model of tests/test_scheduler.py with a stub vision tower whose rows change the outputs:
  each image is embedded once, at the right rows across chunk boundaries, and the caches (memory and disk) tell
  images apart
"""
import base64
import io
import random
import types

import pytest
import torch

import engine.runtime.scheduler as S
from engine.model.vision import Image, rope_delta, rope_positions
from tests.test_scheduler import BND, CFG, L, StubModel, V, prompt, reference, stub_prefill

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def image(grid, key=1, start=-1):
    return Image(torch.empty(0), grid, key, 2, start)


# ------------------------------------------------------------------------------------------------ positions
@pytest.mark.parametrize("layout", [[7, (1, 6, 10), 4, (1, 12, 4), 9], [(1, 4, 4), 3], [5, (1, 2, 30), 1]])
def test_rope_positions_match_transformers(layout):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    fake = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    fake.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(fake, *a, **k)
    types_, images, grids = [], [], []
    for x in layout:
        if isinstance(x, int):
            types_ += [0] * x
        else:
            images.append(image(x, start=len(types_)))
            grids.append(x)
            types_ += [1] * images[-1].n
    T = len(types_)
    pos, deltas = Qwen3_5Model.get_rope_index(fake, torch.zeros(1, T, dtype=torch.long), torch.tensor([types_]),
                                             image_grid_thw=torch.tensor(grids))
    want = pos[:, 0].T.numpy()
    assert (rope_positions(images, 0, T) == want).all()
    for lo, hi in ((0, 1), (3, T - 2), (T - 1, T)):
        assert (rope_positions(images, lo, hi) == want[lo:hi]).all()
    assert rope_delta(images) == int(deltas)


# ------------------------------------------------------------------------------------------------ the prologue
@GPU
def test_prologue_rotary_positions_match_the_reference():
    """q / k after the prologue at given (t, h, w) positions, against the reference RMSNorm and multimodal rotary."""
    from engine.kernels import ops
    from engine.model.qwen35 import Qwen35Config, RMSNorm, apply_rotary, rope_inv_freq, rotary
    torch.manual_seed(0)
    cfg = Qwen35Config(layer_types=["full_attention"], num_hidden_layers=1)
    Hq, Hkv, D, T = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, 7
    qp = torch.randn(1, T, Hq * 2 * D, device="cuda").bfloat16()
    kp = torch.randn(1, T, Hkv * D, device="cuda").bfloat16()
    vp = torch.randn(1, T, Hkv * D, device="cuda").bfloat16()
    qn, kn = RMSNorm(D, cfg.rms_norm_eps).cuda().bfloat16(), RMSNorm(D, cfg.rms_norm_eps).cuda().bfloat16()
    with torch.no_grad():
        qn.weight.normal_(0, 0.3)
        kn.weight.normal_(0, 0.3)
    pos3 = torch.tensor([[40, 40, 40], [41, 41, 42], [41, 45, 41], [41, 300, 1200], [7, 9, 11], [1300, 1300, 1300], [0, 5, 2]],
                        dtype=torch.int32, device="cuda")
    inv = rope_inv_freq(cfg, "cuda")
    kc = torch.zeros(1, Hkv, 64, D, device="cuda", dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    q = torch.empty(1, Hq, T, D, device="cuda", dtype=torch.bfloat16)
    pos_t = torch.full((1,), 10, dtype=torch.int32, device="cuda")
    ops().attn_prologue(qp, kp, vp, qn.weight, kn.weight, inv, pos_t, kc, vc, q, cfg.rms_norm_eps, None, None, pos3, *cfg.mrope_section[1:])
    with torch.no_grad():
        rq = qn(qp.view(1, T, Hq, 2 * D)[..., :D]).transpose(1, 2)
        rk = kn(kp.view(1, T, Hkv, D)).transpose(1, 2)
        cos, sin = rotary(cfg, pos3.long(), torch.bfloat16)
        rq, rk = apply_rotary(rq, rk, cos, sin)
    for got, want in ((q, rq), (kc[:, :, 10:10 + T].bfloat16(), rk.to(torch.float8_e4m3fn).bfloat16())):
        rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
        assert rel < 2e-2, rel  # the reference rounds cos / sin to bf16; the kernel keeps them in fp32


@GPU
def test_prologue_text_rows_are_unchanged():
    """No delta, a zero delta and (t, h, w) positions that are all equal give the bits of the plain text path."""
    from engine.kernels import ops
    torch.manual_seed(1)
    Hq, Hkv, D, T = 24, 4, 256, 5
    qp = torch.randn(2, T, Hq * 2 * D, device="cuda").bfloat16()
    kp, vp = torch.randn(2, T, Hkv * D, device="cuda").bfloat16(), torch.randn(2, T, Hkv * D, device="cuda").bfloat16()
    w = torch.randn(D, device="cuda").bfloat16() * 0.3
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
    pos_t = torch.tensor([30, 900], dtype=torch.int32, device="cuda")

    def run(**kw):
        kc = torch.zeros(2, Hkv, 1024, D, device="cuda", dtype=torch.float8_e4m3fn)
        q = torch.empty(2, Hq, T, D, device="cuda", dtype=torch.bfloat16)
        ops().attn_prologue(qp, kp, vp, w, w, inv, pos_t, kc, torch.zeros_like(kc), q, 1e-6, **kw)
        return q, kc
    base = run()
    delta = torch.tensor([-25, 400], dtype=torch.int32, device="cuda")
    for other in (run(rope_delta=torch.zeros(2, dtype=torch.int32, device="cuda")), ):
        assert all(torch.equal(a.view(torch.uint8) if a.dtype != torch.bfloat16 else a, b.view(torch.uint8) if b.dtype != torch.bfloat16 else b)
                   for a, b in zip(base, other))
    shifted = run(rope_delta=delta)
    p = (pos_t[:, None] + delta[:, None] + torch.arange(T, device="cuda")).reshape(-1, 1).expand(-1, 3).contiguous().int()
    explicit = run(rope_pos=p)
    assert torch.equal(shifted[0], explicit[0]) and torch.equal(shifted[1].view(torch.uint8), explicit[1].view(torch.uint8))
    assert not torch.equal(shifted[0], base[0])


# ------------------------------------------------------------------------------------------------ requests
def png(color, size=(96, 64)):
    from PIL import Image as PI
    buf = io.BytesIO()
    PI.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_image_parts_and_data_urls():
    from engine.server import images
    url = "data:image/png;base64," + base64.b64encode(png("red")).decode()
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": url}}]},
            {"role": "tool", "content": [{"type": "image_url", "image_url": url}]}]
    out, blobs = images.extract(msgs, True)
    assert blobs == [png("red")] * 2 and out[1]["content"][1] == {"type": "image"} and out[2]["content"] == [{"type": "image"}]
    assert out[0] is msgs[0] and msgs[1]["content"][1]["type"] == "image_url"  # the request's own messages are not changed
    out, blobs = images.extract(msgs, False)
    assert not blobs and out[1]["content"][1] == {"type": "text", "text": images.NOTE}
    for bad in ("https://example.com/a.png", "data:image/png,abc", "data:image/png;base64,***"):
        with pytest.raises(ValueError):
            images.extract([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": bad}}]}], True)


def test_anthropic_image_blocks():
    from engine.server import anthropic as anth
    src = {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png("blue")).decode()}
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": src}]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"p": "x.png"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                                                       "content": [{"type": "image", "source": src}]},
                                                      {"type": "text", "text": "and?"}]}]}
    m = anth.to_messages(body, images=True)
    url = "data:image/png;base64," + src["data"]
    assert m[0]["content"] == [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}]
    assert m[2] == {"role": "tool", "tool_call_id": "t1", "content": [{"type": "text", "text": anth.TOOL_ERROR + "\n"},
                                                                       {"type": "image_url", "image_url": {"url": url}}]}
    assert m[3] == {"role": "user", "content": "and?"}
    m = anth.to_messages(body)  # without the vision tower: the text notes of before
    assert m[0]["content"] == "look\n\n[image omitted: this server reads text only]"
    with pytest.raises(ValueError):
        anth.to_messages({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "http://x"}}]}]}, True)


def test_image_processor_and_expansion():
    from engine.server import images
    from engine.weights.loader import MODEL, resolve
    try:
        proc = images.ImageProcessor(resolve(MODEL))
    except Exception:  # noqa: BLE001
        pytest.skip("needs the checkpoint's preprocessor_config.json")
    a, b = proc.load(png("red", (640, 480))), proc.load(png("green", (640, 480)))
    assert a.grid == (1, 30, 40) and a.n == 300 and a.pixels is None and a.patches().shape == (1200, 3 * 2 * 16 * 16)
    assert a.key != b.key and proc.load(png("red", (640, 480))).key == a.key
    assert proc.load(png("red", (8, 8))).n == 64  # small images grow to the minimum of 65,536 pixels
    with pytest.raises(ValueError):
        proc.load(b"not an image")
    # the grid comes from the image's size alone: it must be the processor's, and the patches made later its patches
    import random

    from PIL import Image as PI
    rng = random.Random(0)
    sizes = [(8, 8), (1023, 769), (4000, 300), (300, 4000), (2048, 2048), (3000, 2000)]
    sizes += [(rng.randrange(1, 1200), rng.randrange(1, 1200)) for _ in range(20)]
    for w, h in sizes:
        if max(w, h) / min(w, h) > 200:
            continue
        data = png((rng.randrange(256), 0, 0), (w, h))
        im = proc.load(data)
        want = proc.proc(images=[PI.open(io.BytesIO(data)).convert("RGB")], return_tensors="pt")
        assert im.grid == tuple(int(x) for x in want["image_grid_thw"][0]), (w, h)
        if w * h <= 1200 * 1200:
            assert torch.equal(im.patches(), want["pixel_values"].float()), (w, h)
    p = images.expand([1, 2, 99, 3, 99], [a, b], 99, 1000)
    assert a.start == 2 and b.start == 303 and p == [1, 2] + a.ids(1000) + [3] + b.ids(1000)
    with pytest.raises(ValueError):
        images.expand([1, 99], [a, b], 99, 1000)


# ------------------------------------------------------------------------------------------------ the scheduler
PAD = 5  # the stub's image pad token


class StubVision:
    """Row j of an image is the value (key + j) mod (V - 1) in every column: the stub prefill reads it as a token."""
    patch_dim, merge = 1, 2

    def __init__(self):
        self.calls = 0

    def encode(self, im):
        self.calls += 1
        v = (im.key + torch.arange(im.n, device="cuda")) % (V - 1)
        return v[:, None].expand(-1, CFG.hidden_size).to(torch.bfloat16)


def stub_prefill_img(model, ids, state, return_hidden=False, embeds=None, rope_pos=None, logits=True):
    toks = ids[0].tolist()
    assert all(t < V for t in toks) and (rope_pos is None or rope_pos.shape == (len(toks), 3))
    for off, e in embeds or ():
        for r in range(max(off, 0), min(off + e.shape[0], len(toks))):
            toks[r] = int(e[r - off, 0])
    return stub_prefill(model, torch.tensor([toks]), state, return_hidden, logits)


def effective(p, images):
    """The prompt as the stub model sees it: image tokens as their stub rows."""
    p = list(p)
    for im in images:
        p[im.start:im.start + im.n] = [(im.key + j) % (V - 1) for j in range(im.n)]
    return p


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(S, "prefill", stub_prefill_img)

    def make(**kw):
        sched = S.Scheduler(StubModel(), **(dict(n_slots=2, max_seq_len=L, n_checkpoints=8, prefill_chunk=64, boundary_token=BND) | kw))
        sched.vision, sched.image_pad = StubVision(), PAD
        return sched
    return make


def with_image(rng, key, head=50, grid=(1, 12, 12), tail=30):
    im = image(grid, key, head)
    return prompt(rng, head) + im.ids(V) + prompt(rng, tail), [im]


@GPU
def test_images_are_embedded_once_and_told_apart(make):
    sched, rng = make(), random.Random(0)
    pa, ia = with_image(rng, key=11)  # the image's 36 rows cross the chunk boundary at 64
    ra = sched.run([S.Request(pa, max_new_tokens=20, images=ia)])[0]
    assert ra.output == reference(effective(pa, ia), 20) and sched.vision.calls == 1
    pb, ib = list(pa), [image((1, 12, 12), 22, 50)]  # the same text, another image
    pb[50:50 + 36] = ib[0].ids(V)
    rb = sched.run([S.Request(pb, max_new_tokens=20, images=ib)])[0]
    assert rb.reused <= 50 and rb.output == reference(effective(pb, ib), 20) and rb.output != ra.output
    p2 = pa + ra.output + prompt(rng, 20)  # turn 2 of the first conversation: its image is in the restored prefix
    calls = sched.vision.calls
    r2 = sched.run([S.Request(p2, max_new_tokens=20, images=[image((1, 12, 12), 11, 50)])])[0]
    assert r2.reused == len(pa) + len(ra.output) - 1 and sched.vision.calls == calls
    assert r2.output == reference(effective(p2, [image((1, 12, 12), 11, 50)]), 20)
    assert all(im.embeds is None for im in ia + ib)  # released when their requests ended


@GPU
def test_the_vision_tower_loads_on_the_first_image():
    """Built without its weights: the patch shape and merge size come from the config; encode() loads it (not run here:
    the real tower is exercised by tests/scheduler_check.py)."""
    from engine.model.vision import VisionTower, vision_available
    from engine.weights.loader import MODEL, resolve
    try:
        path = resolve(MODEL)
    except Exception:  # noqa: BLE001
        pytest.skip("needs the checkpoint in the HF cache")
    if not vision_available(path):
        pytest.skip("the checkpoint has no vision tower")
    tower = VisionTower(path, load=False)
    assert not tower.loaded and tower.patch_dim == 3 * 2 * 16 * 16 and tower.merge == 2


def test_submit_checks_image_tokens(make):
    sched, rng = make(), random.Random(1)
    p, ims = with_image(rng, key=3)
    for bad in (p[:60] + p[61:], [t if t < V else t + 1 for t in p]):
        with pytest.raises(ValueError):
            sched.submit(S.Request(bad, images=ims))
    with pytest.raises(ValueError):
        sched.submit(S.Request(p))  # ids past the vocabulary without their image
    sched.vision = None
    with pytest.raises(ValueError):
        sched.submit(S.Request(p, images=ims))


@GPU
def test_an_image_conversation_restores_from_disk(make, tmp_path):
    from engine.runtime.sessions import SessionStore
    rng = random.Random(2)
    pa, ia = with_image(rng, key=1 << 40, head=200, tail=100)  # token ids past 2^31
    sched = make()
    with torch.inference_mode():
        sched.store = SessionStore(sched.state, None, {"model": "stub"}, 10**9, str(tmp_path), block=64, min_tokens=64, log=lambda *a: None)
    sched.store.min_gain = 16
    ra = sched.run([S.Request(pa, max_new_tokens=20, images=ia)])[0]
    sched.store.flush()
    sched2 = make()
    with torch.inference_mode():
        sched2.store = SessionStore(sched2.state, None, {"model": "stub"}, 10**9, str(tmp_path), block=64, min_tokens=64, log=lambda *a: None)
    sched2.store.min_gain = 16
    p2, i2 = pa + ra.output + prompt(rng, 20), [image((1, 12, 12), 1 << 40, 200)]
    r2 = sched2.run([S.Request(p2, max_new_tokens=20, images=i2)])[0]
    assert r2.reused == len(pa) + len(ra.output) - 1 and sched2.vision.calls == 0
    assert r2.output == reference(effective(p2, i2), 20)
    sched2.store.flush()

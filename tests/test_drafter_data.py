"""tools/drafter_data.py build_prompts: the mix of a seed is reproducible, and a change in one corpus (a transformers
upgrade changes the code files) moves only that corpus's choices, not the mix or the other corpora's. The corpora are
stubbed; the real ones read local datasets."""
import random

import pytest

from tools import drafter_data as dd


@pytest.fixture
def corpora(monkeypatch):
    def install(code):
        monkeypatch.setattr(dd, "wiki_material", lambda rng: ([f"title {i}" for i in range(600)], [f"para {i}" for i in range(50)]))
        monkeypatch.setattr(dd, "story_starts", lambda rng: [f"story {i}" for i in range(50)])
        monkeypatch.setattr(dd, "code_snippets", lambda rng: list(code))
    return install


def test_same_seed_same_prompts(corpora):
    corpora([f"def f{i}(): pass" for i in range(30)])
    a = dd.build_prompts(60, random.Random(1))
    b = dd.build_prompts(60, random.Random(1))
    assert a == b and len({k for _, k, _ in a}) == 4


def test_a_corpus_change_moves_only_its_own_prompts(corpora):
    corpora([f"def f{i}(): pass" for i in range(30)])
    a = dd.build_prompts(60, random.Random(1))
    corpora([f"def g{i}(): return {i}" for i in range(30)])  # other contents (the real corpus always has 3,000 snippets)
    b = dd.build_prompts(60, random.Random(1))
    assert [(k, t) for _, k, t in a] == [(k, t) for _, k, t in b]  # the mix: kinds and the thinking switch
    assert [p for p, k, _ in a if k != "code"] == [p for p, k, _ in b if k != "code"]  # the other corpora's prompts
    assert any(pa != pb for (pa, ka, _), (pb, _, _) in zip(a, b) if ka == "code" and "def " in pa)  # the code prompts did change

"""Fixtures shared by the test modules."""
import pytest


@pytest.fixture(scope="module")
def fmt():
    """ChatFormat on the real Qwen3.8 tokenizer (CPU); the test is skipped without the checkpoint in the HF cache."""
    from transformers import AutoTokenizer

    from engine.server.chat import ChatFormat
    from engine.weights.loader import resolve
    try:
        path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    except Exception:
        pytest.skip("checkpoint not in the local HF cache")
    return ChatFormat(AutoTokenizer.from_pretrained(path))

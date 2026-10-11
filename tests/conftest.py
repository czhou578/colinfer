"""Fixtures shared by the test modules."""
import pytest


@pytest.fixture(scope="module")
def fmt():
    """ChatFormat on the real Qwen3.8 tokenizer (CPU); the test is skipped without the checkpoint in the HF cache."""
    from engine.server.chat import ChatFormat
    from engine.weights.loader import MODEL, resolve
    try:
        path = resolve(MODEL)
    except Exception:
        pytest.skip("checkpoint not in the local HF cache")
    return ChatFormat.from_checkpoint(path)

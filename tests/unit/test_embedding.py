"""Tests for drivers.embedding: EmbeddingDriver ABC and concrete drivers.

No real model download or network call happens here — the local driver's
SentenceTransformer is monkeypatched, and the OpenAI driver's
max_sequence_length() never touches the network in the first place.
"""

from unittest.mock import MagicMock

import pytest

import drivers.embedding as embedding_module
from drivers.embedding import (
    EmbeddingDriver,
    LocalSentenceTransformerDriver,
    OpenAIEmbeddingDriver,
    get_embedding_driver,
)


class _FakeDriver(EmbeddingDriver):
    """Minimal concrete subclass to exercise the ABC's default behavior.

    Deliberately does NOT override embed_text — it should come for free
    from the ABC's default implementation.
    """

    @property
    def dimension(self) -> int:
        return 1

    def embed_batch(self, texts):
        return [[float(len(t))] for t in texts]


def test_default_max_sequence_length_is_none():
    assert _FakeDriver().max_sequence_length() is None


def test_default_embed_text_delegates_to_embed_batch():
    """embed_text has no per-subclass override — it must come from the ABC."""
    assert _FakeDriver().embed_text("abc") == [3.0]


def test_openai_driver_max_sequence_length_is_none():
    assert OpenAIEmbeddingDriver().max_sequence_length() is None


def test_local_driver_max_sequence_length_reads_from_model(monkeypatch):
    fake_model = MagicMock()
    fake_model.max_seq_length = 128
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda name: fake_model
    )

    driver = LocalSentenceTransformerDriver()
    assert driver.max_sequence_length() == 128


def test_local_driver_loads_model_only_once(monkeypatch):
    load_calls = []
    fake_model = MagicMock()
    fake_model.max_seq_length = 128

    def fake_constructor(name):
        load_calls.append(name)
        return fake_model

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_constructor)

    driver = LocalSentenceTransformerDriver()
    driver.max_sequence_length()
    driver.embed_batch(["hello"])

    assert len(load_calls) == 1


def test_get_embedding_driver_returns_local_by_default(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="local")
    )
    assert isinstance(get_embedding_driver(), LocalSentenceTransformerDriver)


def test_get_embedding_driver_returns_openai(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="openai")
    )
    assert isinstance(get_embedding_driver(), OpenAIEmbeddingDriver)


def test_get_embedding_driver_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="bogus")
    )
    with pytest.raises(ValueError, match="Unknown EMBEDDING_DRIVER"):
        get_embedding_driver()

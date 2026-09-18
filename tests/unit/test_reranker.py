"""Tests for drivers.reranker: RerankerDriver ABC and concrete drivers.

No real model download happens here — CrossEncoderRerankerDriver's
sentence_transformers.CrossEncoder is monkeypatched.
"""

from unittest.mock import MagicMock

import pytest

import drivers.reranker as reranker_module
from drivers.reranker import (
    CrossEncoderRerankerDriver,
    NoopRerankerDriver,
    get_reranker_driver,
)


def _chunk(content, page=1):
    return {"id": 1, "content": content, "metadata": {"page_number": page}, "score": 0.5}


def test_noop_driver_returns_chunks_unchanged():
    chunks = [_chunk("a"), _chunk("b")]
    assert NoopRerankerDriver().rerank("question", chunks) is chunks


def test_cross_encoder_driver_sorts_by_predicted_score(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9]
    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder", lambda name: fake_model
    )

    chunks = [_chunk("low relevance"), _chunk("high relevance")]
    reranked = CrossEncoderRerankerDriver().rerank("question", chunks)

    assert [c["content"] for c in reranked] == ["high relevance", "low relevance"]
    assert reranked[0]["score"] == 0.9
    assert reranked[1]["score"] == 0.1


def test_cross_encoder_driver_scores_question_chunk_pairs(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.5]
    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder", lambda name: fake_model
    )

    CrossEncoderRerankerDriver().rerank("my question", [_chunk("chunk text")])

    fake_model.predict.assert_called_once_with([("my question", "chunk text")])


def test_cross_encoder_driver_returns_empty_list_without_loading_model(monkeypatch):
    load_calls = []
    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        lambda name: load_calls.append(name),
    )

    assert CrossEncoderRerankerDriver().rerank("question", []) == []
    assert load_calls == []


def test_cross_encoder_driver_loads_model_only_once(monkeypatch):
    load_calls = []
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1]

    def fake_constructor(name):
        load_calls.append(name)
        return fake_model

    monkeypatch.setattr("sentence_transformers.CrossEncoder", fake_constructor)

    driver = CrossEncoderRerankerDriver()
    driver.rerank("q1", [_chunk("a")])
    driver.rerank("q2", [_chunk("b")])

    assert len(load_calls) == 1


def test_get_reranker_driver_returns_noop_by_default(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="none")
    )
    assert isinstance(get_reranker_driver(), NoopRerankerDriver)


def test_get_reranker_driver_returns_cross_encoder(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="cross_encoder")
    )
    assert isinstance(get_reranker_driver(), CrossEncoderRerankerDriver)


def test_get_reranker_driver_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="bogus")
    )
    with pytest.raises(ValueError, match="Unknown RERANKER_DRIVER"):
        get_reranker_driver()

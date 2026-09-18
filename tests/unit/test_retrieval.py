"""Tests for query.retrieval's pure logic: RetrievalStrategy implementations,
the strategy factory, and the relevance gate.

No real DB, embedding model, or LLM is needed here — VectorStore and the
reranker driver are mocked/faked, and select_chunks()/_passes_relevance_gate()
are exercised directly with plain chunk dicts. Full end-to-end coverage
(embedding + real Postgres) lives in tests/db/test_retrieval_db.py.
"""

from unittest.mock import MagicMock

import pytest

import query.retrieval as retrieval_module
from query.retrieval import (
    HybridRetrievalStrategy,
    VectorRetrievalStrategy,
    _passes_relevance_gate,
    get_retrieval_strategy,
    retrieve_chunks,
)


def _chunk(chunk_id, score=0.0, source="a.pdf"):
    return {"id": chunk_id, "content": f"chunk {chunk_id}", "metadata": {"source_file": source}, "score": score}


class _NoopFakeReranker:
    """Stands in for drivers.reranker.NoopRerankerDriver — passthrough."""

    def rerank(self, question, chunks):
        return chunks


def test_passes_relevance_gate_true_when_top_result_clears_threshold():
    assert _passes_relevance_gate([_chunk(1, score=0.9)], top_k=4, min_score=0.25) is True


def test_passes_relevance_gate_false_when_nothing_clears_threshold():
    results = [_chunk(1, score=0.1), _chunk(2, score=0.2)]
    assert _passes_relevance_gate(results, top_k=4, min_score=0.25) is False


def test_passes_relevance_gate_false_for_empty_results():
    assert _passes_relevance_gate([], top_k=4, min_score=0.25) is False


def test_passes_relevance_gate_ignores_results_beyond_top_k():
    # The only qualifying result is at index 1, beyond top_k=1.
    results = [_chunk(1, score=0.1), _chunk(2, score=0.9)]
    assert _passes_relevance_gate(results, top_k=1, min_score=0.25) is False


def test_vector_strategy_filters_by_min_score_and_truncates_to_top_k():
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.1), _chunk(3, score=0.5)]
    strategy = VectorRetrievalStrategy()

    result = strategy.select_chunks(
        "q", vector_results, store=MagicMock(), top_k=1, min_score=0.25
    )

    assert [c["id"] for c in result] == [1]


def test_vector_strategy_returns_all_qualifying_when_under_top_k():
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.5)]
    strategy = VectorRetrievalStrategy()

    result = strategy.select_chunks(
        "q", vector_results, store=MagicMock(), top_k=5, min_score=0.25
    )

    assert [c["id"] for c in result] == [1, 2]


def test_hybrid_strategy_fuses_vector_and_fulltext_results(monkeypatch):
    vector_results = [_chunk(1, score=0.9)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = [_chunk(2, score=0.5)]
    monkeypatch.setattr(retrieval_module, "get_reranker_driver", lambda: _NoopFakeReranker())

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks("q", vector_results, fake_store, top_k=5, min_score=0.25)

    assert {c["id"] for c in result} == {1, 2}
    fake_store.search_fulltext.assert_called_once_with("q", top_k=1)


def test_hybrid_strategy_truncates_to_top_k_after_fusion(monkeypatch):
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.8)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    monkeypatch.setattr(retrieval_module, "get_reranker_driver", lambda: _NoopFakeReranker())

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks("q", vector_results, fake_store, top_k=1, min_score=0.25)

    assert len(result) == 1


def test_get_retrieval_strategy_returns_hybrid_by_default(monkeypatch, settings_override):
    monkeypatch.setattr(
        retrieval_module, "settings", settings_override(RETRIEVAL_STRATEGY="hybrid")
    )
    assert isinstance(get_retrieval_strategy(), HybridRetrievalStrategy)


def test_get_retrieval_strategy_returns_vector(monkeypatch, settings_override):
    monkeypatch.setattr(
        retrieval_module, "settings", settings_override(RETRIEVAL_STRATEGY="vector")
    )
    assert isinstance(get_retrieval_strategy(), VectorRetrievalStrategy)


def test_get_retrieval_strategy_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        retrieval_module, "settings", settings_override(RETRIEVAL_STRATEGY="bogus")
    )
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_STRATEGY"):
        get_retrieval_strategy()


def _patch_driver_and_store(monkeypatch, search_results=None):
    fake_driver = MagicMock()
    fake_driver.dimension = 384
    monkeypatch.setattr(retrieval_module, "get_embedding_driver", lambda: fake_driver)

    fake_store = MagicMock()
    fake_store.search.return_value = search_results or []
    # MagicMock treats "assert_*" names as typo-guards by default (raises
    # AttributeError), not real attributes — assign explicitly since
    # VectorStore genuinely has a method with this name.
    fake_store.assert_dimension_matches = MagicMock()
    monkeypatch.setattr(retrieval_module, "VectorStore", lambda: fake_store)

    return fake_driver, fake_store


def test_retrieve_chunks_skips_embedding_when_query_vector_given(monkeypatch):
    """The whole point of the query_vector override: avoid re-embedding (and
    re-triggering the driver's lazy model load) when a caller already has
    the vector — see scripts/evaluate_retrieval.py.
    """
    fake_driver, fake_store = _patch_driver_and_store(monkeypatch)

    retrieve_chunks("question", query_vector=[0.1, 0.2])

    fake_driver.embed_text.assert_not_called()
    fake_store.search.assert_called_once()
    assert fake_store.search.call_args.args[0] == [0.1, 0.2]


def test_retrieve_chunks_embeds_when_no_query_vector_given(monkeypatch):
    fake_driver, fake_store = _patch_driver_and_store(monkeypatch)
    fake_driver.embed_text.return_value = [0.9, 0.9]

    retrieve_chunks("question")

    fake_driver.embed_text.assert_called_once_with("question")
    assert fake_store.search.call_args.args[0] == [0.9, 0.9]

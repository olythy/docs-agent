"""Tests for query.retrieval's pure logic: RetrievalStrategy implementations,
the strategy factory, and the relevance gate.

No real DB, embedding model, or LLM is needed here — VectorStore and the
reranker driver are mocked/faked, and select_chunks()/_passes_relevance_gate()
are exercised directly with plain chunk dicts. Full end-to-end coverage
(embedding + real Postgres) lives in tests/db/test_retrieval_db.py.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

import query.retrieval as retrieval_module
from drivers.reranker import CrossEncoderRerankerDriver
from models import ChunkMetadata, RetrievedChunk
from query.retrieval import (
    HybridRetrievalStrategy,
    VectorRetrievalStrategy,
    _apply_top_k_with_guarantees,
    _csls_rerank,
    _merge_unique,
    _passes_relevance_gate,
    _round_robin_by_document,
    get_retrieval_strategy,
    retrieve_chunks,
)


def _chunk(chunk_id, score=0.0, source="a.pdf"):
    return RetrievedChunk(
        id=chunk_id,
        content=f"chunk {chunk_id}",
        metadata=ChunkMetadata(source_file=source, page_number=None, chunk_index=0),
        score=score,
    )


class _NoopFakeReranker:
    """Stands in for drivers.reranker.NoopRerankerDriver — passthrough."""

    def rerank(self, question, chunks):
        return chunks


class _FakeCrossEncoderReranker(CrossEncoderRerankerDriver):
    """Subclass of CrossEncoderRerankerDriver for testing reranker thresholds."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def rerank(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return [replace(c, score=s) for c, s in zip(chunks, self._scores, strict=True)]


def test_passes_relevance_gate_true_when_top_result_clears_threshold():
    assert (
        _passes_relevance_gate([_chunk(1, score=0.9)], top_k=4, min_score=0.25) is True
    )


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

    assert [c.id for c in result] == [1]


def test_vector_strategy_returns_all_qualifying_when_under_top_k():
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.5)]
    strategy = VectorRetrievalStrategy()

    result = strategy.select_chunks(
        "q", vector_results, store=MagicMock(), top_k=5, min_score=0.25
    )

    assert [c.id for c in result] == [1, 2]


def test_hybrid_strategy_fuses_vector_and_fulltext_results(monkeypatch):
    vector_results = [_chunk(1, score=0.9)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = [_chunk(2, score=0.5)]
    monkeypatch.setattr(
        retrieval_module, "get_reranker_driver", lambda *a, **k: _NoopFakeReranker()
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "q", vector_results, fake_store, top_k=5, min_score=0.25
    )

    assert {c.id for c in result} == {1, 2}
    fake_store.search_fulltext.assert_called_once_with("q", top_k=1)


def test_hybrid_strategy_rescues_identifier_match_missed_by_vector_and_fulltext(
    monkeypatch,
):
    """Regression test for a real, live miss: a case number present in the
    question can lose to common words in ts_rank's fusion scoring, and
    never appear in either vector_results or search_fulltext()'s output at
    all (see docs/decisions.md) -- search_by_identifier() must still surface
    it, merged in ahead of the RRF-fused list."""
    vector_results = [_chunk(1, score=0.9)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    fake_store.search_by_identifier.return_value = [_chunk(99, score=1.0)]
    monkeypatch.setattr(
        retrieval_module, "get_reranker_driver", lambda *a, **k: _NoopFakeReranker()
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "Mi történt a 4.P.20.409/2023/4. ügyben?",
        vector_results,
        fake_store,
        top_k=5,
        min_score=0.25,
    )

    assert [c.id for c in result] == [99, 1]
    fake_store.search_by_identifier.assert_called_once_with(
        ["4.P.20.409/2023/4"], top_k=1, per_token=True
    )


def test_apply_top_k_with_guarantees_caps_guaranteed_chunks_at_top_k():
    """Regression for the identifier-rescue flooding bug (docs/decisions.md):
    embedding a document's identifier into every one of its chunks means a
    single cited case number can make search_by_identifier() match an
    entire document's ~20+ chunks, all "guaranteed." Letting every one
    through used to flood the LLM's context with repetitive content from
    one document -- the guaranteed set must be capped at top_k like
    everything else, keeping the highest-reranked ones first."""
    # 6 guaranteed chunks (ids 1-6, already in descending-score order) and
    # 2 non-guaranteed chunks (ids 7-8) scoring higher than some guaranteed
    # ones -- but guarantee still wins priority within the top_k budget.
    guaranteed_chunks = [_chunk(i, score=1.0 - i * 0.01) for i in range(1, 7)]
    rest_chunks = [_chunk(7, score=0.99), _chunk(8, score=0.5)]
    chunks = guaranteed_chunks + rest_chunks
    guaranteed_ids = {c.id for c in guaranteed_chunks}

    result = _apply_top_k_with_guarantees(chunks, guaranteed_ids, top_k=4)

    assert len(result) == 4
    assert [c.id for c in result] == [1, 2, 3, 4]


def test_apply_top_k_with_guarantees_fills_remaining_budget_with_non_guaranteed():
    """When there are fewer guaranteed chunks than top_k, the remaining
    budget is still filled from the highest-scoring non-guaranteed chunks,
    same as before this fix."""
    chunks = [_chunk(99, score=1.0), _chunk(1, score=0.9), _chunk(2, score=0.8)]

    result = _apply_top_k_with_guarantees(chunks, {99}, top_k=2)

    assert [c.id for c in result] == [99, 1]


def _chunk_with_hub(chunk_id, score, hub_score):
    from dataclasses import replace as _replace

    base = _chunk(chunk_id, score=score)
    return _replace(base, metadata=_replace(base.metadata, hub_score=hub_score))


def test_csls_rerank_promotes_less_generic_chunk_above_higher_raw_score():
    """Regression for the real near-duplicate-dilution case (docs/decisions.md):
    a chunk with a lower raw score but a *much* lower hub_score (i.e. it's
    less generic/central in the embedding space) must be promoted above a
    chunk with a higher raw score but a very high (generic) hub_score."""
    chunks = [
        _chunk_with_hub(1, score=0.87, hub_score=0.98),  # generic, high raw score
        _chunk_with_hub(2, score=0.84, hub_score=0.90),  # less generic
    ]

    result = _csls_rerank(chunks)

    assert [c.id for c in result] == [2, 1]


def test_csls_rerank_falls_back_to_raw_score_without_hub_score():
    """A chunk with no hub_score yet (compute_hub_scores() hasn't run, or
    it's a brand new chunk) must fall back to its raw score, not crash or
    get pushed to the bottom."""
    chunks = [_chunk(1, score=0.5), _chunk(2, score=0.9)]

    result = _csls_rerank(chunks)

    assert [c.id for c in result] == [2, 1]


def test_hybrid_strategy_guarantees_identifier_match_survives_top_k_truncation(
    monkeypatch,
):
    """Regression test for a real, live miss: an identifier match can be
    correctly merged into the candidate pool and still pass RERANKER_MIN_SCORE,
    yet still get cut by top_k if several other chunks score higher on
    generic semantic relevance (see docs/decisions.md -- the reranker has
    no notion that an identifier match is definitionally correct)."""
    vector_results = [_chunk(i, score=0.9) for i in range(1, 5)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    fake_store.search_by_identifier.return_value = [_chunk(99, score=1.0)]

    class _LowScoringIdentifierReranker:
        """Scores the identifier match lowest of all -- it would be cut by
        a plain top_k truncation despite being merged into the pool."""

        def rerank(self, question, chunks):
            return sorted(
                (replace(c, score=0.1 if c.id == 99 else 1.0) for c in chunks),
                key=lambda c: c.score,
                reverse=True,
            )

    monkeypatch.setattr(
        retrieval_module,
        "get_reranker_driver",
        lambda *a, **k: _LowScoringIdentifierReranker(),
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "Mi történt a 4.P.20.409/2023/4. ügyben?",
        vector_results,
        fake_store,
        top_k=4,
        min_score=0.25,
    )

    assert 99 in [c.id for c in result]
    assert len(result) == 4


def test_hybrid_strategy_guarantees_identifier_match_survives_reranker_threshold(
    monkeypatch, settings_override
):
    """Regression test for a real, live miss: a compound multi-case question
    can make the cross-encoder score a definitionally-correct identifier
    match well below RERANKER_MIN_SCORE, since the chunk only reads as
    on-topic for *part* of the question (see docs/decisions.md). The
    threshold filter must not drop it before _apply_top_k_with_guarantees
    ever sees it."""
    monkeypatch.setattr(
        retrieval_module, "settings", settings_override(RERANKER_MIN_SCORE=-2.0)
    )
    vector_results = [_chunk(1, score=0.9)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    fake_store.search_by_identifier.return_value = [_chunk(99, score=1.0)]
    # The identifier match (99) is merged in ahead of the fused list (see
    # select_chunks), so rerank() receives [99, 1] -- 99 scores well below
    # threshold, candidate 1 scores above it.
    monkeypatch.setattr(
        retrieval_module,
        "get_reranker_driver",
        lambda *a, **k: _FakeCrossEncoderReranker([-4.3, 1.5]),
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "Mi történt a 4.P.20.409/2023/4. ügyben?",
        vector_results,
        fake_store,
        top_k=5,
        min_score=0.25,
    )

    assert 99 in [c.id for c in result]


def test_hybrid_strategy_skips_identifier_search_when_question_has_no_identifiers(
    monkeypatch,
):
    vector_results = [_chunk(1, score=0.9)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    monkeypatch.setattr(
        retrieval_module, "get_reranker_driver", lambda *a, **k: _NoopFakeReranker()
    )

    strategy = HybridRetrievalStrategy()
    strategy.select_chunks("q", vector_results, fake_store, top_k=5, min_score=0.25)

    fake_store.search_by_identifier.assert_not_called()


def test_hybrid_strategy_truncates_to_top_k_after_fusion(monkeypatch):
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.8)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    monkeypatch.setattr(
        retrieval_module, "get_reranker_driver", lambda *a, **k: _NoopFakeReranker()
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "q", vector_results, fake_store, top_k=1, min_score=0.25
    )

    assert len(result) == 1


def test_hybrid_strategy_cross_encoder_filters_low_scores(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        retrieval_module, "settings", settings_override(RERANKER_MIN_SCORE=0.0)
    )
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.8)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    # Candidate 1 gets score 1.5 (passes >= 0.0), Candidate 2 gets -2.0 (filtered out)
    monkeypatch.setattr(
        retrieval_module,
        "get_reranker_driver",
        lambda *a, **k: _FakeCrossEncoderReranker([1.5, -2.0]),
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "q", vector_results, fake_store, top_k=5, min_score=0.25
    )

    assert len(result) == 1
    assert result[0].id == 1
    assert result[0].score == 1.5


def test_hybrid_strategy_cross_encoder_rejects_when_all_below_threshold(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        retrieval_module, "settings", settings_override(RERANKER_MIN_SCORE=0.0)
    )
    vector_results = [_chunk(1, score=0.9), _chunk(2, score=0.8)]
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    # Both candidates score negative logits (e.g. unanswerable / irrelevant)
    monkeypatch.setattr(
        retrieval_module,
        "get_reranker_driver",
        lambda *a, **k: _FakeCrossEncoderReranker([-3.5, -7.2]),
    )

    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "q", vector_results, fake_store, top_k=5, min_score=0.25
    )

    assert result == []


def test_get_retrieval_strategy_returns_hybrid_by_default(
    monkeypatch, settings_override
):
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

    fake_driver.embed_query.assert_not_called()
    fake_store.search.assert_called_once()
    assert fake_store.search.call_args.args[0] == [0.1, 0.2]


def test_retrieve_chunks_embeds_when_no_query_vector_given(monkeypatch):
    fake_driver, fake_store = _patch_driver_and_store(monkeypatch)
    fake_driver.embed_query.return_value = [0.9, 0.9]

    retrieve_chunks("question")

    fake_driver.embed_query.assert_called_once_with("question")
    assert fake_store.search.call_args.args[0] == [0.9, 0.9]


def test_retrieve_chunks_forwards_metadata_filter(monkeypatch):
    fake_driver, fake_store = _patch_driver_and_store(
        monkeypatch, search_results=[_chunk(1, score=0.9)]
    )
    fake_driver.embed_query.return_value = [0.1, 0.2]

    filter_dict = {"source_file": "notes.md"}
    # Explicit VectorRetrievalStrategy: this test is only about
    # metadata_filter plumbing, so it shouldn't depend on whatever
    # RETRIEVAL_STRATEGY/RERANKER_DRIVER real settings happen to be
    # configured (e.g. a real CrossEncoderRerankerDriver would load an
    # actual model and score this fake chunk unpredictably).
    results = retrieve_chunks(
        "question", metadata_filter=filter_dict, strategy=VectorRetrievalStrategy()
    )

    assert len(results) == 1
    assert fake_store.search.call_args.kwargs.get("metadata_filter") == filter_dict


def test_hybrid_strategy_skips_listwise_rerank_when_disabled(monkeypatch):
    """Default (LISTWISE_RERANK_ENABLED=False): must not call the answer
    driver at all -- it's an opt-in, extra-LLM-call feature."""
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    monkeypatch.setattr(
        retrieval_module, "get_reranker_driver", lambda *a, **k: _NoopFakeReranker()
    )
    fake_get_answer_driver = MagicMock()
    monkeypatch.setattr(retrieval_module, "get_answer_driver", fake_get_answer_driver)

    strategy = HybridRetrievalStrategy()
    strategy.select_chunks(
        "question", [_chunk(1, score=0.9)], fake_store, top_k=4, min_score=0.25
    )

    fake_get_answer_driver.assert_not_called()


def test_hybrid_strategy_applies_listwise_rerank_when_enabled(
    monkeypatch, settings_override
):
    """LISTWISE_RERANK_ENABLED=True: the final order must reflect
    listwise_rerank()'s reordering, applied right before the top_k cut."""
    monkeypatch.setattr(
        retrieval_module,
        "settings",
        settings_override(
            LISTWISE_RERANK_ENABLED=True, LISTWISE_RERANK_MAX_CANDIDATES=20
        ),
    )
    fake_store = MagicMock()
    fake_store.search_fulltext.return_value = []
    monkeypatch.setattr(
        retrieval_module, "get_reranker_driver", lambda *a, **k: _NoopFakeReranker()
    )
    monkeypatch.setattr(retrieval_module, "get_answer_driver", lambda: MagicMock())
    fake_listwise_rerank = MagicMock(
        side_effect=lambda q, chunks, driver, **kw: chunks[::-1]
    )
    monkeypatch.setattr(retrieval_module, "listwise_rerank", fake_listwise_rerank)

    vector_results = [
        _chunk(1, score=0.9, source="a.pdf"),
        _chunk(2, score=0.8, source="b.pdf"),
    ]
    strategy = HybridRetrievalStrategy()
    result = strategy.select_chunks(
        "question", vector_results, fake_store, top_k=4, min_score=0.25
    )

    fake_listwise_rerank.assert_called_once()
    assert [c.id for c in result] == [2, 1]


def test_round_robin_by_document_takes_turns_across_documents():
    """Regression for q0018: one long document's chunks must not take every
    slot while a second named document gets none."""
    chunks = [
        _chunk(1, 0.9, "a.docx"),
        _chunk(2, 0.8, "a.docx"),
        _chunk(3, 0.7, "a.docx"),
        _chunk(4, 0.6, "b.docx"),
    ]

    result = _round_robin_by_document(chunks, limit=3)

    assert [c.id for c in result] == [1, 4, 2]


def test_round_robin_by_document_returns_everything_when_under_limit():
    chunks = [_chunk(1, 0.9, "a.docx"), _chunk(2, 0.8, "b.docx")]
    assert [c.id for c in _round_robin_by_document(chunks, limit=4)] == [1, 2]


def test_apply_top_k_with_guarantees_diversify_represents_every_guaranteed_document():
    chunks = [_chunk(i, 1.0 - i / 10, "a.docx") for i in range(1, 5)] + [
        _chunk(9, 0.1, "b.docx")
    ]
    guaranteed_ids = {1, 2, 3, 4, 9}

    default = _apply_top_k_with_guarantees(chunks, guaranteed_ids, top_k=4)
    diversified = _apply_top_k_with_guarantees(
        chunks, guaranteed_ids, top_k=4, diversify=True
    )

    assert {c.metadata.source_file for c in default} == {"a.docx"}
    assert {c.metadata.source_file for c in diversified} == {"a.docx", "b.docx"}
    assert len(diversified) == 4


def test_merge_unique_keeps_primary_order_and_drops_duplicate_ids():
    primary = [_chunk(1), _chunk(2)]
    extra = [_chunk(2), _chunk(3)]

    assert [c.id for c in _merge_unique(primary, extra)] == [1, 2, 3]


def _dated_chunk(chunk_id, score, source, date):
    chunk = _chunk(chunk_id, score, source)
    return replace(chunk, metadata=replace(chunk.metadata, document_date=date))


def test_apply_top_k_reserves_half_the_slots_for_the_questions_years():
    """Regression for the soft period widening: the reranker knows nothing
    about dates, so off-year chunks kept the whole top_k."""
    chunks = [
        _dated_chunk(1, 0.9, "a.docx", "2019-01-01"),
        _dated_chunk(2, 0.8, "b.docx", "2019-02-01"),
        _dated_chunk(3, 0.7, "c.docx", "2018-03-01"),
        _dated_chunk(4, 0.6, "d.docx", "2017-04-01"),
        _dated_chunk(5, 0.5, "e.docx", "2022-05-01"),
        _dated_chunk(6, 0.4, "f.docx", "2022-06-01"),
    ]

    result = _apply_top_k_with_guarantees(chunks, set(), top_k=4, years=[2022])

    in_period = [
        c.id for c in result if (c.metadata.document_date or "").startswith("2022")
    ]
    assert sorted(in_period) == [5, 6]
    assert len(result) == 4


def test_apply_top_k_period_reservation_changes_nothing_without_in_period_chunks():
    chunks = [
        _dated_chunk(i, 1.0 - i / 10, "a.docx", "2019-01-01") for i in range(1, 7)
    ]

    with_years = _apply_top_k_with_guarantees(chunks, set(), top_k=4, years=[2022])
    without = _apply_top_k_with_guarantees(chunks, set(), top_k=4)

    assert [c.id for c in with_years] == [c.id for c in without]


def test_apply_top_k_period_reservation_counts_guaranteed_chunks_already_in_period():
    chunks = [
        _dated_chunk(1, 0.9, "a.docx", "2022-01-01"),
        _dated_chunk(2, 0.8, "b.docx", "2022-02-01"),
        _dated_chunk(3, 0.7, "c.docx", "2019-03-01"),
        _dated_chunk(4, 0.6, "d.docx", "2019-04-01"),
        _dated_chunk(5, 0.5, "e.docx", "2022-05-01"),
    ]

    result = _apply_top_k_with_guarantees(chunks, {1, 2}, top_k=4, years=[2022])

    # two guaranteed in-period chunks already meet the reservation of 2, so the
    # remaining slots follow plain score order, not more in-period chunks.
    assert [c.id for c in result] == [1, 2, 3, 4]


def test_apply_top_k_period_reservation_spreads_across_documents_when_diversifying():
    chunks = [
        _dated_chunk(1, 0.9, "x.docx", "2019-01-01"),
        _dated_chunk(2, 0.8, "x.docx", "2019-01-01"),
        _dated_chunk(3, 0.7, "a.docx", "2022-01-01"),
        _dated_chunk(4, 0.6, "a.docx", "2022-01-01"),
        _dated_chunk(5, 0.5, "b.docx", "2022-02-01"),
    ]

    result = _apply_top_k_with_guarantees(
        chunks, set(), top_k=4, diversify=True, years=[2022]
    )

    reserved = {c.metadata.source_file for c in result if c.id in {3, 4, 5}}
    assert reserved == {"a.docx", "b.docx"}

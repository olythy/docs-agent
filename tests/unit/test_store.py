"""Tests for store.VectorStore (no real DB required — connection mocked)."""

from unittest.mock import MagicMock

import pytest

import store
from models import Chunk, ChunkMetadata
from store import VectorStore, _to_pgvector_literal


@pytest.fixture(autouse=True)
def _reset_dimension_cache():
    VectorStore.clear_dimension_cache()
    yield
    VectorStore.clear_dimension_cache()


def test_to_pgvector_literal_formats_as_bracketed_csv():
    assert _to_pgvector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def _fake_conn_with_cursor(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn


def test_save_inserts_one_row_per_chunk(monkeypatch):
    cursor = MagicMock()
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    chunks = [
        Chunk(
            content="hello",
            metadata=ChunkMetadata(source_file="doc.pdf", page_number=1, chunk_index=0),
        ),
        Chunk(
            content="world",
            metadata=ChunkMetadata(source_file="doc.pdf", page_number=2, chunk_index=1),
        ),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    inserted = VectorStore().save(chunks, embeddings)

    assert inserted == 2
    assert cursor.execute.call_count == 2
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


def test_save_raises_on_mismatched_lengths(monkeypatch):
    conn = _fake_conn_with_cursor(MagicMock())
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    chunks = [
        Chunk(
            content="hello",
            metadata=ChunkMetadata(source_file="doc.pdf", page_number=1, chunk_index=0),
        )
    ]
    embeddings = []  # length mismatch vs. chunks

    with pytest.raises(ValueError, match="zip"):
        VectorStore().save(chunks, embeddings)


def test_search_filters_by_min_score_and_parses_json_metadata(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            1,
            "above threshold",
            '{"source_file": "doc.pdf", "page_number": 1, "chunk_index": 0}',
            0.9,
        ),
        (
            2,
            "below threshold",
            '{"source_file": "doc.pdf", "page_number": 2, "chunk_index": 1}',
            0.1,
        ),
    ]
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    results = VectorStore().search([0.1, 0.2], top_k=5, min_score=0.5)

    assert len(results) == 1
    assert results[0].id == 1
    assert results[0].content == "above threshold"
    assert results[0].metadata == ChunkMetadata(
        source_file="doc.pdf", page_number=1, chunk_index=0
    )
    assert results[0].score == 0.9
    conn.close.assert_called_once()


def test_search_fulltext_parses_json_metadata(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            7,
            "Player Central tennis booking",
            '{"source_file": "doc.pdf", "page_number": 1, "chunk_index": 0}',
            0.42,
        ),
    ]
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    results = VectorStore().search_fulltext("tennis booking", top_k=5)

    assert len(results) == 1
    assert results[0].id == 7
    assert results[0].content == "Player Central tennis booking"
    assert results[0].metadata == ChunkMetadata(
        source_file="doc.pdf", page_number=1, chunk_index=0
    )
    assert results[0].score == 0.42
    conn.close.assert_called_once()


def test_search_fulltext_or_joins_query_words(monkeypatch):
    """A raw natural-language question must filter stopwords and OR-join words.

    Stopwords and non-technical short words are filtered out before OR-joining,
    preventing common grammatical filler words from dominating rankings.
    """
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    VectorStore().search_fulltext("What tennis system is this?", top_k=5)

    query_arg, top_k_arg = (
        cursor.execute.call_args[0][1][0],
        cursor.execute.call_args[0][1][2],
    )
    assert query_arg == "tennis or system"
    assert top_k_arg == 5


def test_prepare_fulltext_query_filters_stopwords_and_short_words():
    query, kept, dropped = store.prepare_fulltext_query("What tennis system is this?")
    assert query == "tennis or system"
    assert kept == ["tennis", "system"]
    assert "What" in dropped
    assert "is" in dropped
    assert "this" in dropped


def test_prepare_fulltext_query_preserves_short_tech_terms():
    query, kept, dropped = store.prepare_fulltext_query("AI pipeline and UI in Python")
    assert query == "AI or pipeline or UI or Python"
    assert "AI" in kept
    assert "UI" in kept
    assert "pipeline" in kept
    assert "Python" in kept
    assert "and" in dropped
    assert "in" in dropped


def test_prepare_fulltext_query_fallback_when_all_stopwords():
    query, kept, dropped = store.prepare_fulltext_query("Who is it?")
    assert query == "Who or is or it"
    assert kept == ["Who", "is", "it"]
    assert dropped == []


def test_extract_identifier_tokens_finds_case_number():
    q = "Mi volt a per tárgya a Budapest Környéki Törvényszék 4.P.20.409/2023/4. számú ügyében?"
    assert store.extract_identifier_tokens(q) == ["4.P.20.409/2023/4"]


def test_extract_identifier_tokens_finds_letter_digit_mix():
    assert store.extract_identifier_tokens("A HU001 számla kifizetve") == ["HU001"]


def test_extract_identifier_tokens_finds_long_pure_number_but_not_a_year():
    tokens = store.extract_identifier_tokens("A 202300471 számú tétel és a 2023-as év")
    assert tokens == ["202300471"]


def test_extract_identifier_tokens_ignores_plain_words_and_short_numbers():
    assert store.extract_identifier_tokens("Ez egy sima mondat 42 szóval") == []


def test_extract_identifier_tokens_ignores_inflected_hungarian_numbers():
    """Regression test for a real, live false positive: "2020-as"/"2022-es"
    (Hungarian "of 2020"/"of 2022") matched the digit+separator rule,
    flooding VectorStore.search_by_identifier's result limit with common
    year mentions before the real case numbers in the same question could
    appear (see docs/decisions.md)."""
    q = "a 2020-as G.40110.2020.29. számú és a 2022-es K.700650.2022.13. számú ügyekben"
    tokens = store.extract_identifier_tokens(q)
    assert "2020-as" not in tokens
    assert "2022-es" not in tokens
    assert "G.40110.2020.29" in tokens
    assert "K.700650.2022.13" in tokens


def test_extract_identifier_tokens_deduplicates():
    q = "4.P.20.409/2023/4 majd újra 4.P.20.409/2023/4"
    assert store.extract_identifier_tokens(q) == ["4.P.20.409/2023/4"]


def test_search_by_identifier_returns_empty_list_without_a_db_call_for_no_tokens():
    assert VectorStore().search_by_identifier([], top_k=5) == []


def test_search_by_identifier_builds_ored_ilike_query(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (1, "... 4.P.20.409/2023/4 ...", {"source_file": "a.docx", "chunk_index": 0})
    ]
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    results = VectorStore().search_by_identifier(
        ["4.P.20.409/2023/4", "HU001"], top_k=5
    )

    assert len(results) == 1
    assert results[0].metadata.source_file == "a.docx"
    assert results[0].score == 1.0

    sql_executed = cursor.execute.call_args[0][0]
    params_executed = cursor.execute.call_args[0][1]
    assert sql_executed.count("content ILIKE %s") == 2
    assert params_executed == ("%4.P.20.409/2023/4%", "%HU001%", 5)


def test_search_fulltext_returns_empty_list_for_no_matches(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    assert VectorStore().search_fulltext("nothing matches this", top_k=5) == []


def test_has_chunks_from_source_returns_true_when_found(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    assert VectorStore().has_chunks_from_source("sample.pdf") is True
    sql_executed = cursor.execute.call_args[0][0]
    args_executed = cursor.execute.call_args[0][1]
    assert "WHERE metadata->>'source_path' = %s" in sql_executed
    assert "OR metadata->>'source_file' = %s" in sql_executed
    assert args_executed == ("sample.pdf", "sample.pdf")


def test_has_chunks_from_source_returns_false_when_missing(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    assert VectorStore().has_chunks_from_source("missing.pdf") is False


def test_get_embedding_dimension_returns_atttypmod(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = (384,)
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    assert VectorStore().get_embedding_dimension() == 384


def test_get_embedding_dimension_returns_none_when_table_missing(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    assert VectorStore().get_embedding_dimension() is None


def test_assert_dimension_matches_passes_when_equal(monkeypatch):
    monkeypatch.setattr(VectorStore, "get_embedding_dimension", lambda self: 384)
    VectorStore().assert_dimension_matches(384)  # must not raise


def test_assert_dimension_matches_passes_when_table_missing(monkeypatch):
    monkeypatch.setattr(VectorStore, "get_embedding_dimension", lambda self: None)
    VectorStore().assert_dimension_matches(1536)  # must not raise


def test_assert_dimension_matches_raises_on_mismatch(monkeypatch):
    monkeypatch.setattr(VectorStore, "get_embedding_dimension", lambda self: 384)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        VectorStore().assert_dimension_matches(1536)


def test_search_with_metadata_filter(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    VectorStore().search(
        [0.1, 0.2], top_k=3, min_score=0.0, metadata_filter={"source_file": "doc.md"}
    )

    sql_executed = cursor.execute.call_args[0][0]
    args_executed = cursor.execute.call_args[0][1]

    assert "WHERE metadata @> %s::jsonb" in sql_executed
    assert '{"source_file": "doc.md"}' in args_executed


def test_search_fulltext_with_metadata_filter(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    VectorStore().search_fulltext(
        "tennis", top_k=3, metadata_filter={"source_file": "doc.md"}
    )

    sql_executed = cursor.execute.call_args[0][0]
    args_executed = cursor.execute.call_args[0][1]

    assert "AND metadata @> %s::jsonb" in sql_executed
    assert '{"source_file": "doc.md"}' in args_executed


def test_vector_store_reuses_provided_connection():
    cursor = MagicMock()
    conn = _fake_conn_with_cursor(cursor)
    custom_store = VectorStore(conn=conn)

    chunks = [
        Chunk(
            content="hello",
            metadata=ChunkMetadata(source_file="doc.pdf", page_number=1, chunk_index=0),
        )
    ]
    embeddings = [[0.1, 0.2]]

    custom_store.save(chunks, embeddings)

    assert cursor.execute.call_count == 1
    conn.commit.assert_called_once()
    # When an external connection is provided, VectorStore must NOT close it
    conn.close.assert_not_called()


def test_delete_chunks_by_hash(monkeypatch):
    cursor = MagicMock()
    cursor.rowcount = 4
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    test_hash = "f" * 64
    deleted = VectorStore().delete_chunks_by_hash(test_hash)

    assert deleted == 4
    sql_executed = cursor.execute.call_args[0][0]
    args_executed = cursor.execute.call_args[0][1]
    assert (
        "DELETE FROM document_chunks WHERE metadata->>'content_hash' = %s"
        in sql_executed
    )
    assert args_executed == (test_hash,)
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


def test_delete_chunks_from_source_delegates_to_hash(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = ("a" * 64,)
    cursor.rowcount = 3
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    deleted = VectorStore().delete_chunks_from_source("sample.pdf")

    assert deleted == 3
    # Two queries: 1 to look up hash, 1 to delete by hash
    assert cursor.execute.call_count == 2
    assert (
        "DELETE FROM document_chunks WHERE metadata->>'content_hash' = %s"
        in cursor.execute.call_args[0][0]
    )


def test_has_content_hash(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    exists = VectorStore().has_content_hash("b" * 64)
    assert exists is True
    assert "metadata->>'content_hash' = %s" in cursor.execute.call_args[0][0]


def test_add_source_alias(monkeypatch):
    cursor = MagicMock()
    cursor.rowcount = 2
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    updated = VectorStore().add_source_alias("c" * 64, "copy.md")
    assert updated == 2
    assert "jsonb_set" in cursor.execute.call_args[0][0]
    conn.commit.assert_called_once()


def test_vector_store_context_manager_reuses_single_connection(monkeypatch):
    cursor = MagicMock()
    conn = _fake_conn_with_cursor(cursor)
    get_conn_mock = MagicMock(return_value=conn)
    monkeypatch.setattr(store, "get_connection", get_conn_mock)

    store_instance = VectorStore()
    assert store_instance._conn is None

    with store_instance:
        assert store_instance._conn is conn
        get_conn_mock.assert_called_once()

        # Perform multiple operations on the same instance
        store_instance.has_content_hash("a" * 64)
        store_instance.has_content_hash("b" * 64)

        # Still only one connection opened
        get_conn_mock.assert_called_once()
        conn.close.assert_not_called()

    # Closed upon exiting the context
    conn.close.assert_called_once()
    assert store_instance._conn is None


def test_vector_store_context_manager_nested_reentrancy(monkeypatch):
    cursor = MagicMock()
    conn = _fake_conn_with_cursor(cursor)
    get_conn_mock = MagicMock(return_value=conn)
    monkeypatch.setattr(store, "get_connection", get_conn_mock)

    store_instance = VectorStore()
    with store_instance:
        # Outer scope
        assert store_instance._conn_depth == 1
        with store_instance:
            # Inner scope (e.g. add_directory -> add_document)
            assert store_instance._conn_depth == 2
            conn.close.assert_not_called()
        # Exited inner scope
        assert store_instance._conn_depth == 1
        conn.close.assert_not_called()

    # Exited outer scope
    assert store_instance._conn_depth == 0
    conn.close.assert_called_once()


def test_assert_dimension_matches_caches_successful_check(monkeypatch):
    get_dim_mock = MagicMock(return_value=384)
    monkeypatch.setattr(VectorStore, "get_embedding_dimension", get_dim_mock)

    s = VectorStore()
    s.assert_dimension_matches(384)
    assert get_dim_mock.call_count == 1

    # Second call for the same dimension must hit cache and NOT query DB
    s.assert_dimension_matches(384)
    assert get_dim_mock.call_count == 1

    # Another instance also benefits from the cache
    VectorStore().assert_dimension_matches(384)
    assert get_dim_mock.call_count == 1

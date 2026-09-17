"""Tests for store.VectorStore (no real DB required — connection mocked)."""

from unittest.mock import MagicMock

import pytest

import store
from store import VectorStore, _to_pgvector_literal


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
        {"content": "hello", "metadata": {"page_number": 1}},
        {"content": "world", "metadata": {"page_number": 2}},
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

    chunks = [{"content": "hello", "metadata": {}}]
    embeddings = []  # length mismatch vs. chunks

    with pytest.raises(ValueError, match="zip"):
        VectorStore().save(chunks, embeddings)


def test_search_filters_by_min_score_and_parses_json_metadata(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("above threshold", '{"page_number": 1}', 0.9),
        ("below threshold", '{"page_number": 2}', 0.1),
    ]
    conn = _fake_conn_with_cursor(cursor)
    monkeypatch.setattr(store, "get_connection", lambda: conn)

    results = VectorStore().search([0.1, 0.2], top_k=5, min_score=0.5)

    assert len(results) == 1
    assert results[0]["content"] == "above threshold"
    assert results[0]["metadata"] == {"page_number": 1}
    assert results[0]["score"] == 0.9
    conn.close.assert_called_once()


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

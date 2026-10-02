"""Tests for db.get_connection (no real DB required — psycopg2.connect mocked)."""

from unittest.mock import MagicMock

import pytest

import db


def test_get_connection_raises_without_database_url(monkeypatch, settings_override):
    monkeypatch.setattr(db, "settings", settings_override(DATABASE_URL=""))

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.get_connection()


def test_get_connection_calls_psycopg2_connect_with_database_url(
    monkeypatch, settings_override
):
    calls = []
    fake_conn = MagicMock()
    monkeypatch.setattr(
        db.psycopg2,
        "connect",
        lambda url: calls.append(url) or fake_conn,
    )
    monkeypatch.setattr(
        db, "settings", settings_override(DATABASE_URL="postgresql://example")
    )

    result = db.get_connection()

    assert calls == ["postgresql://example"]
    assert result is fake_conn


def test_get_connection_sets_hnsw_ef_search(monkeypatch, settings_override):
    """Regression test for a real, live gap: pgvector's own hnsw.ef_search
    default (40) silently caps how many rows a vector query can return,
    regardless of the SQL LIMIT requested, once the corpus grows large
    enough -- every connection must set this explicitly."""
    fake_conn = MagicMock()
    monkeypatch.setattr(db.psycopg2, "connect", lambda url: fake_conn)
    monkeypatch.setattr(
        db,
        "settings",
        settings_override(DATABASE_URL="postgresql://example", HNSW_EF_SEARCH=500),
    )

    db.get_connection()

    fake_cursor = fake_conn.cursor.return_value.__enter__.return_value
    fake_cursor.execute.assert_called_once_with(
        "SET hnsw.ef_search = %s", (500,)
    )
    fake_conn.commit.assert_called_once()

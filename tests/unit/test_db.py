"""Tests for db.get_connection (no real DB required — psycopg2.connect mocked)."""

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
    monkeypatch.setattr(db.psycopg2, "connect", lambda url: calls.append(url))
    monkeypatch.setattr(
        db, "settings", settings_override(DATABASE_URL="postgresql://example")
    )

    db.get_connection()

    assert calls == ["postgresql://example"]

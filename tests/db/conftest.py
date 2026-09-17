"""Fixtures for tests that require a real Postgres + pgvector database.

Every test module under ``tests/db/`` gates itself with a module-level
``pytestmark = [pytest.mark.db, pytest.mark.skipif(not TEST_DATABASE_URL, ...)]``
so these tests report as SKIPPED (not failed, not silently absent) until
``TEST_DATABASE_URL`` points at a disposable Postgres+pgvector instance.
"""

import os

import psycopg2
import pytest

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.fixture
def db_conn():
    """Real psycopg2 connection to TEST_DATABASE_URL, truncated after each test."""
    conn = psycopg2.connect(TEST_DATABASE_URL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("TRUNCATE document_chunks RESTART IDENTITY;")
    conn.commit()
    conn.close()

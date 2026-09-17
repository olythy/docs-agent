"""Fixtures for tests that require a real Postgres + pgvector database.

Every test module under ``tests/db/`` gates itself with a module-level
``pytestmark = [pytest.mark.db, pytest.mark.skipif(settings.AGENT_ENV != "test", ...)]``
so these tests report as SKIPPED (not failed, not silently absent) unless
run with ``AGENT_ENV=test`` (``make test`` sets this automatically) and a
``.env.test`` file exists pointing at a disposable Postgres+pgvector
instance — see ``config.py`` and ``.env.test.example``.

No monkeypatching needed here: under ``AGENT_ENV=test``, ``settings.DATABASE_URL``
(and therefore ``db.get_connection()``) already correctly points at the test
database for the whole process, since config.py loads ``.env.test`` on top
of ``.env`` at import time.

The schema itself is ensured by ``_ensure_test_schema`` below (session-scoped,
autouse) rather than by a Makefile prerequisite — this way the test database
is always ready regardless of *how* pytest gets invoked (``make test``, a
bare ``AGENT_ENV=test uv run pytest``, or an IDE's "run test" button, which
typically calls pytest directly, bypassing Make entirely).
"""

import psycopg2
import pytest

from config import settings
from db import get_connection
from scripts.migrate import cmd_up, ensure_migrations_table


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_schema():
    """Migrate, then unconditionally empty, the test database once per session.

    Reuses scripts.migrate's own functions (not a reimplementation) for the
    migration step — idempotent, prints "Nothing to migrate" and does
    nothing if already applied.

    Truncates *at the start* of the session, not at the end. A prior "drop
    at teardown" design was rejected: if a test run is killed mid-test
    (Ctrl+C, crash), teardown never executes, and the next session would
    silently start from whatever half-finished state was left behind —
    confirmed by deliberately leaving a stray row and re-running: a
    completely unrelated test failed on an assertion about row content,
    not because of a real bug, but because of contamination from the
    earlier run. Truncating at the *start* has no such blind spot: it
    never depends on a previous session having exited cleanly, since it
    only ever looks at, and clears, the state that exists right now.

    Guarded explicitly on AGENT_ENV, even though every test in this package
    is already skipped when it isn't 'test' (and pytest never sets up a
    skipped test's fixtures) — cheap defense-in-depth against ever touching
    settings.DATABASE_URL when it might be a real dev/shared database.
    """
    if settings.AGENT_ENV != "test":
        return
    conn = get_connection()
    try:
        ensure_migrations_table(conn)
        cmd_up(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE document_chunks RESTART IDENTITY;")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def db_conn():
    """Real psycopg2 connection to settings.DATABASE_URL, truncated after each test."""
    conn = psycopg2.connect(settings.DATABASE_URL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("TRUNCATE document_chunks RESTART IDENTITY;")
    conn.commit()
    conn.close()

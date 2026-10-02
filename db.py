"""Postgres connection factory.

The single, canonical way to open a connection to the database configured
via ``settings.DATABASE_URL``. Deliberately does nothing else — chunk
persistence lives in ``store.py``, migration structure in
``migrations/base.py`` (see AGENTS.md's "Design philosophy" section for why
these are kept separate).
"""

import psycopg2
from psycopg2.extensions import connection as PgConnection

from config import settings


def get_connection() -> PgConnection:
    """Open and return a new Postgres connection using settings.DATABASE_URL.

    Sets ``hnsw.ef_search`` for the session (see ``settings.HNSW_EF_SEARCH``'s
    docstring) -- confirmed live that pgvector's own default (40) silently
    caps how many rows a vector query can return, independent of the SQL
    ``LIMIT`` requested, once the corpus grows large enough. A session-level
    ``SET`` here, rather than per-query, since every vector query on this
    connection should use the same value.

    Returns:
        A ``psycopg2`` connection object, with ``hnsw.ef_search`` already set.

    Raises:
        RuntimeError: If ``DATABASE_URL`` is not configured.
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set DATABASE_URL in your .env file."
        )
    conn = psycopg2.connect(settings.DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (settings.HNSW_EF_SEARCH,))
    conn.commit()
    return conn

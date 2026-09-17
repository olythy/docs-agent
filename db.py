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

    Returns:
        A ``psycopg2`` connection object.

    Raises:
        RuntimeError: If ``DATABASE_URL`` is not configured.
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set DATABASE_URL in your .env file."
        )
    return psycopg2.connect(settings.DATABASE_URL)

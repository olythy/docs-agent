"""Persistence for the structured-metadata layer: documents, key catalog, values, status.

The counterpart of :class:`store.VectorStore` (which owns ``document_chunks``):
this class owns *all* SQL for the ``documents``, ``meta_keys``,
``document_meta`` and ``document_meta_status`` tables, so no orchestrator ever
builds SQL for them. See docs/structured-metadata-design.md.

Key exports:
    DocumentStore -- The data-access class.
    SyncResult    -- What :meth:`DocumentStore.sync_from_chunks` changed.
"""

import json
from dataclasses import dataclass
from typing import Self

from connection_scope import ConnectionScope
from db import get_connection
from models import (
    Document,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaState,
    MetaStatus,
    MetaValue,
    ValueType,
)


@dataclass(frozen=True)
class SyncResult:
    """The outcome of syncing ``documents`` with the ingested chunks.

    Attributes:
        upserted: Documents inserted or refreshed from their chunks.
        removed: Documents deleted because no chunk carries their hash any more
            (their extracted values and status rows go with them, by cascade).
    """

    upserted: int
    removed: int


class DocumentStore:
    """Data access for documents, the key catalog, extracted values and their status.

    Can be handed an existing connection to reuse, used as a context manager to
    share one connection across several calls, or used bare, in which case
    every call opens and closes its own short-lived connection (the same
    contract as :class:`store.VectorStore`).

    Args:
        conn: Optional active connection. If provided, the caller closes it.
    """

    def __init__(self, conn=None) -> None:
        # The factory is looked up at call time so tests can patch
        # ``document_store.get_connection``.
        self._scope = ConnectionScope(conn, connect=lambda: get_connection())

    def __enter__(self) -> Self:
        """Open a reusable connection for the duration of the ``with`` block."""
        self._scope.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the connection if this store opened it and this is the outermost block."""
        self._scope.exit()

    # ------------------------------------------------------------- documents

    def upsert_document(self, document: Document) -> None:
        """Insert a document, or refresh its file name and summary if it exists.

        Args:
            document: The document. ``ingested_at`` is set by the database on
                insert and left untouched on update.
        """
        sql = """
            INSERT INTO documents (content_hash, source_file, summary)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash) DO UPDATE
            SET source_file = EXCLUDED.source_file, summary = EXCLUDED.summary;
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (document.content_hash, document.source_file, document.summary)
                )
            conn.commit()

    def get_document(self, content_hash: str) -> Document | None:
        """Return the document with this hash, or ``None``."""
        sql = """
            SELECT content_hash, source_file, summary, ingested_at
            FROM documents WHERE content_hash = %s;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (content_hash,))
            row = cur.fetchone()
        return None if row is None else Document(*row)

    def count_documents(self) -> int:
        """Return how many documents are registered."""
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents;")
            row = cur.fetchone()
        assert row is not None  # an aggregate query always returns one row
        return row[0]

    def delete_document(self, content_hash: str) -> int:
        """Delete a document; its values and status rows are removed by cascade.

        Returns:
            The number of documents deleted (0 or 1).
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE content_hash = %s;", (content_hash,)
                )
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def sync_from_chunks(self) -> SyncResult:
        """Make ``documents`` match the ingested chunks, in both directions.

        Registers (or refreshes) one document per distinct chunk
        ``content_hash`` -- taking the file name and the summary from its first
        chunk -- and removes documents whose chunks are gone, which is what a
        re-ingest that replaced a document leaves behind. Idempotent.

        Returns:
            How many documents were upserted and how many removed.
        """
        upsert_sql = """
            INSERT INTO documents (content_hash, source_file, summary)
            SELECT DISTINCT ON (content_hash)
                   content_hash,
                   metadata->>'source_file',
                   metadata->>'document_summary'
            FROM document_chunks
            WHERE content_hash IS NOT NULL
            ORDER BY content_hash, id
            ON CONFLICT (content_hash) DO UPDATE
            SET source_file = EXCLUDED.source_file, summary = EXCLUDED.summary;
        """
        remove_sql = """
            DELETE FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM document_chunks c WHERE c.content_hash = d.content_hash
            );
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(upsert_sql)
                upserted = cur.rowcount
                cur.execute(remove_sql)
                removed = cur.rowcount
            conn.commit()
        return SyncResult(upserted=upserted, removed=removed)

    # --------------------------------------------------------------- catalog

    def upsert_key(self, key: MetaKey) -> None:
        """Insert a catalog key, or replace its definition if (doc_type, key) exists."""
        sql = """
            INSERT INTO meta_keys
                (doc_type, key, value_type, description, example,
                 allowed_values, multi_valued, status, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_type, key) DO UPDATE SET
                value_type = EXCLUDED.value_type,
                description = EXCLUDED.description,
                example = EXCLUDED.example,
                allowed_values = EXCLUDED.allowed_values,
                multi_valued = EXCLUDED.multi_valued,
                status = EXCLUDED.status,
                version = EXCLUDED.version;
        """
        allowed = None if key.allowed_values is None else list(key.allowed_values)
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        key.doc_type,
                        key.key,
                        key.value_type.value,
                        key.description,
                        key.example,
                        allowed,
                        key.multi_valued,
                        key.status.value,
                        key.version,
                    ),
                )
            conn.commit()

    def list_keys(
        self, doc_type: str, status: KeyStatus | None = None
    ) -> list[MetaKey]:
        """Return a document type's catalog keys, optionally only those with a status.

        Args:
            doc_type: The document type.
            status: If given, only keys in this lifecycle state.
        """
        sql = """
            SELECT doc_type, key, value_type, description, example,
                   allowed_values, multi_valued, status, version
            FROM meta_keys
            WHERE doc_type = %s AND (%s::text IS NULL OR status = %s)
            ORDER BY key;
        """
        wanted = None if status is None else status.value
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (doc_type, wanted, wanted))
            rows = cur.fetchall()
        return [
            MetaKey(
                doc_type=r[0],
                key=r[1],
                value_type=ValueType(r[2]),
                description=r[3],
                example=r[4],
                allowed_values=None if r[5] is None else tuple(r[5]),
                multi_valued=r[6],
                status=KeyStatus(r[7]),
                version=r[8],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- values

    def add_value(self, value: MetaValue) -> None:
        """Insert one extracted value.

        Raises:
            psycopg2.errors.CheckViolation: If not exactly one ``value_*`` field
                is set.
            psycopg2.errors.ForeignKeyViolation: If the document is not registered.
            psycopg2.errors.UniqueViolation: If the (document, key, ordinal) row exists.
        """
        sql = """
            INSERT INTO document_meta
                (content_hash, key, key_version, value_text, value_number,
                 value_date, value_bool, unit, ordinal, qualifiers, evidence,
                 evidence_chunk_index, page, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s);
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        value.content_hash,
                        value.key,
                        value.key_version,
                        value.value_text,
                        value.value_number,
                        value.value_date,
                        value.value_bool,
                        value.unit,
                        value.ordinal,
                        json.dumps(value.qualifiers),
                        value.evidence,
                        value.evidence_chunk_index,
                        value.page,
                        value.source.value,
                    ),
                )
            conn.commit()

    def get_values(self, content_hash: str, key: str) -> list[MetaValue]:
        """Return a document's values for one key, in ordinal order."""
        sql = """
            SELECT content_hash, key, key_version, source, value_text, value_number,
                   value_date, value_bool, unit, ordinal, qualifiers, evidence,
                   evidence_chunk_index, page
            FROM document_meta
            WHERE content_hash = %s AND key = %s
            ORDER BY ordinal;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (content_hash, key))
            rows = cur.fetchall()
        return [
            MetaValue(
                content_hash=r[0],
                key=r[1],
                key_version=r[2],
                source=MetaSource(r[3]),
                value_text=r[4],
                value_number=r[5],
                value_date=r[6],
                value_bool=r[7],
                unit=r[8],
                ordinal=r[9],
                qualifiers=r[10],
                evidence=r[11],
                evidence_chunk_index=r[12],
                page=r[13],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- status

    def set_status(self, status: MetaStatus) -> None:
        """Record (or replace) what is known about a (document, key) pair."""
        sql = """
            INSERT INTO document_meta_status (content_hash, key, state, key_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (content_hash, key) DO UPDATE SET
                state = EXCLUDED.state,
                key_version = EXCLUDED.key_version,
                attempted_at = now();
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        status.content_hash,
                        status.key,
                        status.state.value,
                        status.key_version,
                    ),
                )
            conn.commit()

    def get_status(self, content_hash: str, key: str) -> MetaStatus:
        """Return what is known about a (document, key) pair.

        A missing row means no attempt was made, which is reported as
        :attr:`models.MetaState.NOT_ATTEMPTED`, not as an error.
        """
        sql = """
            SELECT state, key_version FROM document_meta_status
            WHERE content_hash = %s AND key = %s;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (content_hash, key))
            row = cur.fetchone()
        if row is None:
            return MetaStatus(content_hash, key, MetaState.NOT_ATTEMPTED, key_version=0)
        return MetaStatus(content_hash, key, MetaState(row[0]), key_version=row[1])

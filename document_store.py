"""Persistence for the structured-metadata layer: documents, key catalog, values, status.

The counterpart of :class:`store.VectorStore` (which owns ``document_chunks``):
this class owns *all* SQL for the ``documents``, ``meta_keys``,
``document_meta`` and ``document_meta_status`` tables, so no orchestrator ever
builds SQL for them. See docs/structured-metadata-design.md.

Key exports:
    DocumentStore -- The data-access class.
    KeyCoverage   -- Per-key counts of documents by state (for the coverage report).
"""

import json
from dataclasses import dataclass
from typing import Self

import psycopg2.errors

from connection_scope import ConnectionScope
from db import get_connection
from models import (
    Document,
    DocumentType,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaState,
    MetaStatus,
    MetaValue,
    TypeStatus,
    ValueType,
)


def _to_type(row: tuple) -> DocumentType:
    """Build a :class:`DocumentType` from a ``document_types`` row."""
    return DocumentType(row[0], row[1], row[2], TypeStatus(row[3]))


@dataclass(frozen=True)
class KeyCoverage:
    """How many documents are in each :class:`models.MetaState` for one key.

    ``not_attempted`` counts documents with no status row at all, so the four
    numbers always add up to ``total``.
    """

    key: str
    present: int
    confirmed_absent: int
    unverified: int
    not_attempted: int
    total: int

    @property
    def unknown(self) -> int:
        """Documents a count over this key cannot account for."""
        return self.unverified + self.not_attempted


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

    def upsert_document(self, document: Document) -> int:
        """Insert a document, or refresh its file name and summary if it exists.

        Args:
            document: The document. ``ingested_at`` is set by the database on
                insert and left untouched on update.

        Returns:
            The document's numeric id (stable across refreshes).
        """
        sql = """
            INSERT INTO documents (content_hash, source_file, summary)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash) DO UPDATE
            SET source_file = EXCLUDED.source_file, summary = EXCLUDED.summary
            RETURNING id;
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (document.content_hash, document.source_file, document.summary)
                )
                row = cur.fetchone()
            conn.commit()
        assert row is not None  # RETURNING always yields the row
        return row[0]

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

    def remove_documents_without_chunks(self) -> int:
        """Remove documents that have no chunks, and return how many.

        A chunk cannot exist without its document (``document_chunks.document_id``
        is ``NOT NULL`` and cascades), but a document can be left without chunks:
        an ingest registers the document before it saves the chunks, so one that
        failed in between leaves a bare row. Their values and statuses go with
        them, by cascade. Idempotent.
        """
        sql = """
            DELETE FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM document_chunks c WHERE c.document_id = d.id
            );
        """
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                removed = cur.rowcount
            conn.commit()
        return removed

    # ------------------------------------------------------ classification

    def unclassified_documents(
        self, limit: int | None = None, seed: int | None = None
    ) -> list[Document]:
        """Return documents that have no document type yet.

        Args:
            limit: Return at most this many.
            seed: If given, the documents are taken in a random order fixed by this
                seed (a representative, reproducible trial sample); otherwise in
                file-name order.
        """
        sql = """
            SELECT d.content_hash, d.source_file, d.summary, d.ingested_at
            FROM documents d
            WHERE d.document_type IS NULL
            ORDER BY CASE WHEN %s::text IS NULL THEN d.source_file
                          ELSE md5(%s::text || d.content_hash) END
            LIMIT %s;
        """
        seed_text = None if seed is None else str(seed)
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (seed_text, seed_text, limit))
            return [Document(*row) for row in cur.fetchall()]

    def set_document_type(self, content_hash: str, type_name: str) -> bool:
        """Give one document its type. Returns whether the document exists.

        Raises:
            ValueError: If the type is not registered.
        """
        sql = "UPDATE documents SET document_type = %s WHERE content_hash = %s;"
        with self._scope.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (type_name, content_hash))
                    changed = cur.rowcount == 1
            except psycopg2.errors.ForeignKeyViolation:
                conn.rollback()
                raise ValueError(f"unknown document type {type_name!r}") from None
            conn.commit()
        return changed

    def assign_type_to_unclassified(self, type_name: str) -> int:
        """Give every document that has no type this type. Returns how many.

        A manual shortcut for a corpus that has only one kind of document; the
        classifier is the general route.

        Raises:
            ValueError: If the type is not registered.
        """
        sql = "UPDATE documents SET document_type = %s WHERE document_type IS NULL;"
        with self._scope.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (type_name,))
                    changed = cur.rowcount
            except psycopg2.errors.ForeignKeyViolation:
                conn.rollback()
                raise ValueError(f"unknown document type {type_name!r}") from None
            conn.commit()
        return changed

    def count_by_type(self) -> dict[str | None, int]:
        """Return how many documents each type has; the key ``None`` is "no type yet"."""
        sql = "SELECT document_type, count(*) FROM documents GROUP BY 1;"
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            return {row[0]: row[1] for row in cur.fetchall()}

    # --------------------------------------------------------------- catalog

    def upsert_type(self, doc_type: DocumentType) -> None:
        """Insert a document type, or replace its name, description and status."""
        sql = """
            INSERT INTO document_types (type, name, description, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (type) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                status = EXCLUDED.status;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    doc_type.type,
                    doc_type.name,
                    doc_type.description,
                    doc_type.status.value,
                ),
            )
            conn.commit()

    def ensure_type(self, type_name: str) -> None:
        """Make sure a type exists, registering it as ``approved`` if it does not.

        A bridge for catalogs that only name their type (the multi-type catalog
        format carries a name and description and uses :meth:`upsert_type`). An
        existing type is left untouched.
        """
        sql = """
            INSERT INTO document_types (type, name, description, status)
            VALUES (%s, %s, '', 'approved')
            ON CONFLICT (type) DO NOTHING;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (type_name, type_name))
            conn.commit()

    def get_type(self, type_name: str) -> DocumentType | None:
        """Return one document type, or ``None`` if it is not registered."""
        sql = "SELECT type, name, description, status FROM document_types WHERE type = %s;"
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (type_name,))
            row = cur.fetchone()
        return None if row is None else _to_type(row)

    def list_types(self, status: TypeStatus | None = None) -> list[DocumentType]:
        """Return the registered types, optionally only those with one status."""
        sql = "SELECT type, name, description, status FROM document_types"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = %s"
            params = (status.value,)
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql + " ORDER BY type;", params)
            return [_to_type(row) for row in cur.fetchall()]

    def set_type_status(self, type_name: str, status: TypeStatus) -> bool:
        """Approve, retire or re-propose a type. Returns whether it exists."""
        sql = "UPDATE document_types SET status = %s WHERE type = %s;"
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (status.value, type_name))
            changed = cur.rowcount == 1
            conn.commit()
        return changed

    def upsert_key(self, key: MetaKey) -> None:
        """Insert a catalog key, or replace its definition if (doc_type, key) exists.

        Raises:
            ValueError: If the key's document type is not registered (register it
                first with :meth:`upsert_type` or :meth:`ensure_type`).
        """
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
        params = (
            key.doc_type,
            key.key,
            key.value_type.value,
            key.description,
            key.example,
            allowed,
            key.multi_valued,
            key.status.value,
            key.version,
        )
        with self._scope.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            except psycopg2.errors.ForeignKeyViolation:
                conn.rollback()
                raise ValueError(
                    f"unknown document type {key.doc_type!r}; register it first "
                    "(upsert_type or ensure_type)"
                ) from None
            conn.commit()

    def set_key_status(self, doc_type: str, key: str, status: KeyStatus) -> bool:
        """Change a catalog key's lifecycle status (approve a proposal, retire a key).

        Returns:
            Whether the key exists.
        """
        sql = "UPDATE meta_keys SET status = %s WHERE doc_type = %s AND key = %s;"
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status.value, doc_type, key))
                found = cur.rowcount == 1
            conn.commit()
        return found

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
            ValueError: If the document is not registered.
            psycopg2.errors.UniqueViolation: If the (document, key, ordinal) row exists.
        """
        with self._scope.connection() as conn:
            self._insert_value(conn, value)
            conn.commit()

    @staticmethod
    def _insert_value(conn, value: MetaValue) -> None:
        """Insert one value row on ``conn`` without committing.

        Raises:
            ValueError: If the document is not registered.
        """
        sql = """
            INSERT INTO document_meta
                (document_id, key, key_version, value_text,
                 value_number, value_date, value_bool, unit, ordinal, qualifiers,
                 evidence, evidence_chunk_index, page, source)
            VALUES ((SELECT id FROM documents WHERE content_hash = %s),
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s);
        """
        params = (
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
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        except psycopg2.errors.NotNullViolation:
            conn.rollback()
            raise ValueError(
                f"document {value.content_hash[:8]} is not registered"
            ) from None

    def replace_values(
        self, content_hash: str, key: str, values: list[MetaValue]
    ) -> None:
        """Replace all of a document's values for one key, atomically.

        Re-extracting a key (because its definition changed, or an earlier run
        failed) must not leave the old rows behind or insert next to them.

        Args:
            content_hash: The document.
            key: The catalog key.
            values: The new rows (may be empty, which just clears the key).
        """
        delete_sql = (
            "DELETE FROM document_meta WHERE key = %s AND document_id = "
            "(SELECT id FROM documents WHERE content_hash = %s);"
        )
        with self._scope.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_sql, (key, content_hash))
            for value in values:
                self._insert_value(conn, value)
            conn.commit()

    def get_statuses(self, content_hash: str) -> dict[str, MetaStatus]:
        """Return a document's recorded status per key (keys never attempted are absent)."""
        sql = """
            SELECT s.key, s.state, s.key_version
            FROM document_meta_status s JOIN documents d ON d.id = s.document_id
            WHERE d.content_hash = %s;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (content_hash,))
            rows = cur.fetchall()
        return {
            key: MetaStatus(content_hash, key, MetaState(state), key_version=version)
            for key, state, version in rows
        }

    def documents_needing(
        self, keys: list[MetaKey], limit: int | None = None, seed: int | None = None
    ) -> list[Document]:
        """Return documents for which at least one of ``keys`` still has to be extracted.

        A key is *done* for a document when a status row exists at the key's
        current version or later; a missing row (never attempted) or one from an
        older definition counts as pending. This is what makes extraction
        resumable and lets a changed definition refresh only what it affects.

        Args:
            keys: The catalog keys to consider (their ``version`` is the bar).
            limit: Return at most this many documents.
            seed: If given, the documents are taken in a random order that is
                fixed by this seed (so a trial sample is representative yet
                reproducible); otherwise in file-name order.
        """
        if not keys:
            return []
        sql = """
            SELECT d.content_hash, d.source_file, d.summary, d.ingested_at
            FROM documents d
            WHERE EXISTS (
                SELECT 1 FROM unnest(%s::text[], %s::int[]) AS k(key, version)
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_meta_status s
                    WHERE s.document_id = d.id
                      AND s.key = k.key AND s.key_version >= k.version
                )
            )
            ORDER BY CASE WHEN %s::text IS NULL THEN d.source_file
                          ELSE md5(%s::text || d.content_hash) END
            LIMIT %s;
        """
        seed_text = None if seed is None else str(seed)
        params = (
            [k.key for k in keys],
            [k.version for k in keys],
            seed_text,
            seed_text,
            limit,
        )
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [Document(*row) for row in rows]

    def distinct_text_values(self, key: str, limit: int) -> list[str] | None:
        """Return every distinct stored text value of a key, if there are few enough.

        Lets a planner see the exact spellings a text filter has to match (a
        combined court name is one value, not two).

        Args:
            key: The catalog key.
            limit: The most distinct values worth listing.

        Returns:
            The values, most frequent first; ``None`` when there are more than
            ``limit`` (the key is too free-form to list).
        """
        sql = """
            SELECT value_text FROM document_meta
            WHERE key = %s AND value_text IS NOT NULL
            GROUP BY value_text ORDER BY count(*) DESC, value_text LIMIT %s;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (key, limit + 1))
            rows = [row[0] for row in cur.fetchall()]
        return rows if len(rows) <= limit else None

    def execute_query(
        self, sql: str, params: tuple, timeout_ms: int = 10_000
    ) -> list[tuple]:
        """Run a compiled, parameterised read-only query and return its rows.

        The SQL comes from :class:`metadata.compiler.PlanCompiler`, which only ever
        puts bound parameters and fixed fragments in it. As defence in depth the
        statement runs in a read-only transaction with a time limit, so even a
        compiler bug could neither modify data nor hold the database for long.

        Args:
            sql: The query, with ``%s`` placeholders.
            params: Its parameters.
            timeout_ms: Abort the statement after this many milliseconds.

        Returns:
            The result rows.
        """
        with self._scope.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            finally:
                conn.rollback()  # nothing to keep; also ends the read-only transaction
        return rows

    def coverage(self, keys: list[MetaKey]) -> list[KeyCoverage]:
        """Count, per key, how many documents are in each state.

        Only a status row at the key's *current* version counts; an older one is
        treated as not attempted, because the definition has changed since.

        Args:
            keys: The catalog keys to report on.
        """
        total = self.count_documents()
        sql = """
            SELECT state, count(*) FROM document_meta_status
            WHERE key = %s AND key_version >= %s
            GROUP BY state;
        """
        result = []
        with self._scope.connection() as conn, conn.cursor() as cur:
            for key in keys:
                cur.execute(sql, (key.key, key.version))
                counts = {MetaState(state): n for state, n in cur.fetchall()}
                recorded = sum(counts.values())
                result.append(
                    KeyCoverage(
                        key=key.key,
                        present=counts.get(MetaState.PRESENT, 0),
                        confirmed_absent=counts.get(MetaState.CONFIRMED_ABSENT, 0),
                        unverified=counts.get(MetaState.UNVERIFIED, 0),
                        not_attempted=total
                        - recorded
                        + counts.get(MetaState.NOT_ATTEMPTED, 0),
                        total=total,
                    )
                )
        return result

    def list_values(self, key: str) -> list[tuple[str, MetaValue]]:
        """Return every stored value of one key, with its document's file name.

        Meant for reports and measurement over a whole corpus (it reads every row
        of the key), not for per-document lookups.

        Args:
            key: The catalog key.

        Returns:
            ``(source_file, value)`` pairs ordered by file name and ordinal.
        """
        sql = """
            SELECT d.source_file, d.content_hash, m.key, m.key_version, m.source,
                   m.value_text, m.value_number, m.value_date, m.value_bool, m.unit,
                   m.ordinal, m.qualifiers, m.evidence, m.evidence_chunk_index, m.page
            FROM document_meta m JOIN documents d ON d.id = m.document_id
            WHERE m.key = %s
            ORDER BY d.source_file, m.ordinal;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (key,))
            rows = cur.fetchall()
        return [
            (
                r[0],
                MetaValue(
                    content_hash=r[1],
                    key=r[2],
                    key_version=r[3],
                    source=MetaSource(r[4]),
                    value_text=r[5],
                    value_number=r[6],
                    value_date=r[7],
                    value_bool=r[8],
                    unit=r[9],
                    ordinal=r[10],
                    qualifiers=r[11],
                    evidence=r[12],
                    evidence_chunk_index=r[13],
                    page=r[14],
                ),
            )
            for r in rows
        ]

    def get_values(self, content_hash: str, key: str) -> list[MetaValue]:
        """Return a document's values for one key, in ordinal order."""
        sql = """
            SELECT d.content_hash, m.key, m.key_version, m.source, m.value_text,
                   m.value_number, m.value_date, m.value_bool, m.unit, m.ordinal,
                   m.qualifiers, m.evidence, m.evidence_chunk_index, m.page
            FROM document_meta m JOIN documents d ON d.id = m.document_id
            WHERE d.content_hash = %s AND m.key = %s
            ORDER BY m.ordinal;
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
        """Record (or replace) what is known about a (document, key) pair.

        Raises:
            ValueError: If the document is not registered.
        """
        sql = """
            INSERT INTO document_meta_status (document_id, key, state, key_version)
            VALUES ((SELECT id FROM documents WHERE content_hash = %s), %s, %s, %s)
            ON CONFLICT (document_id, key) DO UPDATE SET
                state = EXCLUDED.state,
                key_version = EXCLUDED.key_version,
                attempted_at = now();
        """
        with self._scope.connection() as conn:
            try:
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
            except psycopg2.errors.NotNullViolation:
                conn.rollback()
                raise ValueError(
                    f"document {status.content_hash[:8]} is not registered"
                ) from None
            conn.commit()

    def get_status(self, content_hash: str, key: str) -> MetaStatus:
        """Return what is known about a (document, key) pair.

        A missing row means no attempt was made, which is reported as
        :attr:`models.MetaState.NOT_ATTEMPTED`, not as an error.
        """
        sql = """
            SELECT s.state, s.key_version
            FROM document_meta_status s JOIN documents d ON d.id = s.document_id
            WHERE d.content_hash = %s AND s.key = %s;
        """
        with self._scope.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (content_hash, key))
            row = cur.fetchone()
        if row is None:
            return MetaStatus(content_hash, key, MetaState.NOT_ATTEMPTED, key_version=0)
        return MetaStatus(content_hash, key, MetaState(row[0]), key_version=row[1])

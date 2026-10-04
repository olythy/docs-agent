"""DB-gated tests for VectorStore's document restriction (``content_hashes``).

A restricted store must only ever see the chunks of the given documents, through
every search path, because the restriction is what keeps a normal retrieval
inside the set a structured filter selected.
"""

import json

import pytest

from config import settings
from drivers.embedding import get_embedding_driver
from store import VectorStore, _to_pgvector_literal

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        settings.AGENT_ENV != "test",
        reason="AGENT_ENV is not 'test' -- see tests/db/conftest.py.",
    ),
]


@pytest.fixture
def two_documents(db_conn):
    driver = get_embedding_driver()
    for name, text in (
        ("a", "booking invoice Pfv.100"),
        ("b", "booking invoice Pfv.200"),
    ):
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO document_chunks (content, metadata, embedding) VALUES (%s, %s, %s);",
                (
                    text,
                    json.dumps(
                        {
                            "source_file": f"{name}.pdf",
                            "page_number": 1,
                            "chunk_index": 0,
                            "content_hash": f"hash-{name}",
                        }
                    ),
                    _to_pgvector_literal(driver.embed_text(text)),
                ),
            )
    db_conn.commit()
    return driver


def _files(chunks):
    return sorted(c.metadata.source_file for c in chunks)


def test_vector_search_only_sees_the_given_documents(two_documents):
    vec = two_documents.embed_text("booking invoice")

    both = VectorStore().search(vec, top_k=5, min_score=0.0)
    only_b = VectorStore(content_hashes=["hash-b"]).search(vec, top_k=5, min_score=0.0)

    assert _files(both) == ["a.pdf", "b.pdf"]
    assert _files(only_b) == ["b.pdf"]


def test_fulltext_search_only_sees_the_given_documents(two_documents):
    only_a = VectorStore(content_hashes=["hash-a"]).search_fulltext("booking", top_k=5)

    assert _files(only_a) == ["a.pdf"]


@pytest.mark.parametrize("per_token", [False, True])
def test_identifier_search_only_sees_the_given_documents(two_documents, per_token):
    store = VectorStore(content_hashes=["hash-a"])

    inside = store.search_by_identifier(["Pfv.100"], top_k=5, per_token=per_token)
    outside = store.search_by_identifier(["Pfv.200"], top_k=5, per_token=per_token)

    assert _files(inside) == ["a.pdf"]
    assert outside == []


def test_an_empty_restriction_sees_nothing_not_everything(two_documents):
    vec = two_documents.embed_text("booking invoice")
    store = VectorStore(content_hashes=[])

    assert store.search(vec, top_k=5, min_score=0.0) == []
    assert store.search_fulltext("booking", top_k=5) == []
    assert store.search_by_identifier(["Pfv.100"], top_k=5) == []

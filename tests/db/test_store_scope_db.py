"""DB-gated tests for VectorStore's document restriction (a ``DocumentSelection``).

A restricted store must only ever see the chunks of the selected documents, through
every search path, because the restriction is what keeps a normal retrieval inside
the set a structured filter selected. The selection is a *sub-select* the database
evaluates, not a list of documents, so it has no size cap.
"""

import json

import pytest

from config import settings
from drivers.embedding import get_embedding_driver
from models import DocumentSelection
from store import VectorStore, _to_pgvector_literal

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        settings.AGENT_ENV != "test",
        reason="AGENT_ENV is not 'test' -- see tests/db/conftest.py.",
    ),
]


def _only(content_hash: str) -> DocumentSelection:
    """A selection of the one document with this hash."""
    return DocumentSelection(
        "SELECT d.id FROM documents d WHERE d.content_hash = %s", (content_hash,)
    )


@pytest.fixture
def two_documents(db_conn):
    driver = get_embedding_driver()
    for name, text in (
        ("a", "booking invoice Pfv.100"),
        ("b", "booking invoice Pfv.200"),
    ):
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (content_hash, source_file) VALUES (%s, %s) "
                "RETURNING id;",
                (f"hash-{name}", f"{name}.pdf"),
            )
            document_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO document_chunks (content, metadata, embedding, document_id) "
                "VALUES (%s, %s, %s, %s);",
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
                    document_id,
                ),
            )
    db_conn.commit()
    return driver


def _files(chunks):
    return sorted(c.metadata.source_file for c in chunks)


def test_vector_search_only_sees_the_selected_documents(two_documents):
    vec = two_documents.embed_text("booking invoice")

    both = VectorStore().search(vec, top_k=5, min_score=0.0)
    only_b = VectorStore(selection=_only("hash-b")).search(vec, top_k=5, min_score=0.0)

    assert _files(both) == ["a.pdf", "b.pdf"]
    assert _files(only_b) == ["b.pdf"]


def test_fulltext_search_only_sees_the_selected_documents(two_documents):
    only_a = VectorStore(selection=_only("hash-a")).search_fulltext("booking", top_k=5)

    assert _files(only_a) == ["a.pdf"]


@pytest.mark.parametrize("per_token", [False, True])
def test_identifier_search_only_sees_the_selected_documents(two_documents, per_token):
    store = VectorStore(selection=_only("hash-a"))

    inside = store.search_by_identifier(["Pfv.100"], top_k=5, per_token=per_token)
    outside = store.search_by_identifier(["Pfv.200"], top_k=5, per_token=per_token)

    assert _files(inside) == ["a.pdf"]
    assert outside == []


def test_a_selection_that_matches_nothing_sees_nothing_not_everything(two_documents):
    vec = two_documents.embed_text("booking invoice")
    store = VectorStore(
        selection=DocumentSelection("SELECT d.id FROM documents d WHERE FALSE")
    )

    assert store.search(vec, top_k=5, min_score=0.0) == []
    assert store.search_fulltext("booking", top_k=5) == []
    assert store.search_by_identifier(["Pfv.100"], top_k=5) == []


def test_a_selection_with_several_bound_parameters_keeps_them_in_order(two_documents):
    """The sub-select's parameters sit between the other conditions' parameters."""
    vec = two_documents.embed_text("booking invoice")
    selection = DocumentSelection(
        "SELECT d.id FROM documents d WHERE d.source_file = %s OR d.content_hash = %s",
        ("a.pdf", "hash-b"),
    )
    store = VectorStore(selection=selection)

    found = store.search(
        vec, top_k=5, min_score=0.0, metadata_filter={"page_number": 1}
    )

    assert _files(found) == ["a.pdf", "b.pdf"]
    assert _files(store.search_fulltext("booking", top_k=5)) == ["a.pdf", "b.pdf"]


def test_a_selection_by_document_ids_restricts_every_search(two_documents, db_conn):
    """The scope of a question that names its documents by an identifier is a list of ids
    in an ``ANY(%s)``: the array parameter must travel through every search path."""
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE source_file = 'a.pdf';")
        a_id = cur.fetchone()[0]
    vec = two_documents.embed_text("booking invoice")
    store = VectorStore(
        selection=DocumentSelection(
            "SELECT id FROM documents WHERE id = ANY(%s)", ([a_id],)
        )
    )

    assert _files(store.search(vec, top_k=5, min_score=0.0)) == ["a.pdf"]
    assert _files(store.search_fulltext("booking", top_k=5)) == ["a.pdf"]
    assert _files(store.search_by_identifier(["Pfv.100"], top_k=5)) == ["a.pdf"]
    assert store.search_by_identifier(["Pfv.200"], top_k=5) == []


def test_a_retrieval_service_restricts_a_real_store_to_the_scope(
    two_documents, db_conn
):
    """End to end on the real database: the scope a question's identifiers resolve to."""
    from config import settings
    from drivers.reranker import NoopRerankerDriver
    from query.decision import Scope
    from query.facts import QueryFactsReader
    from query.outcome import Answerable
    from query.profiles import PipelineFactory, ProfileResolver
    from query.service import RetrievalRequest, RetrievalService

    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE source_file = 'b.pdf';")
        b_id = cur.fetchone()[0]
    embedding = two_documents
    service = RetrievalService(
        QueryFactsReader(),
        ProfileResolver(settings),
        PipelineFactory(embedding, NoopRerankerDriver(), lambda q, chunks: chunks),
        embedding,
    )
    vector = embedding.embed_query("booking invoice")

    def retrieve(scope):
        result = service.retrieve(
            RetrievalRequest(
                "booking invoice",
                profile="vector",
                query_vector=vector,
                min_score=0.0,
                scope=scope,
            ),
            VectorStore(),
        )
        assert isinstance(result.outcome, Answerable)
        return _files(list(result.outcome.chunks))

    everywhere = retrieve(None)
    inside = retrieve(
        Scope(
            selection=DocumentSelection(
                "SELECT id FROM documents WHERE id = ANY(%s)", ([b_id],)
            )
        )
    )

    assert everywhere == ["a.pdf", "b.pdf"]
    assert inside == ["b.pdf"]


def test_the_scope_of_a_source_file_restricts_every_search_to_that_file(two_documents):
    from query.decision import scope_of_source_file

    selection = scope_of_source_file("b.pdf").selection
    assert selection is not None
    vec = two_documents.embed_text("booking invoice")

    only_b = VectorStore(selection=selection).search(vec, top_k=5, min_score=0.0)
    unknown = VectorStore(selection=scope_of_source_file("zzz.pdf").selection).search(
        vec, top_k=5, min_score=0.0
    )

    assert _files(only_b) == ["b.pdf"]
    assert unknown == []  # a file that was never ingested selects nothing

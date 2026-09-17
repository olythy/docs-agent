"""DB-gated tests for VectorStore.search() and query_knowledge_base() end-to-end.

Requires AGENT_ENV=test (and a .env.test) — see tests/db/conftest.py and
config.py. The embedding driver is real (local, already cached, no
network); the LLM driver is stubbed to avoid a real API call/cost.
"""

import json

import pytest

import query.retrieval as retrieval_module
from config import settings
from drivers.embedding import get_embedding_driver
from query.retrieval import NO_RESULTS_MESSAGE, query_knowledge_base
from store import VectorStore, _to_pgvector_literal

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        settings.AGENT_ENV != "test",
        reason=(
            "AGENT_ENV is not 'test' — no test database is configured. "
            "These tests are written and ready; run with AGENT_ENV=test "
            "(e.g. `make test`, which sets this automatically) and a "
            ".env.test file (see .env.test.example) to run them."
        ),
    ),
]


def _insert_chunk(db_conn, content, embedding, source_file="t.pdf", page=1):
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document_chunks (content, metadata, embedding)
            VALUES (%s, %s, %s);
            """,
            (
                content,
                json.dumps({"source_file": source_file, "page_number": page}),
                _to_pgvector_literal(embedding),
            ),
        )
    db_conn.commit()


def test_search_orders_results_by_similarity(db_conn):
    driver = get_embedding_driver()
    close_text = "Paris is the capital of France."
    far_text = "Bananas are yellow fruit."

    _insert_chunk(db_conn, close_text, driver.embed_text(close_text))
    _insert_chunk(db_conn, far_text, driver.embed_text(far_text))

    query_vec = driver.embed_text("What is the capital of France?")
    results = VectorStore().search(query_vec, top_k=5, min_score=0.0)

    assert results[0]["content"] == close_text


def test_search_filters_below_min_score(db_conn):
    driver = get_embedding_driver()
    _insert_chunk(
        db_conn, "Bananas are yellow fruit.", driver.embed_text("Bananas are yellow fruit.")
    )

    query_vec = driver.embed_text("What is the capital of France?")
    results = VectorStore().search(query_vec, top_k=5, min_score=0.99)

    assert results == []


def test_get_embedding_dimension_matches_real_column(db_conn):
    assert VectorStore().get_embedding_dimension() == 384


class _StubAnswerDriver:
    """Avoids a real LLM API call — records what it was asked, returns a canned reply."""

    def __init__(self):
        self.received_chunks = None

    def answer(self, question, context_chunks):
        self.received_chunks = context_chunks
        return "STUBBED ANSWER"


def test_query_knowledge_base_end_to_end_with_stubbed_llm(db_conn, monkeypatch):
    driver = get_embedding_driver()
    _insert_chunk(db_conn, "The sky is blue.", driver.embed_text("The sky is blue."))

    stub = _StubAnswerDriver()
    monkeypatch.setattr(retrieval_module, "get_answer_driver", lambda: stub)

    answer = query_knowledge_base("What color is the sky?", min_score=0.0)

    assert answer == "STUBBED ANSWER"
    assert stub.received_chunks is not None
    assert len(stub.received_chunks) >= 1


def test_query_knowledge_base_returns_fallback_when_nothing_matches(db_conn, monkeypatch):
    stub = _StubAnswerDriver()
    monkeypatch.setattr(retrieval_module, "get_answer_driver", lambda: stub)

    answer = query_knowledge_base("Anything?", min_score=0.99)

    assert answer == NO_RESULTS_MESSAGE
    assert stub.received_chunks is None  # answer() was never called

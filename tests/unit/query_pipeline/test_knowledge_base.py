"""The agent-facing entry point: it builds the service from the settings and asks it."""

import query.knowledge_base as knowledge_base_module
from models import ChunkMetadata, RetrievedChunk
from query.decision import Scope, scope_of_source_file
from query.knowledge_base import query_knowledge_base
from query.outcome import Answerable, Declined, DeclineReason
from query.profiles import DEFAULT_PROFILE
from query.service import RetrievalResult


class FakeService:
    def __init__(self):
        self.calls = []

    def answer(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return type("A", (), {"text": "THE TEXT"})()


def wired(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(
        knowledge_base_module, "build_query_service", lambda settings: service
    )
    return service


def test_it_returns_the_text_of_the_services_answer(monkeypatch):
    service = wired(monkeypatch)

    text = query_knowledge_base("q?", top_k=3, min_score=0.2)

    assert text == "THE TEXT"
    assert service.calls == [
        ("q?", {"top_k": 3, "min_score": 0.2, "scope": None, "store": None})
    ]


def test_a_source_file_becomes_the_scope(monkeypatch):
    service = wired(monkeypatch)

    query_knowledge_base("q?", source_file="a.docx")

    scope = service.calls[0][1]["scope"]
    assert isinstance(scope, Scope)
    assert scope == scope_of_source_file("a.docx")


class TestSearch:
    """The passages only: no decision, no answer."""

    def chunk(self, i):
        return RetrievedChunk(
            id=i,
            content="c",
            metadata=ChunkMetadata(source_file="a.pdf", page_number=1, chunk_index=i),
            score=0.5,
        )

    def wired(self, monkeypatch, outcome):
        requests = []

        class FakeRetrieval:
            def retrieve(self, request, store):
                requests.append(request)
                return RetrievalResult(outcome, ())

        monkeypatch.setattr(
            knowledge_base_module, "build_retrieval_service", lambda s: FakeRetrieval()
        )
        return requests

    def test_it_returns_the_chunks_read_with_the_configured_profile(self, monkeypatch):
        requests = self.wired(monkeypatch, Answerable((self.chunk(1), self.chunk(2))))

        chunks = knowledge_base_module.search_knowledge_base("q?")

        assert [c.id for c in chunks] == [1, 2]
        assert requests[0].profile == DEFAULT_PROFILE
        assert requests[0].scope is None  # all documents

    def test_a_source_file_becomes_the_scope(self, monkeypatch):
        requests = self.wired(monkeypatch, Answerable((self.chunk(1),)))

        knowledge_base_module.search_knowledge_base("q?", source_file="a.docx")

        assert requests[0].scope == scope_of_source_file("a.docx")

    def test_a_refusal_is_an_empty_list_not_an_error(self, monkeypatch):
        self.wired(monkeypatch, Declined(DeclineReason.NOT_RELEVANT, "relevance_gate"))

        assert knowledge_base_module.search_knowledge_base("q?") == []
